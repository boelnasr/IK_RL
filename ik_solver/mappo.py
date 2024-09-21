import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import logging
import json
import matplotlib.pyplot as plt
from collections import deque

# Assuming compute_combined_reward and TrainingMetrics are defined elsewhere
from .reward_function import compute_combined_reward
from .training_metrics import TrainingMetrics

class JointActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(JointActor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Use Tanh to bound actions between -1 and 1
        )
        # Use a single scalar log_std parameter for simplicity
        self.log_std = nn.Parameter(torch.zeros(1))

        # Initialize weights
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, state):
        action_mean = self.actor(state)  # Output is bounded between -1 and 1
        action_std = self.log_std.exp().expand_as(action_mean)
        # Ensure std is not too small
        action_std = torch.clamp(action_std, min=1e-3)
        return action_mean, action_std

class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(CentralizedCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize weights
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, states):
        return self.critic(states)

class MAPPOAgent:
    def __init__(self, env, config):
        self.env = env
        self.num_joints = env.num_joints  # Get the number of joints from the environment
        self.num_agents = env.num_joints  # Assuming one agent per joint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = config.get('hidden_dim', 128)
        self.lr = config.get('lr', 1e-4)  # Reduced learning rate for stability
        self.agents = []
        self.optimizers = []

        # Initialize gamma and tau
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.95)

        # Initialize observation dimensions based on the environment's observation space
        self.obs_dims = []
        for obs_space in env.observation_space:
            obs_dim = sum(np.prod(space.shape) for space in obs_space.spaces.values())
            self.obs_dims.append(obs_dim)

        # Initialize agents (actors) for each joint
        for obs_dim in self.obs_dims:
            actor = JointActor(obs_dim, self.hidden_dim, action_dim=1).to(self.device)
            optimizer = optim.Adam(actor.parameters(), lr=self.lr)
            self.agents.append(actor)
            self.optimizers.append(optimizer)

        # Initialize the centralized critic
        self.critic = CentralizedCritic(sum(self.obs_dims), self.hidden_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        # Other training parameters
        self.ppo_epochs = config.get('ppo_epochs', 15)  # Increased epochs
        self.batch_size = config.get('batch_size', 64)
        self.clip_param = config.get('clip_param', 0.1)  # Reduced clip parameter
        self.training_metrics = TrainingMetrics()

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('training.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def _process_state(self, state):
        processed_states = []
        for joint_state in state:  # Assuming state is a list of joint states
            obs = np.concatenate([
                joint_state['joint_angle'].flatten(),
                joint_state['position_error'].flatten(),
                joint_state['orientation_error'].flatten()
            ])
            # Normalize observations
            obs = (obs - np.mean(obs)) / (np.std(obs) + 1e-8)
            processed_states.append(torch.tensor(obs, dtype=torch.float32).to(self.device))
        return processed_states

    def get_actions(self, state):
        processed_states = self._process_state(state)
        actions = []
        log_probs = []
        for agent_idx, agent in enumerate(self.agents):
            state_tensor = processed_states[agent_idx].unsqueeze(0)
            with torch.no_grad():
                mean, std = agent(state_tensor)
            dist = Normal(mean, std)
            action = dist.sample()
            action = action.clamp(-1, 1)  # Ensure action bounds
            log_prob = dist.log_prob(action).sum(dim=-1)
            actions.append(action.squeeze().cpu().numpy().item())
            log_probs.append(log_prob.item())
        return actions, log_probs

    def compute_gae(self, rewards, dones, values):
        advantages = []
        returns = []
        gae = 0
        next_value = 0
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.gamma * next_value * mask - values[step]
            gae = delta + self.gamma * self.tau * mask * gae
            advantages.insert(0, gae)
            next_value = values[step]
            returns.insert(0, gae + values[step])
        advantages = torch.stack(advantages)
        returns = torch.stack(returns)
        return advantages.detach(), returns.detach()

    def update_policy(self, trajectories):
        # Unpack trajectories
        states = [torch.stack(t['states']) for t in trajectories]  # List of tensors [batch_size, obs_dim_i]
        actions = [torch.stack(t['actions']) for t in trajectories]  # List of tensors [batch_size]
        log_probs_old = [torch.stack(t['log_probs']) for t in trajectories]
        rewards = [torch.stack(t['rewards']) for t in trajectories]
        dones = torch.tensor(trajectories[0]['dones'], dtype=torch.float32).to(self.device)

        # Ensure all tensors have the same first dimension
        min_length = min(s.size(0) for s in states)
        states = [s[:min_length] for s in states]
        actions = [a[:min_length] for a in actions]
        log_probs_old = [lp[:min_length] for lp in log_probs_old]
        rewards = [r[:min_length] for r in rewards]
        dones = dones[:min_length]

        # Concatenate states for centralized critic input
        states_cat = torch.cat(states, dim=1)  # Shape: [batch_size, total_state_dim]
        actions_cat = torch.stack(actions, dim=1)  # Shape: [batch_size, num_agents]
        log_probs_old_cat = torch.stack(log_probs_old, dim=1)  # Shape: [batch_size, num_agents]

        # Compute values
        with torch.no_grad():
            values = self.critic(states_cat).squeeze()  # Shape: [batch_size]

        # Stack rewards and compute mean rewards across agents
        rewards_tensor = torch.stack(rewards, dim=1)  # Shape: [batch_size, num_agents]
        mean_rewards = rewards_tensor.mean(dim=1)     # Shape: [batch_size]

        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            rewards=mean_rewards,
            dones=dones,
            values=values
        )

        # Normalize advantages safely
        advantages_mean = advantages.mean()
        advantages_std = advantages.std()
        if advantages_std > 1e-6:
            advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
        else:
            self.logger.warning("Advantages not normalized due to small std")

        # Prepare data for training
        dataset = torch.utils.data.TensorDataset(
            states_cat, actions_cat, log_probs_old_cat, advantages, returns
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        for _ in range(self.ppo_epochs):
            for batch in loader:
                batch_states, batch_actions, batch_log_probs_old, batch_advantages, batch_returns = batch

                # Critic update
                values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(values, batch_returns)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()

                # Actor update
                for agent_idx, agent in enumerate(self.agents):
                    optimizer = self.optimizers[agent_idx]
                    agent_state_dim = self.obs_dims[agent_idx]

                    # Extract agent-specific data
                    start_idx = sum(self.obs_dims[:agent_idx])
                    end_idx = sum(self.obs_dims[:agent_idx+1])
                    agent_state = batch_states[:, start_idx:end_idx]
                    agent_action = batch_actions[:, agent_idx].unsqueeze(1)
                    agent_log_prob_old = batch_log_probs_old[:, agent_idx]
                    batch_advantages = batch_advantages.detach()

                    # Check for NaNs in agent_state
                    if torch.isnan(agent_state).any():
                        self.logger.error(f"NaN detected in agent_state for agent {agent_idx}")
                        continue  # Skip this batch

                    mean, std = agent(agent_state)

                    # Check for NaNs in mean and std
                    if torch.isnan(mean).any() or torch.isnan(std).any():
                        self.logger.error(f"NaN detected in mean or std for agent {agent_idx}")
                        continue  # Skip this batch

                    dist = Normal(mean, std)
                    agent_log_prob = dist.log_prob(agent_action).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1)

                    # PPO loss
                    ratio = (agent_log_prob - agent_log_prob_old).exp()
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                    optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                    optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy.mean().item()

    def train(self, num_episodes, max_steps_per_episode=100):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            step = 0
            total_rewards = [[] for _ in range(self.num_agents)]  # Store rewards per joint
            total_joint_errors = [[] for _ in range(self.num_agents)]  # Store joint errors per joint
            previous_joint_angles = None
            success = 0  # Track success rate
            trajectories = [{'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []} for _ in range(self.num_agents)]

            while not done and step < max_steps_per_episode:
                actions, log_probs = self.get_actions(state)
                next_state, rewards, done, _ = self.env.step(actions)

                # Collect joint angles, positions, and orientations
                current_joint_angles = [agent_state['joint_angle'] for agent_state in state]
                current_position, current_orientation = self.env.get_current_pose()  # Assuming this method exists

                # Convert joint angles to NumPy arrays
                current_joint_angles = np.array(current_joint_angles).flatten()

                # Initialize previous_joint_angles on the first step
                if previous_joint_angles is None:
                    previous_joint_angles = np.array(current_joint_angles)

                # Compute combined reward
                total_reward, position_error, orientation_error, joint_error, success = compute_combined_reward(
                    current_position=current_position,
                    target_position=self.env.target_position,
                    current_orientation=current_orientation,
                    target_orientation=self.env.target_orientation,
                    current_joint_angles=current_joint_angles,
                    target_joint_angles=self.env.target_joint_angles,
                    joint_limits=self.env.joint_limits,
                    previous_joint_angles=previous_joint_angles,
                    iteration=step,
                    max_reward=1,
                    min_reward=-1
                )

                # Log rewards and errors per joint
                for agent_idx in range(self.num_agents):
                    total_rewards[agent_idx].append(rewards[agent_idx])  # Individual rewards per joint
                    total_joint_errors[agent_idx].append(joint_error[agent_idx])  # Joint error per joint

                    # Store experiences in trajectories
                    agent_state = self._process_state(state)[agent_idx]
                    trajectories[agent_idx]['states'].append(agent_state)
                    trajectories[agent_idx]['actions'].append(torch.tensor(actions[agent_idx], dtype=torch.float32).to(self.device))
                    trajectories[agent_idx]['log_probs'].append(torch.tensor(log_probs[agent_idx], dtype=torch.float32).to(self.device))
                    trajectories[agent_idx]['rewards'].append(torch.tensor(total_reward, dtype=torch.float32).to(self.device))
                    trajectories[agent_idx]['dones'].append(done)

                # Update previous joint angles for the next step
                previous_joint_angles = np.copy(current_joint_angles)

                state = next_state
                step += 1

            # Update policy
            actor_loss, critic_loss, entropy = self.update_policy(trajectories)

            # Log episode metrics
            self.training_metrics.log_episode(
                joint_errors=total_joint_errors,
                rewards=total_rewards,
                success=success,
                entropy=entropy,
                actor_loss=actor_loss,
                critic_loss=critic_loss
            )

            # Log joint errors and rewards per joint for this episode
            for agent_idx in range(self.num_agents):
                avg_joint_error = np.mean(total_joint_errors[agent_idx])
                avg_joint_reward = np.mean(total_rewards[agent_idx])
                self.logger.info(f"Episode {episode}, Joint {agent_idx+1} - Avg Error: {avg_joint_error:.4f}, Avg Reward: {avg_joint_reward:.4f}")

            # Log overall episode metrics every 10 episodes
            if episode % 10 == 0:
                avg_reward = sum([sum(agent_rewards) for agent_rewards in total_rewards]) / self.num_agents
                for agent_idx in range(self.num_agents):
                    avg_joint_error = np.mean(total_joint_errors[agent_idx])
                    avg_joint_reward = np.mean(total_rewards[agent_idx])
                    self.logger.info(f"Episode {episode}, Joint {agent_idx+1} - Avg Error: {avg_joint_error:.4f}, Avg Reward: {avg_joint_reward:.4f}")

        # After training, save logs and plot metrics
        self.training_metrics.save_logs("training_logs.json")

        # Calculate metrics and plot them
        metrics = self.training_metrics.calculate_metrics()
        self.training_metrics.plot_metrics(metrics, num_episodes, self.env)

    def test_agent(self, env, num_episodes, max_steps=1000):
        """
        Test the trained agent in the environment.
        """
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            while not done and step_count < max_steps:
                actions, _ = self.get_actions(state)
                next_state, rewards, done, _ = env.step(actions)
                episode_reward += sum(rewards)  # Sum rewards across agents
                state = next_state
                step_count += 1
            print(f"Test Episode {episode+1}, Total Reward: {episode_reward}")
        env.close()
