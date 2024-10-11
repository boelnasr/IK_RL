import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import logging
import json
import matplotlib.pyplot as plt
from collections import deque
from config import config

# Initialize device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if CUDA is available and print device name
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Using CPU.")
# Assuming compute_combined_reward and TrainingMetrics are defined elsewhere
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
        logging.info("Starting MAPPOAgent initialization")

        self.env = env
        self.num_joints = env.num_joints  # Get the number of joints from the environment
        self.num_agents = env.num_joints  # Assuming one agent per joint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = config.get('hidden_dim', 128)
        
        # Global default learning rate and clip parameter
        self.global_lr = config.get('lr', 1e-4)
        self.global_clip_param = config.get('clip_param', 0.1)

        # Joint-specific learning rates and clip parameters (fallback to global values if not defined)
        self.agent_lrs = [
            config.get(f'lr_joint_{i}', self.global_lr) for i in range(self.num_joints)
        ]
        self.agent_clip_params = [
            config.get(f'clip_joint_{i}', self.global_clip_param) for i in range(self.num_joints)
        ]

        self.agents = []
        self.optimizers = []

        # Initialize gamma and tau
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.95)
        self.clip_param = config.get('clip_param', 0.1)  # Global fallback clip parameter

        # Initialize observation dimensions based on the environment's observation space
        self.obs_dims = []
        for obs_space in env.observation_space:
            obs_dim = sum(np.prod(space.shape) for space in obs_space.spaces.values())
            self.obs_dims.append(obs_dim)

        # Initialize agents (actors) for each joint with separate learning rates and clip parameters
        for agent_idx, obs_dim in enumerate(self.obs_dims):
            actor = JointActor(obs_dim, self.hidden_dim, action_dim=1).to(self.device)
            # Assign the learning rate specific to this agent
            optimizer = optim.Adam(actor.parameters(), lr=self.agent_lrs[agent_idx])
            self.agents.append(actor)
            self.optimizers.append(optimizer)

        # Initialize the centralized critic
        self.critic = CentralizedCritic(sum(self.obs_dims), self.hidden_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.global_lr)

        # Other training parameters
        self.ppo_epochs = config.get('ppo_epochs', 15)
        self.batch_size = config.get('batch_size', 64)
        self.training_metrics = TrainingMetrics()

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('training.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Initialize best agent tracking variables
        self.best_agents_state_dict = [None] * self.num_agents
        self.best_joint_errors = [float('inf')] * self.num_agents

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

        cumulative_policy_loss = 0.0

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

                    mean, std = agent(agent_state)
                    dist = Normal(mean, std)
                    agent_log_prob = dist.log_prob(agent_action).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1)

                    # Use the joint-specific clip parameter
                    self.clip_param = config.get('clip_param', 0.1)  # Global clip parameter for fallback
                    
                    # PPO loss with joint-specific clip parameter
                    ratio = (agent_log_prob - agent_log_prob_old).exp()
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                    optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                    optimizer.step()

                    # Calculate policy loss as the negative of the log probability
                    policy_loss = -torch.mean(agent_log_prob)
                    cumulative_policy_loss += policy_loss.item()

        # Calculate average policy loss
        avg_policy_loss = cumulative_policy_loss / (self.ppo_epochs * len(loader))

        return actor_loss.item(), critic_loss.item(), entropy.mean().item(), avg_policy_loss

    def save_best_model(self, agent_idx, actor, episode, avg_error):
        """
        Save the best model for the specified agent based on the average joint error.

        Args:
            agent_idx (int): The index of the agent (joint).
            actor (nn.Module): The actor network for the agent.
            episode (int): The episode number during which the best model was found.
            avg_error (float): The average joint error for the agent.
        """
        model_path = f"best_agent_joint_{agent_idx}_episode_{episode}.pth"
        torch.save(actor.state_dict(), model_path)
        logging.info(f"Saved new best model for agent {agent_idx} at episode {episode} with Avg Error: {avg_error:.6f}")

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
        next_value = 0  # Initialize next_value to 0

        # Convert inputs to tensors if they're not already
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).requires_grad_(True)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).requires_grad_(True)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device).requires_grad_(True)

        # Check if values is a scalar tensor or a single value and convert accordingly
        if isinstance(values, (int, float)) or (isinstance(values, torch.Tensor) and values.dim() == 0):
            # Use the scalar value to create a tensor with the same shape as `rewards`
            values = torch.full((len(rewards),), values.item() if isinstance(values, torch.Tensor) else values, dtype=torch.float32, device=self.device).requires_grad_(True)
        else:
            values = torch.as_tensor(values, dtype=torch.float32, device=self.device).requires_grad_(True)

        for step in reversed(range(len(rewards))):
            current_value = values[step].item()
            
            mask = 1.0 - dones[step].item()
            delta = rewards[step].item() + self.gamma * next_value * mask - current_value
            gae = delta + self.gamma * self.tau * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + current_value)
            
            # Update next_value for the next iteration
            next_value = current_value

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
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

        # Initialize cumulative policy loss for logging
        cumulative_policy_loss = 0.0

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

                    mean, std = agent(agent_state)
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

                    # Calculate policy loss as the negative of the log probability
                    policy_loss = -torch.mean(agent_log_prob)
                    cumulative_policy_loss += policy_loss.item()

        # Calculate average policy loss
        avg_policy_loss = cumulative_policy_loss / (self.ppo_epochs * len(loader))

        return actor_loss.item(), critic_loss.item(), entropy.mean().item(), avg_policy_loss

    def train(self, num_episodes, max_steps_per_episode=100):
        logging.info(f"Starting training for {num_episodes} episodes")

        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            step = 0
            total_rewards = [[] for _ in range(self.num_agents)]  # Store rewards per joint
            total_errors = [[] for _ in range(self.num_agents)]   # Store errors per joint
            trajectories = [{'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []} for _ in range(self.num_agents)]
            logging.info(f"Episode {episode} - Initializing episode variables")

            while not done and step < max_steps_per_episode:
                actions, log_probs = self.get_actions(state)
                next_state, rewards, done, info = self.env.step(actions)

                # Retrieve joint errors and success status from the environment
                joint_errors = self.env.joint_errors
                success_per_agent = info['success_per_agent']

                # For each agent, store experiences
                for agent_idx in range(self.num_agents):
                    total_rewards[agent_idx].append(rewards[agent_idx])
                    total_errors[agent_idx].append(joint_errors[agent_idx])

                    # Store experiences in trajectories
                    agent_state = self._process_state(state)[agent_idx]
                    trajectories[agent_idx]['states'].append(agent_state)
                    trajectories[agent_idx]['actions'].append(torch.tensor(actions[agent_idx], dtype=torch.float32).to(self.device))
                    trajectories[agent_idx]['log_probs'].append(torch.tensor(log_probs[agent_idx], dtype=torch.float32).to(self.device))
                    trajectories[agent_idx]['rewards'].append(torch.tensor(rewards[agent_idx], dtype=torch.float32).to(self.device))
                    trajectories[agent_idx]['dones'].append(done)

                state = next_state
                step += 1

            # Update policy after accumulating trajectories
            actor_loss, critic_loss, entropy, policy_loss = self.update_policy(trajectories)

            # Evaluate the joint errors for this episode and save the best agent for each joint
            for agent_idx in range(self.num_agents):
                mean_joint_error = np.mean(total_errors[agent_idx])  # Calculate the mean joint error for the agent
                logging.info(f"Episode {episode}, Joint {agent_idx} - Mean Joint Error: {mean_joint_error:.6f}")

                # If the joint error is better (lower), save the agent's state dict
                if mean_joint_error < self.best_joint_errors[agent_idx]:
                    self.best_joint_errors[agent_idx] = mean_joint_error
                    self.best_agents_state_dict[agent_idx] = self.agents[agent_idx].state_dict()
                    torch.save(self.best_agents_state_dict[agent_idx], f"best_agent_joint_{agent_idx}.pth")
                    logging.info(f"New best agent for joint {agent_idx} saved with Mean Joint Error: {mean_joint_error:.6f}")

            # Log metrics for this episode
            self.training_metrics.log_episode(
                joint_errors=joint_errors,
                rewards=rewards,
                success=success_per_agent,
                entropy=entropy,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                policy_loss=policy_loss,  # Pass the policy loss value here
                env=self.env
            )

            # Log average metrics for each joint
            for agent_idx in range(self.num_agents):
                avg_error = np.mean(total_errors[agent_idx])
                avg_reward = np.mean(total_rewards[agent_idx])
                logging.info(f"Episode {episode}, Joint {agent_idx} - Avg Error: {avg_error:.6f}, Avg Reward: {avg_reward:.6f}")

            # Optional: Log overall episode metrics
            avg_episode_reward = sum([sum(agent_rewards) for agent_rewards in total_rewards]) / self.num_agents
            overall_success = all(success_per_agent)
            logging.info(f"Episode {episode} - Avg Episode Reward: {avg_episode_reward:.6f}, Overall Success: {overall_success}")

        # After training, save logs and plot metrics
        self.training_metrics.save_logs("training_logs.json")
        metrics = self.training_metrics.calculate_metrics(env=self.env)
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