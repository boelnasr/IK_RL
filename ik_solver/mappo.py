import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Import functional module for ELU and other activations
from torch.distributions import Normal
import numpy as np
import logging
from config import config
from collections import deque
from .reward_function import (compute_jacobian_angular, compute_jacobian_linear, compute_position_error, compute_quaternion_distance, compute_reward, compute_overall_distance,assign_joint_weights,compute_weighted_joint_rewards)
from torch.distributions import Normal
import pybullet as p
from .models import JointActor, CentralizedCritic
from .exploration import ExplorationModule
from .replay_buffer import PrioritizedReplayBuffer
from .curriculum import CurriculumManager

# Initialize device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if CUDA is available and print device name
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Using CPU.")
# Assuming compute_combined_reward and TrainingMetrics are defined elsewhere
from .training_metrics import TrainingMetrics



class MAPPOAgent:

    def __init__(self, env, config):
        logging.info("Starting MAPPOAgent initialization")

        # Environment and device setup
        self.env = env
        self.num_joints = env.num_joints
        self.num_agents = env.num_joints
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters and exploration strategy
        self.hidden_dim = config.get('hidden_dim', 256)  # Set default hidden dimension
        self.epsilon = config.get('initial_epsilon', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.min_epsilon = config.get('min_epsilon', 0.01)

        # Learning rate and clip parameters for agents
        self.global_lr = config.get('lr', 1e-4)  # Set default learning rate
        self.global_clip_param = config.get('clip_param', 0.1)  # Clip parameter for PPO
        self.agent_lrs = [
            config.get(f'lr_joint_{i}', self.global_lr) for i in range(self.num_joints)
        ]
        self.agent_clip_params = [
            config.get(f'clip_joint_{i}', self.global_clip_param) for i in range(self.num_joints)
        ]

        # Learning parameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.95)
        self.clip_param = config.get('clip_param', 0.2)

        # Adaptive learning rate scheduler setup
        self.scheduler_enabled = config.get('use_scheduler', True)
        self.scheduler_patience = config.get('scheduler_patience', 10)
        self.scheduler_decay_factor = config.get('scheduler_decay_factor', 0.5)

        # Training parameters
        self.num_episodes = config.get('num_episodes', 1000)
        self.current_episode = 0
        self.ppo_epochs = config.get('ppo_epochs', 15)
        self.batch_size = config.get('batch_size', 64)

        # Entropy coefficient for exploration
        self.initial_entropy_coef = config.get('initial_entropy_coef', 0.001)
        self.final_entropy_coef = config.get('final_entropy_coef', 0.0001)

        # Observation space dimensions
        try:
            self.obs_dims = [
                sum(np.prod(space.shape) for space in obs_space.spaces.values())
                for obs_space in env.observation_space
            ]
        except AttributeError as e:
            logging.error("Error accessing env.observation_space: Check structure and ensure compatibility.")
            raise e

        # Initialize agents and optimizers
        self.agents = []
        self.optimizers = []
        for agent_idx, obs_dim in enumerate(self.obs_dims):
            actor = JointActor(obs_dim, self.hidden_dim, action_dim=1).to(self.device)
            optimizer = optim.Adam(actor.parameters(), lr=self.agent_lrs[agent_idx], weight_decay=1e-5)
            self.agents.append(actor)
            self.optimizers.append(optimizer)

        # Centralized critic setup
        self.critic = CentralizedCritic(
            state_dim=sum(self.obs_dims),
            hidden_dim=self.hidden_dim,
            num_agents=self.num_agents,
        ).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.global_lr, weight_decay=1e-5)

        if self.scheduler_enabled:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.critic_optimizer,
                mode='min',
                factor=self.scheduler_decay_factor,
                patience=self.scheduler_patience,
            )

        # Process a sample state to determine state_dim
        sample_state = self.env.reset()
        processed_state_list = self._process_state(sample_state)
        global_state = torch.cat(processed_state_list).unsqueeze(0).to(self.device)
        state_dim = global_state.shape[1]

        # Determine action_dim
        action_dim = self.num_agents  # Assuming one action per agent

        # Initialize exploration module for intrinsic rewards
        self.exploration_module = ExplorationModule(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device
        )

        # Optional prioritized experience replay
        self.use_prioritized_replay = config.get('use_prioritized_replay', True)
        if self.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=config.get('replay_buffer_size', 10000)
            )
        # Initialize the AdaptiveRewardScaler
        self.reward_scaler = AdaptiveRewardScaler(window_size=100, initial_scale=1.0)

        # Initialize training metrics
        self.training_metrics = TrainingMetrics(num_joints=self.num_joints)

        # Sliding window for tracking convergence
        self.convergence_window_size = config.get('convergence_window_size', 100)
        self.rewards_window = deque(maxlen=self.convergence_window_size)
        self.success_window = deque(maxlen=self.convergence_window_size)
        self.error_window = deque(maxlen=self.convergence_window_size)

        # Logging setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('training.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Best agent tracking
        self.best_agents_state_dict = [None] * self.num_agents
        self.best_joint_errors = [float('inf')] * self.num_agents

        logging.info("MAPPOAgent initialization complete.")


    def update_policy(self, trajectories):
        # Unpack and preprocess trajectories

        # Ensure trajectories contain the required data for each agent
        assert all(
            'states' in t
            and 'actions' in t
            and 'log_probs' in t
            and 'rewards' in t
            and 'dones' in t
            for t in trajectories
        ), "Each trajectory must contain 'states', 'actions', 'log_probs', 'rewards', and 'dones' keys."

        # Unpack and process trajectories
        states = [torch.stack(t['states']) for t in trajectories]
        actions = [torch.stack(t['actions']) for t in trajectories]
        log_probs_old = [torch.stack(t['log_probs']) for t in trajectories]
        rewards = [torch.stack(t['rewards']) for t in trajectories]
        dones = torch.tensor(trajectories[0]['dones'], dtype=torch.float32).to(self.device)

        # Ensure all tensors are the same length
        min_length = min(s.size(0) for s in states)
        states = [s[:min_length] for s in states]
        actions = [a[:min_length] for a in actions]
        log_probs_old = [lp[:min_length] for lp in log_probs_old]
        rewards = [r[:min_length] for r in rewards]
        dones = dones[:min_length]

        # Concatenate states for critic input
        states_cat = torch.cat(states, dim=1)
        actions_cat = torch.stack(actions, dim=1)
        log_probs_old_cat = torch.stack(log_probs_old, dim=1)

        # Compute values
        with torch.no_grad():
            values = self.critic(states_cat).squeeze().view(min_length, self.num_agents)

        # Stack rewards and compute mean rewards
        rewards_tensor = torch.stack(rewards, dim=1)
        mean_rewards = rewards_tensor.mean(dim=1, keepdim=True).expand(min_length, self.num_agents)

        # Compute advantages and returns
        advantages, returns = self.compute_individual_gae(
            mean_rewards[:min_length], dones, values[:min_length]
        )

        # Transpose advantages and returns for consistent dimensions
        advantages = advantages.transpose(0, 1)
        returns = returns.transpose(0, 1)

        # Clip advantages to reduce outliers
        advantages = torch.clamp(advantages, -10, 10)

        # Normalize advantages per agent
        for agent_idx in range(self.num_agents):
            agent_advantages = advantages[:, agent_idx]
            advantages_mean = agent_advantages.mean()
            advantages_std = agent_advantages.std() + 1e-8
            advantages[:, agent_idx] = (agent_advantages - advantages_mean) / advantages_std

        # Prepare data for training
        dataset = torch.utils.data.TensorDataset(
            states_cat, actions_cat, log_probs_old_cat, advantages, returns
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize a list to store policy losses per agent
        policy_losses_per_agent = [[] for _ in range(self.num_agents)]

        # Calculate current entropy coefficient
        progress = min(self.current_episode / self.num_episodes, 1.0)
        current_entropy_coef = self.initial_entropy_coef * (1 - progress) + self.final_entropy_coef * progress

        # Training loop
        for _ in range(self.ppo_epochs):
            for batch in loader:
                batch_states, batch_actions, batch_log_probs_old, batch_advantages, batch_returns = batch

                # Critic update
                values_pred = self.critic(batch_states).squeeze().view(batch_returns.shape)
                critic_loss = nn.MSELoss()(values_pred, batch_returns)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.critic_optimizer.step()

                # Actor update
                for agent_idx, agent in enumerate(self.agents):
                    optimizer = self.optimizers[agent_idx]
                    start_idx = sum(self.obs_dims[:agent_idx])
                    end_idx = sum(self.obs_dims[:agent_idx + 1])
                    agent_state = batch_states[:, start_idx:end_idx]
                    agent_action = batch_actions[:, agent_idx].unsqueeze(1)
                    agent_log_prob_old = batch_log_probs_old[:, agent_idx]

                    # Reshape batch_advantages to match dimensions
                    agent_advantages = batch_advantages[:, agent_idx].unsqueeze(1)

                    # Forward pass through the agent's policy network
                    mean, std = agent(agent_state)
                    std = torch.clamp(std*self.epsilon, min=1e-4)  # Ensure std is not too small
                    dist = Normal(mean, std)
                    agent_log_prob = dist.log_prob(agent_action).sum(dim=-1, keepdim=True)
                    entropy = dist.entropy().sum(dim=-1, keepdim=True)

                    # Calculate the PPO surrogate losses
                    ratio = torch.exp(agent_log_prob - agent_log_prob_old.unsqueeze(1))
                    surr1 = ratio * agent_advantages
                    surr2 = (
                        torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                        * agent_advantages
                    )
                    actor_loss = (
                        -torch.min(surr1, surr2).mean() - current_entropy_coef * entropy.mean()
                    )

                    optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                    optimizer.step()

                    # Record the policy loss per agent
                    policy_loss = -torch.mean(agent_log_prob)
                    policy_losses_per_agent[agent_idx].append(policy_loss.item())

                    # Detailed logging
                    self.logger.info(
                        f"Agent {agent_idx} - Actor Loss: {actor_loss.item():.6f}, Policy Loss: {policy_loss.item():.6f}"
                    )

        # Compute average policy loss per agent over the epochs
        avg_policy_loss_per_agent = [np.mean(agent_losses) for agent_losses in policy_losses_per_agent]

        # Return the losses for logging and analysis
        return actor_loss.item(), critic_loss.item(), entropy.mean().item(), avg_policy_loss_per_agent

    # Remove one of the duplicate definitions of compute_individual_gae
    def compute_individual_gae(self, rewards, dones, values):
        """
        Computes GAE (Generalized Advantage Estimation) individually for each agent.

        Args:
            rewards (tensor): [batch_size, num_agents] reward tensor for all agents.
            dones (tensor): [batch_size] done flags for each timestep.
            values (tensor): [batch_size, num_agents] value predictions from critic.
        """
        advantages, returns = [], []
        for agent_idx in range(self.num_agents):
            agent_rewards = rewards[:, agent_idx]  # Select rewards per agent
            agent_values = values[:, agent_idx]  # Select values per agent

            gae = 0
            agent_advantages, agent_returns = [], []
            next_value = 0

            # Compute GAE per agent
            for step in reversed(range(len(agent_rewards))):
                mask = 1 - dones[step].item()
                delta = (
                    agent_rewards[step]
                    + self.gamma * next_value * mask
                    - agent_values[step]
                )
                gae = delta + self.gamma * self.tau * mask * gae
                agent_advantages.insert(0, gae)
                agent_returns.insert(0, gae + agent_values[step])

                next_value = agent_values[step]

            advantages.append(torch.tensor(agent_advantages, device=self.device))
            returns.append(torch.tensor(agent_returns, device=self.device))

        return torch.stack(advantages), torch.stack(returns)


    def _compute_gae_for_agent(self, rewards, dones, values):
        """
        Helper function to compute GAE for a single agent.
        
        Args:
            rewards (tensor): Rewards for each timestep for this agent.
            dones (tensor): Done flags for each timestep.
            values (tensor): Value predictions from the critic for this agent.
        
        Returns:
            advantages (tensor): Advantage estimates.
            returns (tensor): Computed returns for the agent.
        """
        gae = 0
        agent_advantages = []
        agent_returns = []
        next_value = 0  # Next value is set to 0 at the end of the trajectory

        # Convert to tensor format
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        values = values.to(self.device)

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.tau * (1 - dones[step]) * gae
            agent_advantages.insert(0, gae)
            agent_returns.insert(0, gae + values[step])

            # Update next_value for next iteration
            next_value = values[step]

        # Stack advantages and returns for the agent
        agent_advantages = torch.stack(agent_advantages)
        agent_returns = torch.stack(agent_returns)
        
        return agent_advantages, agent_returns
    
    def save_best_model(self, agent_idx, actor, episode, avg_error, criterion='mean_error'):
        """
        Save the best model for the specified agent based on the average joint error or other criteria.
        
        Args:
            agent_idx (int): The index of the agent (joint).
            actor (nn.Module): The actor network for the agent.
            episode (int): The episode number during which the best model was found.
            avg_error (float): The average joint error for the agent.
            criterion (str): The criterion for saving ('mean_error' or 'reward').
        """
        model_path = f"best_agent_joint_{agent_idx}_criterion_{criterion}_episode_{episode}.pth"
        torch.save(actor.state_dict(), model_path)
        logging.info(f"Saved new best model for agent {agent_idx} at episode {episode} with Avg {criterion}: {avg_error:.6f}")


    def get_actions(self, state):
        processed_states = self._process_state(state)
        actions = []
        log_probs = []
        for agent_idx, agent in enumerate(self.agents):
            state_tensor = processed_states[agent_idx].unsqueeze(0)
            with torch.no_grad():
                mean, std = agent(state_tensor)
            std = torch.clamp(std * self.epsilon, min=1e-4)  # Scale exploration noise by epsilon
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


    def train(self, num_episodes, max_steps_per_episode=1000):
        logging.info(f"Starting training for {num_episodes} episodes")
        self.training_metrics = TrainingMetrics(num_joints=self.num_joints)

        # Enable anomaly detection for debugging purposes
        torch.autograd.set_detect_anomaly(True)
        self.num_episodes = num_episodes  # Update total number of episodes

        # Initialize CurriculumManager
        curriculum_manager = CurriculumManager(
            initial_difficulty=0.50, max_difficulty=5.0, success_threshold=0.8
        )

        for episode in range(num_episodes):
            self.current_episode = episode  # Update current episode counter
            difficulty = curriculum_manager.get_current_difficulty()
            state = self.env.reset(difficulty=difficulty)
            done = False
            step = 0
            total_rewards = [[] for _ in range(self.num_agents)]
            total_errors = [[] for _ in range(self.num_agents)]
            total_joint_errors = []

            # Tracking variables for episode stats
            begin_distance = None
            prev_best = float('inf')
            cumulative_reward = 0

            # Log initial joint angles
            initial_joint_angles = [p.getJointState(self.env.robot_id, i)[0] for i in self.env.joint_indices]
            logging.info(f"Episode {episode} - Initial joint angles: {initial_joint_angles}")

            # Initialize trajectories for each agent
            trajectories = [{'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []}
                            for _ in range(self.num_agents)]
            logging.info(f"Episode {episode} - Initializing episode variables")

            while not done and step < max_steps_per_episode:
                # Log step-level state if needed for debugging
                if step % 10 == 0:
                    logging.debug(f"Step {step}: Executing action")

                # Get pre-action joint angles
                pre_action_angles = [p.getJointState(self.env.robot_id, i)[0] for i in self.env.joint_indices]

                # Process the current state for agents
                processed_state_list = self._process_state(state)  # List of tensors per agent

                # Process the global state for the exploration module
                global_state = torch.cat(processed_state_list).unsqueeze(0).to(self.device)  # Shape: (1, state_dim)

                actions, log_probs = self.get_actions(state)
                next_state, rewards, done, info = self.env.step(actions)

                # Scale rewards using the AdaptiveRewardScaler
                scaled_rewards = [self.reward_scaler.update_scale(reward) * reward for reward in rewards]

                # Process the next state
                processed_next_state_list = self._process_state(next_state)
                global_next_state = torch.cat(processed_next_state_list).unsqueeze(0).to(self.device)  # Shape: (1, state_dim)

                # Actions tensor
                actions_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1, action_dim)

                # Get post-action joint angles
                post_action_angles = [p.getJointState(self.env.robot_id, i)[0] for i in self.env.joint_indices]

                # Calculate distance to target and update begin_distance
                current_position, current_orientation = self.get_end_effector_pose()
                target_position, target_orientation = self.get_target_pose()
                distance = compute_overall_distance(current_position, target_position,
                                                    current_orientation, target_orientation)
                if begin_distance is None:
                    begin_distance = distance

                # Compute joint errors and assign weights using Jacobians
                jacobian_linear = compute_jacobian_linear(self.env.robot_id, self.env.joint_indices, post_action_angles)
                jacobian_angular = compute_jacobian_angular(self.env.robot_id, self.env.joint_indices, post_action_angles)
                linear_weights, angular_weights = assign_joint_weights(jacobian_linear, jacobian_angular)

                # Calculate joint errors
                joint_errors = []
                for i in range(self.num_agents):
                    # Assuming you have target joint angles for each joint
                    target_joint_angle = initial_joint_angles[i]  # Or use the desired target joint angles
                    error = abs(post_action_angles[i] - target_joint_angle)
                    joint_errors.append(max(error, 1e-6))  # Ensure non-zero error
                total_joint_errors.append(joint_errors)
                current_success_threshold = self.env.success_threshold  # Use the environment's success threshold
                # Calculate rewards and update cumu lative reward
                overall_reward, prev_best, success = compute_reward(
                    distance=distance,
                    begin_distance=begin_distance,
                    prev_best=prev_best,
                    current_orientation=current_orientation,
                    target_orientation=target_orientation,
                    joint_errors=joint_errors,
                    linear_weights=linear_weights,
                    angular_weights=angular_weights,
                    success_threshold=current_success_threshold
                )
                #Calculate reward using the modified compute_reward function

            #     overall_reward, prev_best, success = compute_reward(
            #     distance=distance,
            #     begin_distance=begin_distance,
            #     prev_best=prev_best,
            #     joint_errors=joint_errors,
            #     linear_weights=linear_weights,
            #     angular_weights=angular_weights,
            #     success_threshold=current_success_threshold

            # )


                # Calculate intrinsic reward using exploration module
                intrinsic_reward = self.exploration_module.get_combined_intrinsic_reward(
                    state=global_state,
                    action=actions_tensor,
                    next_state=global_next_state
                )

                # Add intrinsic reward to the overall reward
                overall_reward += intrinsic_reward.item()
                cumulative_reward = 0.95 * cumulative_reward + overall_reward

                # Log rewards and errors for each agent
                for agent_idx in range(self.num_agents):
# Safely calculate the joint success rate
                    if len(total_rewards[agent_idx]) > 0:
                        joint_success_rate = sum([1 for s in total_rewards[agent_idx] if s > self.env.success_threshold]) / len(total_rewards[agent_idx])
                    else:
                        joint_success_rate = 0  # Or handle this case differently if needed
                    # Assign the joint reward to each agent
                    joint_reward = compute_weighted_joint_rewards(
                        [joint_errors[agent_idx]],
                        [linear_weights[agent_idx]],
                        [angular_weights[agent_idx]],
                        overall_reward
                    )[0]

                    total_rewards[agent_idx].append(joint_reward + intrinsic_reward.item())
                    total_errors[agent_idx].append(joint_errors[agent_idx])

                    # Record experience in trajectories
                    agent_state = processed_state_list[agent_idx]  # Use processed state for each agent
                    trajectories[agent_idx]['states'].append(agent_state)
                    trajectories[agent_idx]['actions'].append(torch.tensor(actions[agent_idx],
                                                                        dtype=torch.float32).to(self.device))
                    trajectories[agent_idx]['log_probs'].append(torch.tensor(log_probs[agent_idx],
                                                                            dtype=torch.float32).to(self.device))
                    trajectories[agent_idx]['rewards'].append(torch.tensor(joint_reward + intrinsic_reward.item(),
                                                                        dtype=torch.float32).to(self.device))
                    trajectories[agent_idx]['dones'].append(done)

                state = next_state
                step += 1

            # Update policy and log results
            actor_loss, critic_loss, entropy, policy_loss_per_agent = self.update_policy(trajectories)

            # Log episode statistics
            for agent_idx in range(self.num_agents):
                mean_joint_error = np.mean(total_errors[agent_idx])
                max_joint_error = np.max(total_errors[agent_idx])
                min_joint_error = np.min(total_errors[agent_idx])
                mean_reward = np.mean(total_rewards[agent_idx])

                logging.info(f"Episode {episode}, Joint {agent_idx} - Mean Error: {mean_joint_error:.6f}, "
                            f"Max Error: {max_joint_error:.6f}, Min Error: {min_joint_error:.6f}, "
                            f"Mean Reward: {mean_reward:.6f}")

                # Save best model if the joint error improves
                if mean_joint_error < self.best_joint_errors[agent_idx]:
                    self.best_joint_errors[agent_idx] = mean_joint_error
                    self.best_agents_state_dict[agent_idx] = self.agents[agent_idx].state_dict()
                    torch.save(self.best_agents_state_dict[agent_idx], f"best_agent_joint_{agent_idx}.pth")
                    logging.info(f"New best model saved for joint {agent_idx} with mean error {mean_joint_error:.6f}")

            # Check and update curriculum
            curriculum_manager.log_success(success)
            curriculum_manager.update_difficulty()

            # Log overall episode statistics
            success_status = info.get('success_per_agent', [False] * self.num_agents)
            episode_joint_errors = np.array(total_joint_errors)
            episode_rewards = [np.sum(agent_rewards) for agent_rewards in total_rewards]

                    # Save metrics and generate plots after training is complete
            self.training_metrics.log_episode(
            joint_errors=total_joint_errors,
            rewards=episode_rewards,
            success=success_status,
            entropy=entropy,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            policy_loss=policy_loss_per_agent,
            env=self.env
        )

        # Save final metrics and generate plots
        self.training_metrics.save_logs()
        metrics = self.training_metrics.calculate_metrics(env=self.env)
        self.training_metrics.plot_metrics(metrics, self.env, show_plots=False)
        self.training_metrics.save_final_report(metrics)

        return metrics


    def _process_state(self, state):
        processed_states = []
        for joint_state in state:
            joint_angle = joint_state['joint_angle'].flatten()
            position_error = joint_state['position_error'].flatten()
            orientation_error = joint_state['orientation_error'].flatten()

            # Normalize features as needed
            joint_angle_normalized = joint_angle / np.pi
            position_error_normalized = position_error / 1.0
            orientation_error_normalized = orientation_error / np.pi

            obs = np.concatenate(
                [
                    joint_angle_normalized,
                    position_error_normalized,
                    orientation_error_normalized,
                ]
            )

            processed_states.append(torch.tensor(obs, dtype=torch.float32).to(self.device))
        return processed_states  # Returns a list of tensors, one per agent

    def _process_global_state(self, state):
        """
        Processes the entire state into a single tensor for the exploration module.
        """
        processed_states = []
        for joint_state in state:
            joint_angle = joint_state['joint_angle'].flatten()
            position_error = joint_state['position_error'].flatten()
            orientation_error = joint_state['orientation_error'].flatten()

            # Normalize joint angles to [-1, 1]
            joint_angle_normalized = joint_angle / np.pi

            # Normalize position errors
            position_error_normalized = position_error / 1.0  # Adjust max value as needed

            # Normalize orientation errors to [-1, 1]
            orientation_error_normalized = orientation_error / np.pi

            obs = np.concatenate(
                [
                    joint_angle_normalized,
                    position_error_normalized,
                    orientation_error_normalized,
                ]
            )

            processed_states.append(obs)
        # Concatenate all agents' observations into a single tensor
        global_state = np.concatenate(processed_states)
        return torch.tensor(global_state, dtype=torch.float32).to(self.device)


    # Helper functions to retrieve the end-effector and target pose
    def get_end_effector_pose(self):
        """
        Retrieve the current position and orientation of the end-effector.
        """
        end_effector_state = p.getLinkState(self.env.robot_id, self.env.joint_indices[-1])
        current_position = np.array(end_effector_state[4])
        current_orientation = np.array(end_effector_state[5])
        return current_position, current_orientation

    def get_target_pose(self):
        """
        Retrieve the target position and orientation (assumes target is stored as attributes).
        """
        return self.env.target_position, self.env.target_orientation

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

class AdaptiveRewardScaler:
    def __init__(self, window_size=100, initial_scale=1.0, adjustment_rate=0.05):
        self.window_size = window_size
        self.reward_history = deque(maxlen=window_size)
        self.scale = initial_scale
        self.min_scale = 0.1
        self.max_scale = 10.0
        self.adjustment_rate = adjustment_rate

    def update_scale(self, reward):
        """Update scaling factor based on reward history."""
        self.reward_history.append(reward)
        if len(self.reward_history) >= self.window_size:
            reward_mean = np.mean(self.reward_history)
            if abs(reward_mean) > 1.0:
                self.scale *= (1 - self.adjustment_rate)
            elif abs(reward_mean) < 0.1:
                self.scale *= (1 + self.adjustment_rate)
            self.scale = np.clip(self.scale, self.min_scale, self.max_scale)
        return self.scale

