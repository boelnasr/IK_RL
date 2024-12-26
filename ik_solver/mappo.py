

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Import functional module for ELU and other activations
import numpy as np
import logging
from config import config
from collections import deque
from .reward_function import (compute_jacobian_angular, compute_jacobian_linear, compute_position_error, compute_quaternion_distance, compute_reward, compute_overall_distance,assign_joint_weights,compute_weighted_joint_rewards)
from torch.distributions import Normal
import pybullet as p
import os
import traceback
from .models import JointActor, CentralizedCritic
from .exploration import ExplorationModule
from .replay_buffer import PrioritizedReplayBuffer
from .curriculum import CurriculumManager
from .PD_controller import PDController
from .lr_scheduler import LRSchedulerManager
from .HindsightReplayBuffer import HindsightReplayBuffer, ValidationManager
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

    def __init__(self, env, config, training=True):
        logging.info("Starting MAPPOAgent initialization")

        # Environment and device setup
        self.env = env
        self.num_joints = env.num_joints
        self.num_agents = env.num_joints
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters and exploration strategy
        self.hidden_dim = config.get('hidden_dim', 256)  # Set default hidden dimension
        self.epsilon = config.get('initial_epsilon', 0.20)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.min_epsilon = config.get('min_epsilon', 0.01)

        # Learning rate and clip parameters for agents
        self.global_lr = config.get('lr', 1e-4)  # Set default learning rate
        self.global_clip_param = config.get('clip_param', 0.2)  # Clip parameter for PPO
        self.agent_lrs = [
            config.get(f'lr_joint_{i}', self.global_lr) for i in range(self.num_joints)
        ]
        self.agent_clip_params = [
            config.get(f'clip_joint_{i}', self.global_clip_param) for i in range(self.num_joints)
        ]
        curriculum_manager = CurriculumManager(
            initial_difficulty=0.0, 
            max_difficulty=3.0, 
            success_threshold=0.8
        )
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
        self.max_steps_per_episode = config.get('max_steps_per_episode', 5000)
        self.current_episode = 0
        self.ppo_epochs = config.get('ppo_epochs', 15)
        self.batch_size = config.get('batch_size', 64)

        # Entropy coefficient for exploration
        self.initial_entropy_coef = config.get('initial_entropy_coef', 0.01)
        self.final_entropy_coef = config.get('final_entropy_coef', 0.001)
        self.std_min = 1e-4
        self.std_max = 1.0
        self.entropy_min = -200.0
        self.entropy_max = 200.0
        self.initial_entropy_coef = 0.1
        self.final_entropy_coef = 0.01
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
            actor = JointActor(obs_dim, self.hidden_dim,use_attention=True, action_dim=1).to(self.device)
            optimizer = optim.Adam(actor.parameters(), lr=self.agent_lrs[agent_idx], weight_decay=1e-5)
            self.agents.append(actor)
            self.optimizers.append(optimizer)

        # Centralized critic setup
        self.critic = CentralizedCritic(
            state_dim=sum(self.obs_dims),
            hidden_dim=self.hidden_dim,
            num_agents=self.num_agents,
            use_attention=True,
        ).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.global_lr, weight_decay=1e-5)

        # Initialize learning rate scheduler manager
        all_optimizers = self.optimizers + [self.critic_optimizer]  # Combine actor and critic optimizers
        self.lr_manager = LRSchedulerManager(
            optimizers=all_optimizers,
            initial_lr=self.global_lr,
            warmup_steps=10,  # Set warmup steps as needed
            min_lr=5e-5
        )
        # Initialize early stopping
        self.init_early_stopping(
            patience=config.get("early_stop_patience", 50),
            min_delta=config.get("early_stop_min_delta", 1e-4),
            min_episodes=config.get("early_stop_min_episodes", 100),
        )


        # Process a sample state to determine state_dim
        sample_state = self.env.reset()
        processed_state_list = self._process_state(sample_state)
        global_state = torch.cat(processed_state_list).unsqueeze(0).to(self.device)
        state_dim = global_state.shape[1]
        # Add PD controller initialization
        self.pd_controllers = []
        self.pd_weight = config.get('pd_weight', 0.3)
    
        # Initialize PD controllers for each joint
        for _ in range(self.num_agents):
            self.pd_controllers.append(PDController(
                kp=config.get('pd_kp', 1.0),
                kd=config.get('pd_kd', 0.2),
                dt=config.get('pd_dt', 0.01)
            ))
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
                capacity=10000,
                alpha=0.6,
                beta_start=0.4,
                beta_frames=100000
            )
                # Add these new parameters
        self.value_loss_scale = config.get('value_loss_scale', 0.5)  # Critic loss scaling
        self.entropy_scale = config.get('entropy_scale', 0.001)    # Entropy scaling
        self.max_grad_norm = config.get('max_grad_norm', 0.5)    # Gradient clipping norm
        self.ratio_clip = config.get('ratio_clip', 0.20)         # Policy ratio clipping
        self.advantage_clip = config.get('advantage_clip', 2.0)   # Advantage clipping
                
        # Initialize the AdaptiveRewardScaler
        self.reward_scaler = AdaptiveRewardScaler(window_size=100, initial_scale=1.0)

        # Initialize training metrics
        self.training_metrics = TrainingMetrics(num_joints=self.num_joints)
        self.base_path=self.training_metrics.base_path
        # Sliding window for tracking convergence
        self.convergence_window_size = config.get('convergence_window_size', 100)
        self.rewards_window = deque(maxlen=self.convergence_window_size)
        self.success_window = deque(maxlen=self.convergence_window_size)
        self.error_window = deque(maxlen=self.convergence_window_size)

        # Logging setup
        self.logger = logging.getLogger(__name__)
        if training:
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler('training.log')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        else:
            self.logger.setLevel(logging.WARNING)

        # Best agent tracking
        self.best_agents_state_dict = [None] * self.num_agents
        self.best_joint_errors = [float('inf')] * self.num_agents
        # Initialize HER buffer with hindsight replay
        #Initialize the HindsightReplayBuffer with capacity, alpha, beta, and k_future
        self.her_buffer = HindsightReplayBuffer(
            capacity=config.get('buffer_size', 100000),
            alpha=config.get('alpha', 0.6),
            beta_start=config.get('beta_start', 0.4),
            k_future=config.get('k_future', 4)
        )
        self.reward_stabilizer = RewardStabilizer(
            window_size=100,
            scale_min=0.1,
            scale_max=2.0,
            ewma_alpha=0.95,
            clip_range=(-10.0, 10.0),
            adaptive_clip_percentile=95
        )

        # Initialize the ValidationManager
        self.validation_manager = ValidationManager(
            validation_frequency=config.get('validation_frequency', 10),
            validation_episodes=config.get('validation_episodes', 10)
        )
        logging.info("MAPPOAgent initialization complete.")


    def update_policy(self, trajectories):
        """
        Update policy with improved entropy handling and stability measures.
        """
        try:
            def validate_tensor(t):
                if isinstance(t, list):
                    t = torch.stack(t)
                t = t.float()
                t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
                return t.to(self.device)

            # Initialize entropy stats if needed
            if not hasattr(self, 'entropy_stats'):
                self.entropy_stats = {
                    'running_mean': 1.0,  # Target positive entropy
                    'target_entropy': 0.001,
                    'history': deque(maxlen=100)
                }
                self.entropy_scale = 0.05  # Initial entropy scale

            # Process trajectory data
            states = [validate_tensor(t['states']) for t in trajectories]
            actions = [validate_tensor(t['actions']) for t in trajectories]
            log_probs_old = [validate_tensor(t['log_probs']) for t in trajectories]
            rewards = [validate_tensor(t['rewards']) for t in trajectories]
            dones = validate_tensor(torch.tensor(trajectories[0]['dones']))

            # Ensure consistent sequence length
            min_length = min(s.size(0) for s in states)
            states = [s[:min_length] for s in states]
            actions = [a[:min_length] for a in actions]
            log_probs_old = [lp[:min_length] for lp in log_probs_old]
            rewards = [r[:min_length] for r in rewards]
            dones = dones[:min_length]

            # Concatenate tensors
            states_cat = torch.cat(states, dim=1).float()
            actions_cat = torch.stack(actions, dim=1).float()
            log_probs_old_cat = torch.stack(log_probs_old, dim=1).float()

            # Get value predictions
            with torch.no_grad():
                values = self.critic(states_cat).float()
                values = torch.clamp(values, -50.0, 50.0)

            # Process rewards
            rewards_tensor = torch.stack(rewards, dim=1).float()
            rewards_clamped = torch.clamp(rewards_tensor, min=-10.0, max=10.0)
            rewards_mean = rewards_clamped.mean()
            rewards_std = max(rewards_clamped.std(), 1e-6)
            rewards_normalized = (rewards_clamped - rewards_mean) / rewards_std

            # Compute advantages and returns
            advantages, returns = self.compute_individual_gae(
                rewards_normalized.detach(),
                dones.detach(),
                values.detach()
            )
            advantages = advantages.transpose(0, 1).float()
            returns = returns.transpose(0, 1).float()

            # Normalize advantages
            advantages = torch.clamp(advantages, -self.advantage_clip, self.advantage_clip)
            for agent_idx in range(self.num_agents):
                agent_mean = advantages[:, agent_idx].mean()
                agent_std = max(advantages[:, agent_idx].std(), 1e-6)
                advantages[:, agent_idx] = (advantages[:, agent_idx] - agent_mean) / agent_std

            # Create dataset and loader
            dataset = torch.utils.data.TensorDataset(
                states_cat, actions_cat, log_probs_old_cat,
                advantages.detach(), returns.detach()
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )

            # Initialize tracking variables
            total_critic_loss = 0.0
            total_actor_loss = 0.0
            total_entropy = 0.0
            total_entropy_loss = 0.0
            policy_losses = [[] for _ in range(self.num_agents)]
            actor_loss_list = [[] for _ in range(self.num_agents)]

            for _ in range(self.ppo_epochs):
                for batch in loader:
                    batch_states, batch_actions, batch_log_probs_old, batch_advantages, batch_returns = [
                        b.float().to(self.device) for b in batch
                    ]

                    # Update critic
                    values_pred = self.critic(batch_states).float()
                    values_pred = torch.clamp(values_pred, -50.0, 50.0)
                    critic_loss = F.mse_loss(values_pred, batch_returns)
                    critic_loss = torch.clamp(critic_loss, max=10.0)
                    critic_loss = self.value_loss_scale * critic_loss
                    critic_loss += 0.001 * sum(p.pow(2).sum() for p in self.critic.parameters())

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                    total_critic_loss += critic_loss.item()

                    # Update actors
                    for agent_idx, agent in enumerate(self.agents):
                        start_idx = sum(self.obs_dims[:agent_idx])
                        end_idx = start_idx + self.obs_dims[agent_idx]
                        agent_state = batch_states[:, start_idx:end_idx]

                        # Get policy distribution
                        mean, std = agent(agent_state)
                        std = torch.clamp(std * self.epsilon, min=self.std_min, max=self.std_max)
                        dist = Normal(mean, std)

                        # Compute log probabilities
                        agent_log_prob = dist.log_prob(batch_actions[:, agent_idx])
                        agent_log_prob = torch.clamp(agent_log_prob, -20.0, 2.0)
                        agent_log_prob = agent_log_prob.sum(dim=-1, keepdim=True)

                        # Compute probability ratio
                        old_log_prob = batch_log_probs_old[:, agent_idx].unsqueeze(1)
                        log_ratio = agent_log_prob - old_log_prob
                        log_ratio = torch.clamp(log_ratio, -2.0, 2.0)
                        ratio = torch.exp(log_ratio)

                        # Compute surrogate objectives
                        agent_advantages = batch_advantages[:, agent_idx].unsqueeze(1)
                        surr1 = ratio * agent_advantages
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * agent_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Compute positive entropy and handle entropy loss
                        entropy = -dist.entropy().mean()
                        entropy = torch.clamp(entropy, min=0.001, max=2.0)

                        # Update entropy stats and scale
                        self.entropy_stats['history'].append(entropy.item())
                        if len(self.entropy_stats['history']) > 10:
                            current_mean = np.mean(self.entropy_stats['history'])
                            entropy_diff = current_mean - self.entropy_stats['target_entropy']
                            
                            if entropy_diff > 0:  # Entropy too high
                                self.entropy_scale *= 1.005  # Increase penalty
                            else:  # Entropy too low
                                self.entropy_scale *= 0.995  # Decrease penalty
                            
                            self.entropy_scale = np.clip(self.entropy_scale, 0.001, 0.1)

                        entropy_loss = -self.entropy_scale * entropy  # Negative for maximization
                        entropy_loss = torch.clamp(entropy_loss, -1.0, 1.0)

                        # Compute final actor loss
                        actor_loss = policy_loss + entropy_loss
                        actor_loss += 0.001 * sum(p.pow(2).sum() for p in agent.parameters())

                        self.optimizers[agent_idx].zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                        self.optimizers[agent_idx].step()

                        policy_losses[agent_idx].append(policy_loss.item())
                        actor_loss_list[agent_idx].append(actor_loss.item())
                        total_actor_loss += actor_loss.item()
                        total_entropy += entropy.item()
                        total_entropy_loss += entropy_loss.item()

                        if agent_idx == 0:  # Debug info for first agent
                            print(f"Entropy: {entropy.item():.4f}, Scale: {self.entropy_scale:.4f}")

            # Compute averages
            n_updates = max(self.ppo_epochs * len(loader) * self.num_agents, 1)
            avg_actor_loss = total_actor_loss / n_updates
            avg_critic_loss = total_critic_loss / max(self.ppo_epochs * len(loader), 1)
            avg_entropy = total_entropy / n_updates
            avg_policy_losses = [np.mean(losses) if losses else 0.0 for losses in policy_losses]
            avg_actor_loss_list = [np.mean(losses) if losses else 0.0 for losses in actor_loss_list]
            avg_entropy_loss = total_entropy_loss / n_updates
            print(f"avg_entropy_loss:{avg_entropy_loss}\n")
            return (
                float(avg_actor_loss),
                float(avg_critic_loss),
                float(avg_entropy),
                avg_policy_losses,
                avg_actor_loss_list,
                values_pred.detach(),
                returns.detach(),
                advantages.detach()
            )

        except Exception as e:
            self.logger.error(f"Error in policy update: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def compute_individual_gae(self, rewards, dones, values):
        """
        Computes GAE (Generalized Advantage Estimation) individually for each agent.

        Args:
            rewards (tensor): [batch_size, num_agents] reward tensor for all agents.
            dones (tensor): [batch_size] done flags for each timestep.
            values (tensor): [batch_size, num_agents] value predictions from critic.
        """
        advantages, returns = [], []
        rewards = (rewards - rewards.mean(dim=0, keepdim=True)) / (rewards.std(dim=0, keepdim=True) + 1e-8)  # Normalize rewards
        
        for agent_idx in range(self.num_agents):
            agent_rewards = rewards[:, agent_idx]
            agent_values = values[:, agent_idx]
            agent_values = torch.tensor(
                np.convolve(agent_values.cpu().numpy(), np.ones(3)/3, mode='same'), device=self.device
            )  # Smooth critic values
            
            gae = 0
            agent_advantages, agent_returns = [], []
            next_value = 0

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

            agent_advantages = torch.tensor(agent_advantages, device=self.device)
            advantages.append(agent_advantages)
            returns.append(torch.tensor(agent_returns, device=self.device))

        # Stack advantages and normalize
        advantages = torch.stack(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages

        return advantages, torch.stack(returns)


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
        model_path =os.path.join(self.base_path,f"best_agent_joint_{agent_idx}")
        torch.save(actor.state_dict(), model_path)
        logging.info(f"Saved new best model for agent {agent_idx} at episode {episode} with Avg {criterion}: {avg_error:.6f}")


    def get_actions(self, state, eval_mode=False):
        processed_states = self._process_state(state)
        actions = []
        log_probs = []
        
        for agent_idx, agent in enumerate(self.agents):
            state_tensor = processed_states[agent_idx].unsqueeze(0)
            
            # Get RL policy action
            with torch.no_grad():
                mean, std = agent(state_tensor)
            std = torch.clamp(std * self.epsilon, min=1e-4)
            dist = Normal(mean, std)
            rl_action = dist.sample()
            rl_action = rl_action.clamp(-1, 1)
            log_prob = dist.log_prob(rl_action).sum(dim=-1)
            
            # Extract scalar action value
            rl_action_value = float(rl_action.squeeze().cpu().item())
            
            if eval_mode:
                # Calculate PD correction
                current_angle = state[agent_idx]['joint_angle'].flatten()[0]
                target_angle = self.env.target_joint_angles[agent_idx] if hasattr(self.env, 'target_joint_angles') else 0.0
                error = target_angle - current_angle
                
                pd_correction = self.pd_controllers[agent_idx].compute(error)
                pd_correction = np.clip(pd_correction, -1, 1)
                
                # Ensure PD correction is in the same range as RL action
                pd_correction = np.clip(pd_correction, -1, 1)

                # Combine RL and PD actions
                combined_action = (1 - self.pd_weight) * rl_action_value + self.pd_weight * pd_correction
                combined_action = np.clip(combined_action, -1, 1)
                actions.append(combined_action)

            else:
                # During training, use RL action directly
                action = rl_action_value
                actions.append(action)
            
            # Ensure log_prob is a single scalar
            log_probs.append(log_prob.item())

        return actions, log_probs



    def track_pd_metrics(self, pd_corrections, joint_errors):
        """Track PD controller performance metrics."""
        pd_metrics = {
            'mean_correction': np.nanmean(np.abs(pd_corrections)),
            'max_correction': np.max(np.abs(pd_corrections)),
            'correction_std': np.std(pd_corrections),
            'error_correlation': np.corrcoef(pd_corrections, joint_errors)[0, 1]
        }
        return pd_metrics
        # 4. Add method to update PD parameters dynamically
    def update_pd_parameters(self, agent_idx=None, kp=None, kd=None, weight=None):
        """Update PD controller parameters during training."""
        if agent_idx is not None:
            if kp is not None:
                self.pd_controllers[agent_idx].kp = kp
            if kd is not None:
                self.pd_controllers[agent_idx].kd = kd
        else:
            # Update all controllers
            for controller in self.pd_controllers:
                if kp is not None:
                    controller.kp = kp
                if kd is not None:
                    controller.kd = kd
        
        if weight is not None:
            self.pd_weight = weight

    # 5. Add method to compute stability metrics
    def compute_stability_metrics(self, joint_errors, pd_corrections):
        """Compute stability metrics for the combined controller."""
        stability_metrics = {
            'error_smoothness': np.nanmean(np.abs(np.diff(joint_errors))),
            'control_smoothness': np.nanmean(np.abs(np.diff(pd_corrections))),
            'overall_stability': np.exp(-np.nanmean(np.abs(joint_errors)))
        }
        return stability_metrics

    # 6. Add adaptive PD weight adjustment
    def adapt_pd_weight(self, performance_metrics):
        """Adjust PD contribution weight based on performance."""
        if performance_metrics['error_smoothness'] > self.stability_threshold:
            self.pd_weight = min(self.pd_weight * 1.1, 0.8)  # Increase PD influence
        else:
            self.pd_weight = max(self.pd_weight * 0.9, 0.1)  # Decrease



    def train(self):
        self.training_metrics = TrainingMetrics(num_joints=self.num_joints)
        torch.autograd.set_detect_anomaly(True)
        
        # Initialize curriculum manager
        self.curriculum_manager = CurriculumManager(
            initial_difficulty=0.0, 
            max_difficulty=3.0, 
            success_threshold=0.8
        )

        for episode in range(self.num_episodes):
            self.current_episode = episode
            difficulty = self.curriculum_manager.get_current_difficulty()
            state = self.env.reset(difficulty=difficulty)
            done = False
            step = 0
            
            # Initialize tracking lists
            total_rewards = [[] for _ in range(self.num_agents)]
            total_errors = [[] for _ in range(self.num_agents)]
            total_joint_errors = []
            
            # Update learning rate
            mean_error = np.mean([np.nanmean(errors) for errors in total_errors if errors])
            self.lr_manager.step(mean_error)

            # Episode variables
            begin_distance = None
            prev_best = float('inf')
            cumulative_reward = 0

            # Log initial state
            initial_joint_angles = [
                p.getJointState(self.env.robot_id, i)[0] for i in self.env.joint_indices
            ]
            logging.info(f"Episode {episode} - Initial joint angles: {initial_joint_angles}")

            # Initialize trajectories
            trajectories = [{
                'states': [], 
                'actions': [], 
                'log_probs': [], 
                'rewards': [], 
                'dones': []
            } for _ in range(self.num_agents)]

            while not done and step < self.max_steps_per_episode:
                if step % 10 == 0:
                    logging.debug(f"Step {step}: Executing action")

                # Process current state
                processed_state_list = self._process_state(state)
                global_state = torch.cat(processed_state_list).unsqueeze(0).to(self.device)

                # Get actions and execute
                actions, log_probs = self.get_actions(state, eval_mode=True)
                next_state, rewards, done, info = self.env.step(actions)

                # ----------------------------------------------------
                # Do NOT use RewardStabilizer or AdaptiveRewardScaler
                # Just take raw rewards from the environment:
                # ----------------------------------------------------
                scaled_rewards = rewards  # No scaling or stabilization

                # Process next state
                processed_next_state_list = self._process_state(next_state)
                global_next_state = torch.cat(processed_next_state_list).unsqueeze(0).to(self.device)
                actions_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Calculate distances and errors
                current_position, current_orientation = self.get_end_effector_pose()
                target_position, target_orientation = self.get_target_pose()
                distance = compute_overall_distance(
                    current_position, target_position,
                    current_orientation, target_orientation
                )
                
                if begin_distance is None:
                    begin_distance = distance

                # Compute Jacobians and weights
                jacobian_linear = compute_jacobian_linear(self.env.robot_id, self.env.joint_indices, actions)
                jacobian_angular = compute_jacobian_angular(self.env.robot_id, self.env.joint_indices, actions)
                linear_weights, angular_weights = assign_joint_weights(jacobian_linear, jacobian_angular)

                # Calculate joint errors
                joint_errors = [
                    max(abs(actions[i] - initial_joint_angles[i]), 1e-6)
                    for i in range(self.num_agents)
                ]
                total_joint_errors.append(joint_errors)

                # Calculate rewards with your original reward function
                overall_reward, individual_rewards, prev_best, success = compute_reward(
                    distance=distance,
                    begin_distance=begin_distance,
                    prev_best=prev_best,
                    current_orientation=current_orientation,
                    target_orientation=target_orientation,
                    joint_errors=joint_errors,
                    linear_weights=linear_weights,
                    angular_weights=angular_weights,
                    success_threshold=self.env.success_threshold,
                    episode_number=episode,
                    curriculum_manager=self.curriculum_manager,
                    total_episodes=self.num_episodes
                )

                # Update trajectories and log results for each agent
                for agent_idx in range(self.num_agents):
                    total_rewards[agent_idx].append(scaled_rewards[agent_idx])
                    total_errors[agent_idx].append(joint_errors[agent_idx])
                    
                    # Update trajectories
                    trajectories[agent_idx]['states'].append(processed_state_list[agent_idx])
                    trajectories[agent_idx]['actions'].append(
                        torch.tensor(actions[agent_idx], dtype=torch.float32).to(self.device))
                    trajectories[agent_idx]['log_probs'].append(
                        torch.tensor(log_probs[agent_idx], dtype=torch.float32).to(self.device))
                    trajectories[agent_idx]['rewards'].append(
                        torch.tensor(scaled_rewards[agent_idx], dtype=torch.float32).to(self.device))
                    trajectories[agent_idx]['dones'].append(done)

                state = next_state
                step += 1

            # Update policy and log results
            actor_loss, critic_loss, entropy, policy_loss_per_agent, avg_actor_loss_list, values, returns, advantages = \
                self.update_policy(trajectories)

            # Log episode statistics
            success_status = info.get('success_per_joint', [False] * self.num_agents)
            episode_rewards = [np.sum(agent_rewards) for agent_rewards in total_rewards]

            self.training_metrics.log_episode(
                joint_errors=total_joint_errors,
                rewards=episode_rewards,
                success=success_status,
                entropy=entropy,
                actor_loss=float(actor_loss),
                critic_loss=float(critic_loss),
                policy_loss=policy_loss_per_agent,
                advantages=advantages.cpu().numpy(),
                env=self.env,
                actor_loss_per_actor=avg_actor_loss_list,  # Pass actor losses per actor
                success_threshold=self.env.success_threshold
            )

        # Save final results
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


    # Add these methods to your existing MAPPOAgent class
    def init_early_stopping(self, patience=50, min_delta=1e-4, min_episodes=100):
        self.early_stop_patience = patience
        self.early_stop_min_delta = min_delta
        self.early_stop_min_episodes = min_episodes

        # Per-agent tracking
        self.agent_early_stop_info = [{
            'best_mean_error': float('inf'),
            'best_epoch': 0,
            'no_improvement_count': 0,
            'stopped': False,
            'best_model': None,
            'error_history': deque(maxlen=100)
        } for _ in range(self.num_agents)]

        # Track how many agents have stopped
        self.num_agents_stopped = 0

    def check_early_stopping(self, agent_idx, mean_error, episode):
        agent_info = self.agent_early_stop_info[agent_idx]
        
        if episode < self.early_stop_min_episodes:
            return False
        
        # Update error history
        agent_info['error_history'].append(mean_error)
        
        # Check for improvement
        if mean_error < agent_info['best_mean_error'] - self.early_stop_min_delta:
            agent_info['best_mean_error'] = mean_error
            agent_info['best_epoch'] = episode
            agent_info['no_improvement_count'] = 0
            
            # Save the best model for this agent
            agent_info['best_model'] = self.agents[agent_idx].state_dict()
            self.save_best_model(agent_idx, self.agents[agent_idx], episode, mean_error)
            self.logger.info(f"New best model saved for agent {agent_idx} at episode {episode} with mean error: {mean_error:.6f}")
            return False
        
        # Increment no improvement count
        agent_info['no_improvement_count'] += 1
        
        # Log warning if nearing early stopping
        if agent_info['no_improvement_count'] > self.early_stop_patience * 0.7 and not agent_info['stopped']:
            self.logger.warning(
                f"Agent {agent_idx}: No improvement for {agent_info['no_improvement_count']} episodes. "
                f"Will stop after {self.early_stop_patience - agent_info['no_improvement_count']} more episodes "
                f"without improvement."
            )
        
        # Check if early stopping should be triggered for this agent
        if agent_info['no_improvement_count'] >= self.early_stop_patience and not agent_info['stopped']:
            agent_info['stopped'] = True
            self.num_agents_stopped += 1
            self.logger.info(
                f"Early stopping triggered for agent {agent_idx} at episode {episode}. "
                f"Best performance was at episode {agent_info['best_epoch']} "
                f"with mean error: {agent_info['best_mean_error']:.6f}"
            )
            return True
        
        return False


    def safe_mean(arr):
        if len(arr) == 0:
            return np.nan  # or return 0 if preferred
        return np.mean(arr)

    def restore_best_models(self):
        """Restore the best performing models"""
        for agent_idx, agent_info in enumerate(self.agent_early_stop_info):
            if agent_info['best_model'] is not None:
                self.agents[agent_idx].load_state_dict(agent_info['best_model'])
                self.logger.info(f"Restored best model for agent {agent_idx} from episode {agent_info['best_epoch']}")


    def update_policy_with_experiences(self, experiences, weights, indices):
        """
        Update policy using sampled experiences with proper loss handling.
        
        Args:
            experiences: List of experience tuples
            weights: Importance sampling weights
            indices: Indices of sampled experiences
        """
        try:
            # Convert experiences to trajectories format
            trajectories = [{'states': [], 'actions': [], 'log_probs': [], 
                            'rewards': [], 'dones': []} 
                        for _ in range(self.num_agents)]
            
            # Process each experience
            for exp_idx, experience in enumerate(experiences):
                # Convert experience components to tensors and move to device
                state = self._process_state(experience.state)
                next_state = self._process_state(experience.next_state)
                
                # Ensure actions and rewards are properly formatted
                actions = self._format_experience_data(experience.action, self.num_agents)
                rewards = self._format_experience_data(experience.reward, self.num_agents)
                
                # Get log probs for the actions
                with torch.no_grad():
                    log_probs = []
                    for agent_idx, agent in enumerate(self.agents):
                        agent_state = state[agent_idx].unsqueeze(0)
                        agent_action = torch.tensor(actions[agent_idx], dtype=torch.float32).to(self.device)
                        
                        mean, std = agent(agent_state)
                        std = torch.clamp(std * self.epsilon, min=1e-4, max=1.0)
                        dist = Normal(mean, std)
                        log_prob = dist.log_prob(agent_action).sum()
                        log_probs.append(log_prob.item())
                
                # Update trajectories for each agent
                for agent_idx in range(self.num_agents):
                    trajectories[agent_idx]['states'].append(state[agent_idx])
                    trajectories[agent_idx]['actions'].append(
                        torch.tensor(actions[agent_idx], dtype=torch.float32).to(self.device)
                    )
                    trajectories[agent_idx]['rewards'].append(
                        torch.tensor(rewards[agent_idx], dtype=torch.float32).to(self.device)
                    )
                    trajectories[agent_idx]['log_probs'].append(
                        torch.tensor(log_probs[agent_idx], dtype=torch.float32).to(self.device)
                    )
                    trajectories[agent_idx]['dones'].append(experience.done)
            
            # Apply importance sampling weights
            weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
            for agent_idx in range(self.num_agents):
                trajectories[agent_idx]['weights'] = weights_tensor
            
            # Update policy using trajectories
            actor_loss, critic_loss, entropy, policy_losses, values, returns, _ = self.update_policy(trajectories)
            
            # Compute per-experience TD-errors
            td_errors = self.compute_td_errors(trajectories)
            
            # Convert TD-errors to numpy array
            new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
            
            # Update priorities in the replay buffer
            self.her_buffer.update_priorities(indices, new_priorities)
            
            # Log the values for debugging
            self.logger.debug(f"Critic Loss: {critic_loss}, New Priorities: {new_priorities}")
            
            return new_priorities
            
        except Exception as e:
            self.logger.error(f"Error in update_policy_with_experiences: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def compute_td_errors(self, trajectories):
        """
        Compute TD-errors per experience using per-agent rewards and dones.

        Args:
            trajectories: A list of dictionaries containing trajectories for each agent.

        Returns:
            td_errors: Tensor of TD-errors per experience.
        """
        batch_size = len(trajectories[0]['states'])  # Number of time steps
        num_agents = self.num_agents  # Number of agents

        # Collect global states, per-agent rewards, and per-agent dones
        global_states = []
        rewards = []
        dones = []

        for t in range(batch_size):
            # Concatenate states from all agents at time step t
            state_t = []
            rewards_t = []
            dones_t = []
            for agent_idx in range(num_agents):
                state_t.append(trajectories[agent_idx]['states'][t])
                # Get per-agent reward
                reward = trajectories[agent_idx]['rewards'][t]
                if isinstance(reward, torch.Tensor):
                    reward = reward.item()
                rewards_t.append(reward)
                # Get per-agent done
                done = trajectories[agent_idx]['dones'][t]
                if isinstance(done, torch.Tensor):
                    done = done.item()
                dones_t.append(float(done))
            global_state = torch.cat(state_t, dim=0)  # Shape: [total_state_dim]
            global_states.append(global_state)
            rewards.append(rewards_t)  # Shape: [num_agents]
            dones.append(dones_t)      # Shape: [num_agents]

        # Convert lists to tensors
        global_states = torch.stack(global_states).to(self.device)        # Shape: [batch_size, total_state_dim]
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)  # Shape: [batch_size, num_agents]
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)      # Shape: [batch_size, num_agents]

        # Get values from critic
        with torch.no_grad():
            values = self.critic(global_states)  # Should output shape: [batch_size, num_agents]
            values = values.view(batch_size, num_agents)  # Ensure correct shape

        # Compute next values
        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]

        
        # Compute TD-targets
        td_target = rewards + self.gamma * next_values * (1 - dones)

        # Compute TD-errors
        td_errors = td_target - values

        # Flatten td_errors to shape [batch_size * num_agents] if needed
        td_errors = td_errors.view(-1)

        return td_errors



    def _format_experience_data(self, data, num_agents):
        """Helper method to format experience data"""
        if isinstance(data, (list, np.ndarray)):
            if len(data) < num_agents:
                # Extend data to match number of agents
                return list(data) + [data[-1]] * (num_agents - len(data))
            return data
        # Single value case
        return [data] * num_agents

    def _validate_experience(self, experience):
        """Validate experience data"""
        try:
            state = self._process_state(experience.state)
            if len(state) != self.num_agents:
                self.logger.warning(f"State length mismatch. Expected {self.num_agents}, got {len(state)}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Experience validation error: {str(e)}")
            return False


class AdaptiveRewardScaler:
    def __init__(self, window_size=100, initial_scale=1.0, adjustment_rate=0.02):
        self.window_size = window_size
        self.reward_history = deque(maxlen=window_size)
        self.scale = initial_scale
        self.min_scale = 0.5
        self.max_scale = 5.0
        self.adjustment_rate = adjustment_rate

    def update_scale(self, reward):
        """Update scaling factor based on reward history."""
        self.reward_history.append(reward)
        if len(self.reward_history) >= self.window_size:
            reward_mean = np.nanmean(self.reward_history)
            if abs(reward_mean) > 1.0:
                self.scale *= (1 - self.adjustment_rate)
            elif abs(reward_mean) < 0.1:
                self.scale *= (1 + self.adjustment_rate)
            self.scale = np.clip(self.scale, self.min_scale, self.max_scale)
        return self.scale
class RewardStabilizer:
    def __init__(self, 
                 window_size=100, 
                 scale_min=0.1, 
                 scale_max=2.0,
                 ewma_alpha=0.95,
                 clip_range=(-10.0, 30.0),
                 adaptive_clip_percentile=95):
        self.window_size = window_size
        self.reward_history = deque(maxlen=window_size)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.ewma_alpha = ewma_alpha
        self.ewma_value = None
        self.base_clip_range = clip_range
        self.adaptive_clip_percentile = adaptive_clip_percentile
        self.current_clip_range = clip_range

    def update_clip_range(self):
        """Dynamically adjust clip range based on recent reward history"""
        if len(self.reward_history) >= self.window_size // 2:
            rewards_array = np.array(self.reward_history)
            percentile_value = np.percentile(np.abs(rewards_array), self.adaptive_clip_percentile)
            clip_value = min(max(percentile_value, self.base_clip_range[1] * 0.5), 
                           self.base_clip_range[1] * 1.5)
            self.current_clip_range = (-clip_value, clip_value)

    def stabilize_reward(self, reward, progress_ratio=None):
        """
        Apply multiple stabilization techniques to the reward.
        
        Args:
            reward (float): The original reward value
            progress_ratio (float): Training progress (0 to 1) for adaptive scaling
        """
        # 1. Apply EWMA smoothing
        if self.ewma_value is None:
            self.ewma_value = reward
        else:
            self.ewma_value = self.ewma_alpha * self.ewma_value + (1 - self.ewma_alpha) * reward

        # 2. Adaptive scaling based on training progress
        if progress_ratio is not None:
            scale_factor = self.scale_max - (self.scale_max - self.scale_min) * progress_ratio
        else:
            scale_factor = self.scale_max

        # 3. Update clip range based on recent history
        self.reward_history.append(reward)
        self.update_clip_range()

        # 4. Apply stabilization techniques
        stabilized_reward = self.ewma_value * scale_factor

        # 5. Apply adaptive clipping
        stabilized_reward = np.clip(stabilized_reward, 
                                  self.current_clip_range[0], 
                                  self.current_clip_range[1])

        return stabilized_reward

    def get_stats(self):
        """Return current stabilization statistics"""
        return {
            'ewma_value': self.ewma_value,
            'clip_range': self.current_clip_range,
            'reward_std': np.std(self.reward_history) if len(self.reward_history) > 0 else 0,
            'reward_mean': np.mean(self.reward_history) if len(self.reward_history) > 0 else 0
        }