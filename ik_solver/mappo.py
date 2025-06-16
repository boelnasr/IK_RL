

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Import functional module for ELU and other activations
import numpy as np
import logging
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg')
from .tester import MAPPOAgentTester
import seaborn as sns
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
from .replay_buffer import PrioritizedReplayBuffer, Experience

# Initialize device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if CUDA is available and print device name
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Using CPU.")
# Assuming compute_combined_reward and TrainingMetrics are defined elsewhere
from .training_metrics import TrainingMetrics
import json
import datetime


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
        
        # Initialize curriculum manager
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
            actor = JointActor(obs_dim, self.hidden_dim, use_attention=True, action_dim=1).to(self.device)
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

        # Process a sample state to determine state_dim
        sample_state = self.env.reset()
        processed_state_list = self._process_state(sample_state)
        global_state = torch.cat(processed_state_list).unsqueeze(0).to(self.device)
        state_dim = global_state.shape[1]

        # Initialize PD controllers with enhanced capabilities
        self.pd_controllers = []
        self.pd_weight = config.get('pd_weight', 0.3)  # Global PD weight for blending
        
        # Create enhanced PD controllers for each joint
        for _ in range(self.num_agents):
            self.pd_controllers.append(PDController(
                kp=config.get('pd_kp', 1.0),
                kd=config.get('pd_kd', 0.2),
                dt=config.get('pd_dt', 0.01),
                stability_threshold=config.get('stability_threshold', 0.05)
            ))

        # Determine action_dim
        action_dim = self.num_agents  # Assuming one action per agent

        # Initialize exploration module for intrinsic rewards
        self.exploration_module = ExplorationModule(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device
        )

        # Add or update PR buffer initialization
        self.use_prioritized_replay = config.get('use_prioritized_replay', True)
        self.buffer_update_freq = config.get('buffer_update_freq', 1)  # How often to use buffer
        self.num_buffer_updates = config.get('num_buffer_updates', 3)  # Updates per use
        
        if self.use_prioritized_replay:
            # Initialize the replay buffer with proper parameters
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=config.get('buffer_size', 50000),
                alpha=config.get('pr_alpha', 0.6),
                beta_start=config.get('pr_beta_start', 0.4),
                beta_frames=config.get('pr_beta_frames', 100000)
            )
            self.buffer_frame_idx = 0  # For beta scheduling

        # Add these new parameters
        self.value_loss_scale = config.get('value_loss_scale', 0.5)  # Critic loss scaling
        self.entropy_scale = config.get('entropy_scale', 0.001)    # Entropy scaling
        self.max_grad_norm = config.get('max_grad_norm', 0.5)    # Gradient clipping norm
        self.ratio_clip = config.get('ratio_clip', 0.20)         # Policy ratio clipping
        self.advantage_clip = config.get('advantage_clip', 2.0)   # Advantage clipping

        # Initialize training metrics
        self.training_metrics = TrainingMetrics(num_joints=self.num_joints)
        self.base_path = self.training_metrics.base_path
        
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
        self.her_buffer = HindsightReplayBuffer(
            capacity=config.get('buffer_size', 100000),
            alpha=config.get('alpha', 0.6),
            beta_start=config.get('beta_start', 0.4),
            k_future=config.get('k_future', 4)
        )
# how often to run off‐policy HER updates
        self.her_update_freq = config.get('her_update_freq', 5)
        self.her_batch_size    = config.get('her_batch_size', 256)  
        # Initialize the ValidationManager
        self.validation_manager = ValidationManager(
            validation_frequency=config.get('validation_frequency', 10),
            validation_episodes=config.get('validation_episodes', 10)
        )

        # Initialize best metrics tracking - simpler version
        self.best_metrics = {
            'mean_joint_error': [float('inf')] * self.num_agents
        }
        
        # Add for model selection
        self.reward_min = float('inf')
        self.reward_max = float('-inf')
        self.model_history = {
            'errors': [deque(maxlen=50) for _ in range(self.num_agents)],
            'rewards': [deque(maxlen=50) for _ in range(self.num_agents)],
            'success_rates': [deque(maxlen=50) for _ in range(self.num_agents)],
            'best_scores': [float('-inf')] * self.num_agents
        }

        logging.info("MAPPOAgent initialization complete.")
    def update_policy(self, trajectories, importance_weights=None):
        """
        Update policy with improved entropy handling and stability measures.
        Supports importance sampling weights from prioritized replay.
        
        Args:
            trajectories: List of trajectory dictionaries
            importance_weights: Optional tensor of importance sampling weights
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

            # Process importance weights if provided
            if importance_weights is not None:
                importance_weights = importance_weights.to(self.device)
                if importance_weights.dim() == 1:
                    importance_weights = importance_weights.unsqueeze(1)  # Make it broadcastable

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
                    #values_pred = torch.clamp(values_pred, -50.0, 50.0)
                    
                    # Apply importance weights to critic loss if available
                    if importance_weights is not None:
                        # Use weighted MSE loss
                        critic_loss = (importance_weights * (values_pred - batch_returns).pow(2)).mean()
                    else:
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
                        
                        # Apply importance weights to policy loss if available
                        if importance_weights is not None:
                            policy_loss = -(importance_weights * torch.min(surr1, surr2)).mean()
                        else:
                            policy_loss = -torch.min(surr1, surr2).mean()

                        # Compute positive entropy and handle entropy loss
                        entropy = -dist.entropy().mean()
                        #entropy = torch.clamp(entropy, min=0.001, max=10.0)

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

                        entropy_loss = self.entropy_scale * entropy  # Negative for maximization
                        #entropy_loss = torch.clamp(entropy_loss, -1.0, 1.0)

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

            # Compute averages
            n_updates = max(self.ppo_epochs * len(loader) * self.num_agents, 1)
            avg_actor_loss = total_actor_loss / n_updates
            avg_critic_loss = total_critic_loss / max(self.ppo_epochs * len(loader), 1)
            avg_entropy = total_entropy / n_updates
            avg_policy_losses = [np.mean(losses) if losses else 0.0 for losses in policy_losses]
            avg_actor_loss_list = [np.mean(losses) if losses else 0.0 for losses in actor_loss_list]
            avg_entropy_loss = total_entropy_loss / n_updates
            
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


    # Add this method to MAPPOAgent class:

    def store_experience(self, states, actions, rewards, next_states, dones, log_probs=None):
        """
        Store experiences in the prioritized replay buffer with proper priority calculation.
        
        Args:
            states: List of states for each agent
            actions: List of actions for each agent
            rewards: List of rewards for each agent
            next_states: List of next states for each agent
            dones: Done flags
            log_probs: Optional log probabilities of actions
        """
        if not self.use_prioritized_replay or self.replay_buffer is None:
            return
        
        try:
            # Process experiences for each agent
            for agent_idx in range(self.num_agents):
                # Extract agent-specific data
                state = states[agent_idx]
                action = actions[agent_idx] if isinstance(actions, list) else actions
                reward = rewards[agent_idx] if isinstance(rewards, list) else rewards
                next_state = next_states[agent_idx]
                done = dones
                log_prob = log_probs[agent_idx] if log_probs is not None and isinstance(log_probs, list) else log_probs
                
                # Create experience tuple using the Experience namedtuple
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    info={'log_prob': log_prob, 'agent_idx': agent_idx}
                )
                
                # Calculate position and joint errors for prioritization
                # Position error: distance to target position
                position_error = 0.0
                if hasattr(self.env, 'position_error'):
                    position_error = np.linalg.norm(self.env.position_error)
                elif hasattr(self.env, 'target_position') and hasattr(self.env, 'current_position'):
                    position_error = np.linalg.norm(
                        np.array(self.env.current_position) - np.array(self.env.target_position)
                    )
                
                # Joint error: difference between current joint angle and target
                joint_error = 0.0
                if hasattr(self.env, 'joint_errors'):
                    # If joint_errors is a list, get the specific agent's error
                    if isinstance(self.env.joint_errors, (list, np.ndarray)) and len(self.env.joint_errors) > agent_idx:
                        joint_error = self.env.joint_errors[agent_idx]
                    else:
                        joint_error = np.mean(self.env.joint_errors)
                
                # Calculate TD error approximation for priority
                # For IK tasks: prioritize transitions showing improvement
                if hasattr(self, 'previous_position_error'):
                    # Higher priority for transitions that reduce position error
                    position_improvement = max(0, self.previous_position_error[agent_idx] - position_error)
                    position_priority = 1.0 + 5.0 * position_improvement  # Bonus for improvement
                else:
                    # Initialize previous errors
                    self.previous_position_error = np.zeros(self.num_agents)
                    position_priority = 1.0 / (position_error + 0.1)  # Higher priority for lower errors
                
                # Update previous error for next time
                self.previous_position_error[agent_idx] = position_error
                
                # Joint priority: favor lower errors
                joint_priority = 1.0 / (joint_error + 0.1)
                
                # Reward-based priority: higher absolute rewards get higher priority
                reward_priority = abs(reward) + 0.1
                
                # Combine different priority factors with weights
                combined_priority = (
                    0.4 * position_priority + 
                    0.3 * joint_priority + 
                    0.2 * reward_priority +
                    0.1 * np.random.uniform(0.5, 1.5)  # Add randomness for exploration
                )
                
                # Add to buffer with calculated priority
                self.replay_buffer.add(experience, priority=float(combined_priority))
                
                # Log occasional priority statistics
                if np.random.random() < 0.001:  # Log roughly 0.1% of additions
                    self.logger.debug(
                        f"Agent {agent_idx} Experience Priority: {combined_priority:.4f} "
                        f"(pos: {position_priority:.2f}, joint: {joint_priority:.2f}, reward: {reward_priority:.2f})"
                    )
        
        except Exception as e:
            self.logger.error(f"Error storing experience: {str(e)}")
            self.logger.error(traceback.format_exc())

    def compute_individual_gae(self, rewards, dones, values):
        """
        Fixed GAE computation that prevents high critic losses.
        """
        batch_size = rewards.shape[0]
        num_agents = rewards.shape[1]
        
        # Pre-allocate tensors
        advantages = torch.zeros(num_agents, batch_size, device=self.device)
        returns = torch.zeros(num_agents, batch_size, device=self.device)
        
        # Scale rewards to reasonable range (do this BEFORE GAE)
        reward_scale = 0.01  # Adjust based on your reward magnitudes
        scaled_rewards = rewards * reward_scale
        
        for agent_idx in range(num_agents):
            agent_rewards = scaled_rewards[:, agent_idx]
            agent_values = values[:, agent_idx]
            
            # Bootstrap from last value
            if dones[-1]:
                next_value = 0
                next_gae = 0
            else:
                next_value = agent_values[-1].item()
                next_gae = 0
            
            # Compute GAE backwards
            for t in reversed(range(batch_size)):
                if t == batch_size - 1:
                    next_value = 0 if dones[t] else agent_values[t].item()
                else:
                    next_value = agent_values[t + 1].item()
                
                # Temporal difference error
                td_error = agent_rewards[t] + self.gamma * next_value * (1 - dones[t]) - agent_values[t]
                
                # GAE
                advantages[agent_idx, t] = td_error + self.gamma * self.tau * (1 - dones[t]) * next_gae
                next_gae = advantages[agent_idx, t].item()
                
                # Monte Carlo return (target for value function)
                returns[agent_idx, t] = agent_rewards[t] + self.gamma * next_value * (1 - dones[t])
        
        # Normalize advantages for stable policy gradients
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        
        # Clip returns to prevent extreme values
        returns = torch.clamp(returns, min=-10.0, max=10.0)
        
        return advantages, returns


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
        pd_corrections = []

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

            # Calculate PD correction
            current_angle = state[agent_idx]['joint_angle'].flatten()[0]
            
            # Calculate target joint angle using inverse kinematics
            target_position = self.env.target_position
            target_orientation = self.env.target_orientation
            
            # Use PyBullet's IK solver to get target joint angles
            target_joint_angles = p.calculateInverseKinematics(
                self.env.robot_id,
                self.env.joint_indices[-1],  # End effector link
                target_position,
                target_orientation,
                lowerLimits=[limit[0] for limit in self.env.joint_limits],
                upperLimits=[limit[1] for limit in self.env.joint_limits],
                jointRanges=[limit[1] - limit[0] for limit in self.env.joint_limits],
                restPoses=[0.0] * self.env.num_joints
            )
            
            target_angle = target_joint_angles[agent_idx]
            error = target_angle - current_angle

            pd_correction = self.pd_controllers[agent_idx].compute(error)
            pd_correction = np.clip(pd_correction, -1, 1)
            pd_corrections.append(pd_correction)

            if eval_mode:
                # Combine RL and PD actions during evaluation
                combined_action = (1 - self.pd_weight) * rl_action_value + self.pd_weight * pd_correction
                combined_action = np.clip(combined_action, -1, 1)
                actions.append(combined_action)
            else:
                # During training, use RL action directly
                actions.append(rl_action_value)

            log_probs.append(log_prob.item())

        return actions, log_probs, pd_corrections

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

    def reset_pd_controllers(self):
        """Reset all PD controllers at the start of a new episode."""
        for controller in self.pd_controllers:
            controller.reset()
        
        # Log current PD parameters
        self.logger.debug(f"PD Controllers reset. Current weight: {self.pd_weight:.3f}")

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

    def _compute_td_errors_from_batch(self, experiences):
        """
        Compute TD errors from a batch of experiences for priority updates.
        
        Args:
            experiences: List of experience namedtuples
            
        Returns:
            Tensor of TD errors
        """
        with torch.no_grad():
            # Extract batch data
            states = []
            next_states = []
            rewards = []
            dones = []
            
            for exp in experiences:
                # Handle the state format - could be list of dicts or tensor
                if isinstance(exp.state, list):
                    # Process the state list into a tensor
                    processed_state = self._process_state(exp.state)
                    states.append(torch.cat(processed_state))
                else:
                    states.append(exp.state)
                    
                if isinstance(exp.next_state, list):
                    # Process the next state list into a tensor  
                    processed_next_state = self._process_state(exp.next_state)
                    next_states.append(torch.cat(processed_next_state))
                else:
                    next_states.append(exp.next_state)
                    
                # Handle rewards - could be list or scalar
                if isinstance(exp.reward, list):
                    rewards.append(np.mean(exp.reward))  # Average across agents
                else:
                    rewards.append(exp.reward)
                    
                dones.append(exp.done)
            
            # Convert to tensors
            states_tensor = torch.stack(states).float().to(self.device)
            next_states_tensor = torch.stack(next_states).float().to(self.device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)
            
            # Get current and next state values
            current_values = self.critic(states_tensor).squeeze()
            next_values = self.critic(next_states_tensor).squeeze()
            
            # Ensure proper dimensions
            if current_values.dim() > 1:
                current_values = current_values.mean(dim=1)  # Average across agents
            if next_values.dim() > 1:
                next_values = next_values.mean(dim=1)  # Average across agents
                
            # Compute TD targets and errors
            td_targets = rewards_tensor + self.gamma * next_values * (1 - dones_tensor)
            td_errors = td_targets - current_values
            
            return td_errors

    def update_from_her(self, batch_size: int = None):
        """
        Sample a batch of HER transitions and perform one PPO update on them.

        Args:
            batch_size (int): Number of transitions to sample from the HER buffer.

        Returns:
            Tuple of training metrics or None if insufficient data
        """
        if batch_size is None:
            batch_size = self.her_batch_size
            
        # Check if we have enough stored experiences
        if len(self.her_buffer.buffer) < batch_size:
            return None

        try:
            # Sample a batch of HER experiences
            experiences, weights, indices = self.her_buffer.sample(batch_size)

            # Convert experiences into trajectory format for each agent
            trajectories = []
            for agent_idx in range(self.num_agents):
                agent_traj = {
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'dones': [],
                    'log_probs': []
                }
                
                for exp in experiences:
                    # Extract agent-specific data from experience
                    if isinstance(exp.state, list) and len(exp.state) > agent_idx:
                        agent_state = exp.state[agent_idx]
                    else:
                        agent_state = exp.state
                        
                    if isinstance(exp.action, list) and len(exp.action) > agent_idx:
                        agent_action = exp.action[agent_idx]
                    else:
                        agent_action = exp.action
                        
                    if isinstance(exp.reward, list) and len(exp.reward) > agent_idx:
                        agent_reward = exp.reward[agent_idx]
                    else:
                        agent_reward = exp.reward
                    
                    agent_traj['states'].append(agent_state)
                    agent_traj['actions'].append(agent_action)
                    agent_traj['rewards'].append(agent_reward)
                    agent_traj['dones'].append(exp.done)
                    
                    # Compute log_prob for the action (needed for PPO)
                    try:
                        with torch.no_grad():
                            if isinstance(agent_state, dict):
                                processed_state = self._process_state([agent_state])[0]
                            else:
                                processed_state = agent_state
                                
                            state_tensor = processed_state.unsqueeze(0) if processed_state.dim() == 1 else processed_state
                            mean, std = self.agents[agent_idx](state_tensor)
                            std = torch.clamp(std * self.epsilon, min=1e-4)
                            dist = Normal(mean, std)
                            
                            action_tensor = torch.tensor(agent_action, dtype=torch.float32).to(self.device)
                            log_prob = dist.log_prob(action_tensor).sum().item()
                            
                    except Exception as e:
                        self.logger.warning(f"Error computing log_prob for HER experience: {e}")
                        log_prob = 0.0
                        
                    agent_traj['log_probs'].append(log_prob)
                
                trajectories.append(agent_traj)

            # Perform policy update with HER trajectories and importance weights
            weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
            update_results = self.update_policy(trajectories, importance_weights=weights_tensor)
            
            # Compute TD errors for priority updates
            td_errors = self._compute_td_errors_from_batch(experiences)
            
            # Update priorities in HER buffer
            new_priorities = (td_errors.abs().cpu().numpy() + 1e-6).tolist()
            self.her_buffer.update_priorities(indices, new_priorities)
            
            return update_results
            
        except Exception as e:
            self.logger.error(f"Error in HER update: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
        


    def train(self):
        """
        Main training loop with curriculum learning and comprehensive metrics tracking.
        """
        self.training_metrics = TrainingMetrics(num_joints=self.num_joints)
        torch.autograd.set_detect_anomaly(True)
        
        # Initialize curriculum manager with proper agent count
        self.curriculum_manager = CurriculumManager(
            initial_difficulty=0.0,
            max_difficulty=3.0,
            success_threshold=0.8,
            num_agents=self.num_agents
        )

        for episode in range(self.num_episodes):
            self.current_episode = episode
            
            # Reset PD controllers and clear their history for new episode
            for controller in self.pd_controllers:
                controller.reset()
                controller.clear_history()
            
            # Get current difficulties for each agent before reset
            current_difficulties = [
                self.curriculum_manager.get_agent_difficulty(i) 
                for i in range(self.num_agents)
            ]
            
            # Reset environment with current difficulties
            state = self.env.reset(difficulties=current_difficulties)
            done = False
            step = 0
            
            # Initialize tracking lists for metrics
            total_rewards = [[] for _ in range(self.num_agents)]
            total_errors = [[] for _ in range(self.num_agents)]
            total_joint_errors = []
            difficulties_history = [[] for _ in range(self.num_agents)]
            pd_corrections_episode = []  # Track PD corrections for the episode
            joint_errors_episode = []    # Track joint errors for the episode
            
            # Update learning rate based on mean error
            mean_error = np.mean([np.nanmean(errors) for errors in total_errors if errors])
            self.lr_manager.step(mean_error)

            # Initialize episode variables
            begin_distance = None
            prev_best = float('inf')
            cumulative_reward = 0
            
            # Log initial state
            initial_joint_angles = [
                p.getJointState(self.env.robot_id, i)[0] 
                for i in self.env.joint_indices
            ]
            logging.info(f"Episode {episode} - Initial joint angles: {initial_joint_angles}")
            logging.info(f"Episode {episode} - Initial difficulties: {current_difficulties}")

            # Initialize trajectory storage for each agent
            trajectories = [{
                'states': [],
                'actions': [],
                'log_probs': [],
                'rewards': [],
                'dones': [],
                'difficulties': []
            } for _ in range(self.num_agents)]

            while not done and step < self.max_steps_per_episode:
                if step % 10 == 0:
                    logging.debug(f"Step {step}: Executing action")

                # Process states and get actions
                processed_state_list = self._process_state(state)
                global_state = torch.cat(processed_state_list).unsqueeze(0).to(self.device)
                actions, log_probs, pd_corrections = self.get_actions(state, eval_mode=False)
                
                # Execute action in environment
                next_state, rewards, done, info = self.env.step(actions)
                self.store_experience(state, actions, rewards, next_state, done, log_probs)
                self.her_buffer.add_experience_with_info(
                    state=state,
                    action=actions,
                    reward=rewards,
                    next_state=next_state,
                    done=done,
                    info={
                        **info,
                        'target_position': self.env.target_position,
                        'target_orientation': self.env.target_orientation,
                        # you can include any extra fields you need
                    }
                )
                # Extract difficulties from info
                agent_difficulties = info.get('agent_difficulties', [0.0] * self.num_agents)
                
                # Track difficulties
                for i in range(self.num_agents):
                    difficulties_history[i].append(agent_difficulties[i])

                # Use raw rewards without scaling
                scaled_rewards = rewards

                # Process next state and prepare tensors
                processed_next_state_list = self._process_state(next_state)
                global_next_state = torch.cat(processed_next_state_list).unsqueeze(0).to(self.device)
                actions_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Calculate performance metrics
                current_position, current_orientation = self.get_end_effector_pose()
                target_position, target_orientation = self.get_target_pose()
                distance = compute_overall_distance(
                    current_position, target_position,
                    current_orientation, target_orientation
                )
                
                if begin_distance is None:
                    begin_distance = distance

                # Compute movement weights
                jacobian_linear = compute_jacobian_linear(self.env.robot_id, self.env.joint_indices, actions)
                jacobian_angular = compute_jacobian_angular(self.env.robot_id, self.env.joint_indices, actions)
                linear_weights, angular_weights = assign_joint_weights(jacobian_linear, jacobian_angular)

                # Calculate joint errors
                joint_errors = [
                    max(abs(actions[i] - initial_joint_angles[i]), 1e-6)
                    for i in range(self.num_agents)
                ]
                total_joint_errors.append(joint_errors)
                
                # Track PD corrections and joint errors for this step
                pd_corrections_episode.append(pd_corrections)
                joint_errors_episode.append(joint_errors)
                
                # Update PD controller weights based on performance
                for i, controller in enumerate(self.pd_controllers):
                    new_weight = controller.adapt_weight()
                    if step % 100 == 0:  # Log weight changes periodically
                        logging.debug(f"Joint {i} PD weight: {new_weight:.3f}")

                # Update trajectories for each agent
                for agent_idx in range(self.num_agents):
                    # Store rewards and errors
                    total_rewards[agent_idx].append(scaled_rewards[agent_idx])
                    total_errors[agent_idx].append(joint_errors[agent_idx])
                    
                    # Update trajectory data
                    trajectories[agent_idx]['states'].append(processed_state_list[agent_idx])
                    trajectories[agent_idx]['actions'].append(
                        torch.tensor(actions[agent_idx], dtype=torch.float32).to(self.device))
                    trajectories[agent_idx]['log_probs'].append(
                        torch.tensor(log_probs[agent_idx], dtype=torch.float32).to(self.device))
                    trajectories[agent_idx]['rewards'].append(
                        torch.tensor(scaled_rewards[agent_idx], dtype=torch.float32).to(self.device))
                    trajectories[agent_idx]['dones'].append(done)
                    trajectories[agent_idx]['difficulties'].append(agent_difficulties[agent_idx])

                # Update state
                state = next_state
                step += 1
            
            # End of episode - visualize PD controller performance (optional, every N episodes)
            if episode % 100 == 0:  # Visualize every 100 episodes
                for i, controller in enumerate(self.pd_controllers):
                    viz_path = os.path.join(
                        self.base_path, 
                        f'pd_performance_agent_{i}_episode_{episode}.png'
                    )
                    controller.visualize_performance(save_path=viz_path, agent_idx=i)
                    
                    # Log PD metrics
                    pd_metrics = controller.get_metrics()
                    logging.info(
                        f"Episode {episode} - Agent {i} PD metrics: "
                        f"Mean error: {pd_metrics['recent_error_mean']:.4f}, "
                        f"Stability: {pd_metrics['stability']['overall_stability']:.4f}"
                    )
            
            # At the end of each episode, update best models if needed
            for agent_idx in range(self.num_agents):
                if total_joint_errors:  # Check if we have any errors to process
                    current_avg_error = np.nanmean([errors[agent_idx] for errors in total_joint_errors])
                    
                    # Check if this is the best performance so far
                    if current_avg_error < self.best_metrics['mean_joint_error'][agent_idx]:
                        self.best_metrics['mean_joint_error'][agent_idx] = current_avg_error
                        self.save_best_model(
                            agent_idx=agent_idx,
                            actor=self.agents[agent_idx],
                            episode=episode,
                            avg_error=current_avg_error,
                            criterion='mean_error'
                        )
                        
            # Update policy using collected trajectories
            actor_loss, critic_loss, entropy, policy_loss_per_agent, \
            avg_actor_loss_list, values, returns, advantages = self.update_policy(trajectories)
            
                        # 2) Perform off-policy updates from prioritized replay buffer
            if self.use_prioritized_replay and episode % self.buffer_update_freq == 0:
                if len(self.replay_buffer) >= self.batch_size:
                    buffer_update_metrics = []
                    for _ in range(self.num_buffer_updates):
                        result = self.update_from_buffer()
                        if result:
                            buffer_update_metrics.append(result)
                    if buffer_update_metrics:
                        bu_actor = np.mean([m[0] for m in buffer_update_metrics])
                        bu_critic = np.mean([m[1] for m in buffer_update_metrics])
                        bu_entropy = np.mean([m[2] for m in buffer_update_metrics])
                        self.logger.info(
                            f"PR updates - Actor: {bu_actor:.4f}, Critic: {bu_critic:.4f}, Entropy: {bu_entropy:.4f}"
                        )

            # 3) Periodically sample from HER buffer and do off-policy HER updates
            if episode % self.her_update_freq == 0 and len(self.her_buffer.buffer) >= self.her_batch_size:
                her_results = self.update_from_her(self.her_batch_size)
                
                if her_results is not None:
                    her_actor_loss, her_critic_loss, her_entropy = her_results[:3]
                    self.logger.info(
                        f"HER Update - Episode {episode}: "
                        f"Actor Loss: {her_actor_loss:.4f}, "
                        f"Critic Loss: {her_critic_loss:.4f}, "
                        f"Entropy: {her_entropy:.4f}"
                    )
                    
                    # Track HER metrics
                    if not hasattr(self, 'her_metrics'):
                        self.her_metrics = {
                            'actor_losses': [],
                            'critic_losses': [],
                            'entropies': [],
                            'buffer_stats': []
                        }
                        
                    self.her_metrics['actor_losses'].append(her_actor_loss)
                    self.her_metrics['critic_losses'].append(her_critic_loss)
                    self.her_metrics['entropies'].append(her_entropy)
                    
                    # Log HER buffer statistics
                    her_stats = self.her_buffer.get_statistics()
                    self.her_metrics['buffer_stats'].append(her_stats)
                    
                    if episode % (self.her_update_freq * 10) == 0:  # Log every 10th HER update
                        self.logger.info(f"HER Buffer Stats: {her_stats}")
            
            # 4) Calculate episode statistics
            success_status = info.get('success_per_joint', [False] * self.num_agents)
            episode_rewards = [np.sum(r) for r in total_rewards]
            total_joint_errors_final = info.get('joint_errors', [0.0] * self.num_agents)
            
            # Calculate average difficulties for the episode
            mean_difficulties = [np.mean(difficulties_history[i]) for i in range(self.num_agents)]

            # Log comprehensive episode data
            self.training_metrics.log_episode(
                joint_errors=total_joint_errors_final,
                rewards=episode_rewards,
                success=success_status,
                entropy=entropy,
                actor_loss=float(actor_loss),
                critic_loss=float(critic_loss),
                policy_loss=policy_loss_per_agent,
                advantages=advantages.cpu().numpy(),
                env=self.env,
                actor_loss_per_actor=avg_actor_loss_list,
                success_threshold=self.env.success_threshold,
                curriculum_difficulty=mean_difficulties
            )

        # Save final results and generate reports
        self.training_metrics.save_logs()
        metrics = self.training_metrics.calculate_metrics(env=self.env)
        self.training_metrics.plot_metrics(metrics, self.env, show_plots=False)
        self.training_metrics.save_final_report(metrics)

        return metrics

    def update_from_buffer(self):
        """
        Perform policy updates using experiences from the prioritized replay buffer.
        
        Returns:
            Tuple of update metrics or None if buffer has insufficient samples
        """
        if not self.use_prioritized_replay or len(self.replay_buffer) < self.batch_size:
            return None
        self.buffer_frame_idx += 1

        try:
            # Linear schedule β0 → 1.0 over beta_frames steps
            beta = self.replay_buffer.beta_start + \
                (1.0 - self.replay_buffer.beta_start) * \
                (self.buffer_frame_idx / self.replay_buffer.beta_frames)
            beta = min(beta, 1.0)  # cap at 1
            
            # Sample batch from replay buffer with annealed beta
            batch, weights, indices = self.replay_buffer.sample(
                self.batch_size, 
                beta=beta,
                device=self.device
            )
            
            # If batch is empty, skip this update
            if not batch:
                self.logger.warning("Empty batch returned from replay buffer. Skipping update.")
                return None
            
            # Process batch data
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_next_states = []
            batch_dones = []
            batch_log_probs = []
            
            # Handle experiences based on their format
            # Your buffer can return either dict or namedtuple experiences
            for experience in batch:
                if isinstance(experience, dict):
                    # Dict format
                    state = experience['state']
                    action = experience['action']
                    reward = experience['reward']
                    next_state = experience['next_state']
                    done = experience['done']
                    log_prob = experience.get('log_prob', None)
                else:
                    # Namedtuple format
                    state = experience.state
                    action = experience.action
                    reward = experience.reward
                    next_state = experience.next_state
                    done = experience.done
                    log_prob = None  # Namedtuples might not have log_prob
                    if hasattr(experience, 'info') and experience.info and 'log_prob' in experience.info:
                        log_prob = experience.info['log_prob']
                
                batch_states.append(state)
                batch_actions.append(action)
                batch_rewards.append(reward)
                batch_next_states.append(next_state)
                batch_dones.append(done)
                
                # Generate log_probs if not provided
                if log_prob is None:
                    with torch.no_grad():
                        try:
                            # Try to get state tensor from the state
                            if isinstance(state, list):
                                state_tensor = self._process_state(state)[0].unsqueeze(0)
                            else:
                                # If state is already a tensor
                                state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
                                
                            agent_idx = 0  # Default to first agent
                            mean, std = self.agents[agent_idx](state_tensor)
                            std = torch.clamp(std * self.epsilon, min=1e-4)
                            dist = torch.distributions.Normal(mean, std)
                            
                            action_tensor = action if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.float32).to(self.device)
                            log_prob = dist.log_prob(action_tensor).sum().item()
                        except Exception as e:
                            # If log_prob computation fails, use a default value
                            self.logger.warning(f"Error computing log_prob: {e}. Using default value.")
                            log_prob = 0.0
                            
                batch_log_probs.append(log_prob)
            
            # Convert to appropriate tensor format for policy update
            # Check if states are already tensors (from the buffer's device parameter)
            if all(isinstance(state, torch.Tensor) for state in batch_states):
                processed_states = batch_states
            else:
                processed_states = self._process_batch_states(batch_states)
                
            # Same for other batch components
            if all(isinstance(action, torch.Tensor) for action in batch_actions):
                batch_actions_tensor = torch.stack(batch_actions)
            else:
                batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.float32).to(self.device)
                
            if all(isinstance(reward, torch.Tensor) for reward in batch_rewards):
                batch_rewards_tensor = torch.stack(batch_rewards)
            else:
                batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32).to(self.device)
                
            if all(isinstance(next_state, torch.Tensor) for next_state in batch_next_states):
                processed_next_states = batch_next_states
            else:
                processed_next_states = self._process_batch_states(batch_next_states)
                
            if all(isinstance(done, torch.Tensor) for done in batch_dones):
                batch_dones_tensor = torch.stack(batch_dones)
            else:
                batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float32).to(self.device)
                
            batch_log_probs_tensor = torch.tensor(batch_log_probs, dtype=torch.float32).to(self.device)
            
            # Create trajectories-like format for policy update
            buffer_trajectories = []
            for agent_idx in range(self.num_agents):
                buffer_trajectories.append({
                    'states': processed_states[agent_idx] if isinstance(processed_states, list) else processed_states,
                    'actions': batch_actions_tensor[:, agent_idx] if batch_actions_tensor.dim() > 1 else batch_actions_tensor,
                    'rewards': batch_rewards_tensor[:, agent_idx] if batch_rewards_tensor.dim() > 1 else batch_rewards_tensor,
                    'next_states': processed_next_states[agent_idx] if isinstance(processed_next_states, list) else processed_next_states,
                    'dones': batch_dones_tensor,
                    'log_probs': batch_log_probs_tensor[:, agent_idx] if batch_log_probs_tensor.dim() > 1 else batch_log_probs_tensor
                })
            
            # Convert weights to tensor if not already
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
            
            # Update policy using the created trajectories and importance sampling weights
            update_results = self.update_policy(buffer_trajectories, importance_weights=weights)
            
            # Calculate TD errors for priority updates
            td_errors = self._compute_td_errors(processed_states, batch_actions_tensor, batch_rewards_tensor, 
                                            processed_next_states, batch_dones_tensor)
            
            # Update priorities in the buffer
            self.replay_buffer.update_priorities(indices, td_errors.abs().cpu().numpy() + 1e-6)
            
            # Increment buffer frame counter for beta annealing
            self.buffer_frame_idx += self.batch_size
            
            # Occasionally log beta progress
            if self.buffer_frame_idx % (50 * self.batch_size) == 0:
                self.logger.info(f"PR Buffer: β={beta:.4f}, frame={self.buffer_frame_idx}/{self.replay_buffer.beta_frames}")
            
            return update_results
            
        except Exception as e:
            self.logger.error(f"Error in update_from_buffer: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None


    def _process_batch_states(self, states):
        """
        Turn a batch of raw env states into one tensor [B, total_state_dim].

        states: list length B. Each item is either
                • a tensor that is already concatenated, or
                • the raw per-agent list returned by the env.
        """
        # Case 1 ─ batch is already a list of concatenated tensors
        if isinstance(states[0], torch.Tensor):
            return torch.stack(states).float().to(self.device)   # [B, total_state_dim]

        # Case 2 ─ each item is the raw per-agent list/dict from the env
        flat_batch = []
        for s in states:                         # s is list(len = num_agents)
            per_agents = self._process_state(s)  # list of tensors, one per agent
            flat_batch.append(torch.cat(per_agents))             # → 1-D tensor

        return torch.stack(flat_batch).float().to(self.device)   # [B, total_state_dim]



    def _compute_td_errors(self, states, actions, rewards, next_states, dones):
        """
        Compute TD errors for prioritized replay with proper bootstrapping and value estimation.
        
        Returns:
            Tensor of TD errors for each experience
        """
        with torch.no_grad():
            # Get current state values
            current_values = self.critic(states)
            
            # Get next state values (bootstrapping)
            next_values = self.critic(next_states)
            
            # Ensure proper tensor dimensions
            if rewards.dim() == 1:
                rewards = rewards.unsqueeze(1)
            if dones.dim() == 1:
                dones = dones.unsqueeze(1)
                
            # TD target calculation with discounting and terminal state handling
            # V(s) = r + γ(1-done)V(s')
            td_target = rewards + (1 - dones) * self.gamma * next_values
            
            # TD error is the difference between target and current value estimate
            # δ = r + γV(s') - V(s)
            td_errors = td_target - current_values
            
            # Handle multi-agent case - if we have separate errors per agent
            if td_errors.dim() > 1 and td_errors.shape[1] > 1:
                # Average across agents to get a single TD error per experience
                # This is a design choice - you could also use max error or another aggregation
                td_errors = td_errors.mean(dim=1)
            
            # Absolute TD errors for prioritization (add small epsilon for stability)
            # We return the raw TD errors and handle the abs in the priority update
            
        return td_errors


    def cleanup(self):
        """
        Clean up resources used by the agent.
        Call this when finished with the agent to release memory.
        """
        # Clear replay buffer if it exists
        if hasattr(self, 'replay_buffer'):
            self.replay_buffer.clear()
        
        # Clear other resources
        # You might want to clean up any other resources here
        
        self.logger.info("Agent resources cleaned up.")

    def plot_attention_evolution(self, episode_dir):
        """
        Create a summary plot showing how attention patterns evolved during the episode.
        
        Args:
            episode_dir: Directory containing attention visualizations for the episode
        """
        try:
            import matplotlib.pyplot as plt
            import glob
            
            # Get all step visualization files
            step_files = sorted(glob.glob(os.path.join(episode_dir, 'step_*.png')))
            
            if step_files:
                num_steps = len(step_files)
                fig = plt.figure(figsize=(20, 5 * self.num_agents))
                
                for agent_idx in range(self.num_agents):
                    plt.subplot(self.num_agents, 1, agent_idx + 1)
                    
                    # Plot attention evolution
                    attention_evolution = []
                    step_numbers = []
                    
                    for step_file in step_files:
                        step_num = int(os.path.basename(step_file).split('_')[1].split('.')[0])
                        step_numbers.append(step_num)
                        
                        # Get attention weights for this step
                        with torch.no_grad():
                            attn_weights = self.agents[agent_idx].get_attention_weights()
                            if attn_weights is not None:
                                weights = attn_weights['attention_weights'].mean(dim=1).cpu().numpy()
                                attention_evolution.append(weights)
                    
                    if attention_evolution:
                        attention_evolution = np.array(attention_evolution)
                        plt.imshow(attention_evolution, aspect='auto', cmap='viridis')
                        plt.colorbar(label='Attention Weight')
                        plt.xlabel('Attention Head Index')
                        plt.ylabel('Step Number')
                        plt.title(f'Agent {agent_idx} Attention Evolution')
                        
                plt.tight_layout()
                plt.savefig(os.path.join(episode_dir, 'attention_evolution.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Failed to create attention evolution plot: {str(e)}")
            self.logger.error(traceback.format_exc())



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


    def test_agent(self, env, num_episodes, max_steps=5000):
        tester = MAPPOAgentTester(agent=self, env=env, base_path=self.base_path)
        
        return tester.test_agent(num_episodes=num_episodes, max_steps=max_steps)


    def restore_best_models(self):
        """
        Restore the best models for all agents from the saved files.
        """
        for agent_idx in range(self.num_agents):
            agent_dir = os.path.join(self.base_path, f"agent_{agent_idx}")
            model_path = os.path.join(agent_dir, "best_model.pth")
            metadata_path = os.path.join(agent_dir, "metadata.json")

            if os.path.exists(model_path):
                self.agents[agent_idx].load_state_dict(torch.load(model_path))
                logging.info(f"Restored best model for agent {agent_idx} from {model_path}")

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logging.info(f"Loaded metadata for agent {agent_idx}: {metadata}")


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

    def combine_best_models(self):
        """
        Combine the best models for all joints into a single file.
        """
        try:
            combined_model = {
                'joint_models': {},
                'metadata': {
                    'num_joints': self.num_joints,
                    'date_combined': datetime.datetime.now().isoformat(),
                    'hidden_dim': self.hidden_dim,
                    'obs_dims': self.obs_dims
                }
            }

            # Load each best model
            for joint_idx in range(self.num_joints):
                model_path = os.path.join(self.base_path, f"best_agent_joint_{joint_idx}")
                if os.path.exists(model_path):
                    model_state = torch.load(model_path)
                    combined_model['joint_models'][f'joint_{joint_idx}'] = {
                        'state_dict': model_state,
                        'obs_dim': self.obs_dims[joint_idx],
                        'hidden_dim': self.hidden_dim
                    }
                    logging.info(f"Added model for joint {joint_idx}")

            # Save combined model
            save_path = os.path.join(self.base_path, "combined_ik_model.pth")
            torch.save(combined_model, save_path)
            logging.info(f"Successfully saved combined model to {save_path}")

            return save_path

        except Exception as e:
            logging.error(f"Error combining models: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def load_combined_model_for_inference(self, model_path):
        """
        Load combined model for inference.
        """
        try:
            combined_data = torch.load(model_path)
            for joint_idx in range(self.num_joints):
                if f'joint_{joint_idx}' in combined_data['joint_models']:
                    joint_data = combined_data['joint_models'][f'joint_{joint_idx}']
                    self.agents[joint_idx].load_state_dict(joint_data['state_dict'])
                    logging.info(f"Loaded model for joint {joint_idx}")

            logging.info("Successfully loaded all joint models")
            return True

        except Exception as e:
            logging.error(f"Error loading combined model: {str(e)}")
            logging.error(traceback.format_exc())
            return False



    def visualize_attention(self, state, save_path=None, normalize=False):
        """
        Visualize attention weights for each actor, showing separate plots for each attention head.
        
        Args:
            state: Current state input
            save_path: Optional path to save the visualization
            normalize: Whether to normalize attention weights
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os

        try:
            if save_path:
                base_dir = os.path.dirname(save_path)
                os.makedirs(base_dir, exist_ok=True)

            processed_states = self._process_state(state)

            # Process each actor
            for agent_idx, agent in enumerate(self.agents):
                if hasattr(agent, 'use_attention') and agent.use_attention:
                    try:
                        with torch.no_grad():
                            state_tensor = processed_states[agent_idx].unsqueeze(0)
                            _, _ = agent(state_tensor)  # Forward pass
                            attn_components = agent.get_attention_weights()

                            if attn_components is not None:
                                # Create a figure for this agent's attention components
                                num_heads = attn_components['attention_weights'].shape[1]  # Get number of heads
                                fig = plt.figure(figsize=(20, 5 * num_heads))
                                
                                # Plot Q, K, V matrices for each head
                                for head_idx in range(num_heads):
                                    # Query visualization
                                    plt.subplot(num_heads, 4, head_idx * 4 + 1)
                                    query = attn_components['query'].squeeze()[0, head_idx].cpu().numpy()
                                    sns.heatmap(query.reshape(-1, 1), 
                                            cmap='coolwarm',
                                            center=0,
                                            annot=True,
                                            fmt='.2f')
                                    plt.title(f'Head {head_idx} Query')
                                    
                                    # Key visualization
                                    plt.subplot(num_heads, 4, head_idx * 4 + 2)
                                    key = attn_components['key'].squeeze()[0, head_idx].cpu().numpy()
                                    sns.heatmap(key.reshape(-1, 1),
                                            cmap='coolwarm',
                                            center=0,
                                            annot=True,
                                            fmt='.2f')
                                    plt.title(f'Head {head_idx} Key')
                                    
                                    # Raw attention weights
                                    plt.subplot(num_heads, 4, head_idx * 4 + 3)
                                    raw_weights = attn_components['raw_weights'].squeeze()[head_idx].cpu().numpy()
                                    sns.heatmap(raw_weights,
                                            cmap='coolwarm',
                                            center=0,
                                            annot=True,
                                            fmt='.2f')
                                    plt.title(f'Head {head_idx} Raw Attention')
                                    
                                    # Final attention weights
                                    plt.subplot(num_heads, 4, head_idx * 4 + 4)
                                    attn_weights = attn_components['attention_weights'].squeeze()[head_idx].cpu().numpy()
                                    
                                    if normalize:
                                        attn_weights = (attn_weights - attn_weights.min()) / (
                                            attn_weights.max() - attn_weights.min() + 1e-8
                                        )
                                    
                                    sns.heatmap(attn_weights,
                                            cmap='coolwarm',
                                            vmin=0,
                                            vmax=1,
                                            annot=True,
                                            fmt='.2f')
                                    plt.title(f'Head {head_idx} Final Attention')

                                plt.suptitle(f'Actor {agent_idx} Attention Analysis', fontsize=16, y=1.02)
                                plt.tight_layout()

                                # Save or display the plot
                                if save_path:
                                    agent_path = os.path.join(base_dir, f'actor_{agent_idx}_attention.png')
                                    plt.savefig(agent_path, dpi=300, bbox_inches='tight')
                                    self.logger.info(f"Saved attention visualization for actor {agent_idx} to {agent_path}")
                                    plt.close()
                                else:
                                    plt.show()

                    except Exception as e:
                        self.logger.warning(f"Failed to visualize attention for actor {agent_idx}: {str(e)}")
                        self.logger.error(traceback.format_exc())

            # Similar visualization for critic
            if hasattr(self.critic, 'use_attention') and self.critic.use_attention:
                try:
                    with torch.no_grad():
                        global_state = torch.cat(processed_states).unsqueeze(0)
                        _ = self.critic(global_state)
                        critic_weights = self.critic.get_attention_weights()

                        if critic_weights:
                            for block_idx, block_components in enumerate(critic_weights):
                                num_heads = block_components['attention_weights'].shape[1]
                                fig = plt.figure(figsize=(20, 5 * num_heads))

                                for head_idx in range(num_heads):
                                    # Similar plotting code as above for each head
                                    # Query visualization
                                    plt.subplot(num_heads, 4, head_idx * 4 + 1)
                                    query = block_components['query'].squeeze()[0, head_idx].cpu().numpy()
                                    sns.heatmap(query.reshape(-1, 1),
                                            cmap='magma',
                                            center=0,
                                            annot=True,
                                            fmt='.2f')
                                    plt.title(f'Head {head_idx} Query')
                                    
                                    # Key visualization
                                    plt.subplot(num_heads, 4, head_idx * 4 + 2)
                                    key = block_components['key'].squeeze()[0, head_idx].cpu().numpy()
                                    sns.heatmap(key.reshape(-1, 1),
                                            cmap='magma',
                                            center=0,
                                            annot=True,
                                            fmt='.2f')
                                    plt.title(f'Head {head_idx} Key')
                                    
                                    # Raw attention weights
                                    plt.subplot(num_heads, 4, head_idx * 4 + 3)
                                    raw_weights = block_components['raw_weights'].squeeze()[head_idx].cpu().numpy()
                                    sns.heatmap(raw_weights,
                                            cmap='magma',
                                            center=0,
                                            annot=True,
                                            fmt='.2f')
                                    plt.title(f'Head {head_idx} Raw Attention')
                                    
                                    # Final attention weights
                                    plt.subplot(num_heads, 4, head_idx * 4 + 4)
                                    attn_weights = block_components['attention_weights'].squeeze()[head_idx].cpu().numpy()
                                    
                                    if normalize:
                                        attn_weights = (attn_weights - attn_weights.min()) / (
                                            attn_weights.max() - attn_weights.min() + 1e-8
                                        )
                                    
                                    sns.heatmap(attn_weights,
                                            cmap='magma',
                                            vmin=0,
                                            vmax=1,
                                            annot=True,
                                            fmt='.2f')
                                    plt.title(f'Head {head_idx} Final Attention')

                                plt.suptitle(f'Critic Block {block_idx} Attention Analysis', fontsize=16, y=1.02)
                                plt.tight_layout()

                                if save_path:
                                    critic_path = os.path.join(base_dir, f'critic_block_{block_idx}_attention.png')
                                    plt.savefig(critic_path, dpi=300, bbox_inches='tight')
                                    self.logger.info(f"Saved attention visualization for critic block {block_idx} to {critic_path}")
                                    plt.close()
                                else:
                                    plt.show()

                except Exception as e:
                    self.logger.warning(f"Failed to visualize attention for critic: {str(e)}")
                    self.logger.error(traceback.format_exc())

        except Exception as e:
            self.logger.error(f"Error in visualize_attention: {str(e)}")
            self.logger.error(traceback.format_exc())


