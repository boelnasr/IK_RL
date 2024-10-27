import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Import functional module for ELU and other activations
from torch.distributions import Normal
import numpy as np
import logging
from config import config
from collections import deque
import pybullet as p
from torch.distributions import Normal
from .reward_function import (compute_jacobian_angular,compute_overall_distance,
                            compute_quaternion_distance,compute_reward,
                            compute_position_error, assign_joint_weights,
                            compute_weighted_joint_rewards,compute_jacobian_linear)

# Initialize device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if CUDA is available and print device name
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Using CPU.")
# Assuming compute_combined_reward and TrainingMetrics are defined elsewhere
from .training_metrics import TrainingMetrics

# Multi-Head Attention Class
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Layers for query, key, and value projections
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        
        # Output projection layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Additional layers for stability and flexibility
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Scaling factor for dot-product attention
        self.scale = self.head_dim ** 0.5

    def forward(self, query, key, value, mask=None, causal_mask=False):
        batch_size = query.size(0)

        # Project inputs to queries, keys, and values
        Q = self.query_layer(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, seq_len, head_dim]
        K = self.key_layer(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_layer(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, num_heads, seq_len, seq_len]
        
        # Apply masks if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Causal mask to prevent attending to future positions
        if causal_mask:
            seq_len = scores.size(-1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Softmax to get attention probabilities, then apply dropout
        attn_weights = self.dropout(F.softmax(scores, dim=-1))

        # Calculate attention output
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # [B, seq_len, embed_dim]
        
        # Project output and apply residual connection and layer norm
        attn_output = self.out_proj(attn_output)
        attn_output = self.layer_norm(query + attn_output)  # Residual connection

        return attn_output

# Joint Actor with optional Attention
class JointActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, use_attention=False, num_heads=4):
        super(JointActor, self).__init__()
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        
        # Optional feature extractor layer before attention
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Multi-head attention layer if enabled
        if use_attention:
            self.attention = MultiHeadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # Actor network for action generation
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Tanh bounds actions between -1 and 1
        )
        
        # Log standard deviation for action exploration
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Adjusted to be a vector per action dimension

        # Weight initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, state):
        # Apply feature extraction
        x = self.feature_extractor(state)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x, x, x) + x  # Residual connection with attention

        # Pass through the actor network
        action_mean = self.actor(x)
        
        # Ensure std is not too small for numerical stability
        action_std = torch.clamp(self.log_std.exp().expand_as(action_mean), min=1e-3)
        
        return action_mean, action_std


# Centralized Critic Class
class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_agents, use_attention=False, num_heads=4, dropout=0.1):
        super(CentralizedCritic, self).__init__()
        
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents  # Number of agents

        # Feature extraction with layer normalization
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Optional multi-head attention layer
        if use_attention:
            self.attention = MultiHeadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            self.attention_norm = nn.LayerNorm(hidden_dim)

        # Core critic network with residual connections and dropout
        self.critic_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        ])
        
        # Final output layer to estimate state value for each agent
        self.output_layer = nn.Linear(hidden_dim, num_agents)

        # Weight initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, states):
        # Feature extraction and normalization
        x = self.feature_extractor(states)
        
        # Apply attention with residual connection if enabled
        if self.use_attention:
            attn_out = self.attention(x, x, x)
            x = x + self.attention_norm(attn_out)
        
        # Pass through critic layers with residual connections
        for layer in self.critic_layers:
            x = x + layer(x)
        
        # Final output layer for state value estimation per agent
        return self.output_layer(x)  # Shape: (batch_size, num_agents)


class MAPPOAgent:
    def __init__(self, env, config):
        logging.info("Starting MAPPOAgent initialization")

        # Environment and device setup
        self.env = env
        self.num_joints = env.num_joints
        self.num_agents = env.num_joints
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters and exploration strategy
        self.hidden_dim = config.get('hidden_dim', 128)
        self.epsilon = config.get('initial_epsilon', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.min_epsilon = config.get('min_epsilon', 0.01)

        # Learning rate and clip parameters for agents
        self.global_lr = config.get('lr', 1e-4)
        self.global_clip_param = config.get('clip_param', 0.1)
        self.agent_lrs = [config.get(f'lr_joint_{i}', self.global_lr) for i in range(self.num_joints)]
        self.agent_clip_params = [config.get(f'clip_joint_{i}', self.global_clip_param) for i in range(self.num_joints)]

        # Learning parameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.95)
        self.clip_param = config.get('clip_param', 0.1)
        
        # Adaptive learning rate scheduling setup
        self.scheduler_enabled = config.get('use_scheduler', True)
        self.scheduler_patience = config.get('scheduler_patience', 10)
        self.scheduler_decay_factor = config.get('scheduler_decay_factor', 0.5)
        
        # Initialize agents (actors) and optimizers
        self.agents = []
        self.optimizers = []
        self.obs_dims = [sum(np.prod(space.shape) for space in obs_space.spaces.values()) for obs_space in env.observation_space]
        
        for agent_idx, obs_dim in enumerate(self.obs_dims):
            actor = JointActor(obs_dim, self.hidden_dim, action_dim=1).to(self.device)
            optimizer = optim.Adam(actor.parameters(), lr=self.agent_lrs[agent_idx])
            self.agents.append(actor)
            self.optimizers.append(optimizer)

        # Centralized critic setup with learning rate scheduler
        self.critic = CentralizedCritic(
            state_dim=sum(self.obs_dims),
            hidden_dim=self.hidden_dim,
            num_agents=self.num_agents  # Add this argument
        ).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.global_lr)
        
        if self.scheduler_enabled:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.critic_optimizer, mode='min', factor=self.scheduler_decay_factor, patience=self.scheduler_patience
            )

        # Training parameters
        self.ppo_epochs = config.get('ppo_epochs', 15)
        self.batch_size = config.get('batch_size', 64)
        self.training_metrics = TrainingMetrics()

        # Convergence tracking with sliding window
        self.convergence_window_size = config.get('convergence_window_size', 100)
        self.rewards_window = deque(maxlen=self.convergence_window_size)
        self.success_window = deque(maxlen=self.convergence_window_size)
        self.error_window = deque(maxlen=self.convergence_window_size)

        # Optional prioritized experience replay
        self.use_prioritized_replay = config.get('use_prioritized_replay', False)
        if self.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=config.get('replay_buffer_size', 10000))

        # Logging setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('training.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Best agent tracking
        self.best_agents_state_dict = [None] * self.num_agents
        self.best_joint_errors = [float('inf')] * self.num_agents

    def update_policy(self, trajectories):
        # Unpack and preprocess trajectories (existing code remains the same)

# Ensure trajectories contain the required data for each agent
        assert all('states' in t and 'actions' in t and 'log_probs' in t and 'rewards' in t and 'dones' in t for t in trajectories), \
            "Each trajectory must contain 'states', 'actions', 'log_probs', 'rewards', and 'dones' keys."

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
        advantages, returns = self.compute_individual_gae(mean_rewards[:min_length], dones, values[:min_length])

        # Transpose advantages and returns for consistent dimensions
        advantages = advantages.transpose(0, 1)
        returns = returns.transpose(0, 1)

        # The remaining code stays the same

        # **Print Shapes for Debugging**
        # print(f"States_cat shape: {states_cat.shape}")
        # print(f"Actions_cat shape: {actions_cat.shape}")
        # print(f"Log_probs_old_cat shape: {log_probs_old_cat.shape}")
        # print(f"Advantages shape: {advantages.shape}")
        # print(f"Returns shape: {returns.shape}")

        # Prepare data for training
        dataset = torch.utils.data.TensorDataset(states_cat, actions_cat, log_probs_old_cat, advantages, returns)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Continue with the training loop...


        cumulative_policy_loss = 0.0

        # Training loop
        for _ in range(self.ppo_epochs):
            for batch in loader:
                batch_states, batch_actions, batch_log_probs_old, batch_advantages, batch_returns = batch

                # Critic update
                values = self.critic(batch_states).squeeze().view(batch_returns.shape)
                critic_loss = nn.MSELoss()(values, batch_returns)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()

                # Actor update
                for agent_idx, agent in enumerate(self.agents):
                    optimizer = self.optimizers[agent_idx]
                    start_idx = sum(self.obs_dims[:agent_idx])
                    end_idx = sum(self.obs_dims[:agent_idx + 1])
                    agent_state = batch_states[:, start_idx:end_idx]
                    agent_action = batch_actions[:, agent_idx].unsqueeze(1)
                    agent_log_prob_old = batch_log_probs_old[:, agent_idx]

                    # Reshape batch_advantages to match dimensions of ratio
                    agent_advantages = batch_advantages[:, agent_idx].unsqueeze(1)

                    mean, std = agent(agent_state)
                    dist = Normal(mean, std)
                    agent_log_prob = dist.log_prob(agent_action).sum(dim=-1, keepdim=True)  # Shape: [batch_size, 1]
                    entropy = dist.entropy().sum(dim=-1, keepdim=True)

                    # Calculate the PPO surrogate losses
                    ratio = (agent_log_prob - agent_log_prob_old.unsqueeze(1)).exp()  # Shape: [batch_size, 1]
                    surr1 = ratio * agent_advantages  # Matched dimensions
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * agent_advantages
                    actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                    optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                    optimizer.step()

                    policy_loss = -torch.mean(agent_log_prob)
                    cumulative_policy_loss += policy_loss.item()

        avg_policy_loss = cumulative_policy_loss / (self.ppo_epochs * len(loader))

        return actor_loss.item(), critic_loss.item(), entropy.mean().item(), avg_policy_loss



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
                delta = agent_rewards[step] + self.gamma * next_value * mask - agent_values[step]
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
            std = std * self.epsilon  # Scale exploration noise by epsilon
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

    def train(self, num_episodes, max_steps_per_episode=100):
        logging.info(f"Starting training for {num_episodes} episodes")

        # Define a target position and orientation at the start of training
        target_position = np.random.uniform([0.1, -0.5, 0.1], [0.5, 0.5, 0.5])
        target_orientation = p.getQuaternionFromEuler(np.random.uniform(-np.pi, np.pi, 3))
        target_id = p.loadURDF("sphere_small.urdf", target_position, globalScaling=0.05, useFixedBase=True)

        try:
            for episode in range(num_episodes):
                state = self.env.reset()
                done = False
                step = 0

                # Get initial end-effector pose using PyBullet directly
                link_state = p.getLinkState(self.env.robot_id, self.env.num_joints - 1)
                if link_state is None:
                    raise RuntimeError("Failed to get link state")

                initial_position = np.array(link_state[4])  # Position [x, y, z]
                initial_orientation = np.array(link_state[5])  # Orientation [qx, qy, qz, qw]

                # Compute initial distance
                initial_distance = compute_overall_distance(
                    initial_position, target_position,
                    initial_orientation, target_orientation
                )
                prev_best_distance = initial_distance

                # Initialize episode tracking
                total_rewards = [[] for _ in range(self.num_agents)]
                total_errors = [[] for _ in range(self.num_agents)]
                total_joint_errors = []
                trajectories = [{'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []} 
                            for _ in range(self.num_agents)]

                logging.info(f"Episode {episode} - Initializing episode variables")

                while not done and step < max_steps_per_episode:
                    actions, log_probs = self.get_actions(state)
                    next_state, rewards, done, info = self.env.step(actions)

                    # Get current end-effector pose
                    link_state = p.getLinkState(self.env.robot_id, self.env.num_joints - 1)
                    current_position = np.array(link_state[4])
                    current_orientation = np.array(link_state[5])

                    # Calculate errors and distances
                    position_error = compute_position_error(current_position, target_position)
                    orientation_error = compute_quaternion_distance(current_orientation, target_orientation)
                    overall_distance = compute_overall_distance(
                        current_position, target_position,
                        current_orientation, target_orientation
                    )

                    # Compute reward
                    reward, prev_best_distance, success = compute_reward(
                        distance=overall_distance,
                        begin_distance=initial_distance,
                        prev_best=prev_best_distance
                    )

                    # Get joint states directly from PyBullet
                    joint_states = [p.getJointState(self.env.robot_id, i) for i in range(self.env.num_joints)]
                    joint_angles = np.array([state[0] for state in joint_states])
                    previous_angles = self.env.previous_joint_angles
                    joint_errors = np.abs(joint_angles - previous_angles)
                    total_joint_errors.append(joint_errors)

                    # Calculate Jacobians using custom functions from reward_function script
                    jacobian_linear = compute_jacobian_linear(self.env.robot_id, joint_angles, current_position)
                    jacobian_angular = compute_jacobian_angular(self.env.robot_id, joint_angles, current_orientation)

                    # Compute weights and rewards
                    linear_weights, angular_weights = assign_joint_weights(jacobian_linear, jacobian_angular)
                    joint_rewards = compute_weighted_joint_rewards(
                        joint_errors, linear_weights, angular_weights, reward
                    )

                    # Store experience for each agent
                    for agent_idx in range(self.num_agents):
                        total_rewards[agent_idx].append(joint_rewards[agent_idx])
                        total_errors[agent_idx].append(joint_errors[agent_idx])

                        agent_state = self._process_state(state)[agent_idx]
                        trajectories[agent_idx]['states'].append(agent_state)
                        trajectories[agent_idx]['actions'].append(
                            torch.tensor(actions[agent_idx], dtype=torch.float32).to(self.device))
                        trajectories[agent_idx]['log_probs'].append(
                            torch.tensor(log_probs[agent_idx], dtype=torch.float32).to(self.device))
                        trajectories[agent_idx]['rewards'].append(
                            torch.tensor(joint_rewards[agent_idx], dtype=torch.float32).to(self.device))
                        trajectories[agent_idx]['dones'].append(done)

                    state = next_state
                    step += 1

                # Update policy and log metrics
                actor_loss, critic_loss, entropy, policy_loss = self.update_policy(trajectories)

                # Log and save the best-performing agents
                for agent_idx in range(self.num_agents):
                    mean_joint_error = np.mean(total_errors[agent_idx])
                    logging.info(f"Episode {episode}, Joint {agent_idx} - Mean Joint Error: {mean_joint_error:.6f}")

                    # Update best agent's model if current mean_joint_error is better
                    if mean_joint_error < self.best_joint_errors[agent_idx]:
                        self.best_joint_errors[agent_idx] = mean_joint_error
                        self.best_agents_state_dict[agent_idx] = self.agents[agent_idx].state_dict()
                        torch.save(self.best_agents_state_dict[agent_idx], f"best_agent_joint_{agent_idx}.pth")
                        logging.info(f"New best agent for joint {agent_idx} saved with Mean Joint Error: {mean_joint_error:.6f}")

                # Log training metrics for this episode
                self.training_metrics.log_episode(
                    joint_errors=total_joint_errors,
                    rewards=[sum(reward_list) for reward_list in total_rewards],
                    success=[success] * self.num_agents,
                    entropy=entropy,
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                    policy_loss=policy_loss,
                    env=self.env
                )

                # Log aggregated metrics per joint
                for agent_idx in range(self.num_agents):
                    avg_error = np.mean(total_errors[agent_idx])
                    avg_reward = np.mean(total_rewards[agent_idx])
                    logging.info(f"Episode {episode}, Joint {agent_idx} - Avg Error: {avg_error:.6f}, Avg Reward: {avg_reward:.6f}")

                # Log overall episode metrics
                avg_episode_reward = sum([sum(agent_rewards) for agent_rewards in total_rewards]) / self.num_agents
                overall_success = all([success] * self.num_agents)
                logging.info(f"Episode {episode} - Avg Episode Reward: {avg_episode_reward:.6f}, Overall Success: {overall_success}")

                # Remove the target object at the end of the episode
                p.removeBody(target_id)

        except Exception as e:
            logging.error(f"Error during training: {e}")

        # After training, save logs and plot metrics
        self.training_metrics.save_logs("training_logs.json")
        metrics = self.training_metrics.calculate_metrics(env=self.env)
        self.training_metrics.plot_metrics(metrics, self.env, show_plots=True)

    
    def _ensure_pybullet_connection(self):
        """Ensure PyBullet is properly connected"""
        try:
            if not p.isConnected():
                self.physics_client = p.connect(p.DIRECT)
            return True
        except Exception as e:
            logging.error(f"PyBullet connection error: {e}")
            return False

    def _cleanup_pybullet(self):
        """Cleanup PyBullet connection"""
        try:
            if p.isConnected():
                p.disconnect()
        except:
            pass
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
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        Initializes the Prioritized Experience Replay Buffer.

        Args:
            capacity (int): Maximum number of experiences to store.
            alpha (float): Controls the level of prioritization. alpha=0 gives uniform sampling.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, experience, priority=1.0):
        """
        Adds a new experience to the buffer with a given priority.

        Args:
            experience: The experience to store (tuple).
            priority (float): Priority for sampling this experience.
        """
        max_priority = self.priorities.max() if self.buffer else priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Samples a batch of experiences based on priority.

        Args:
            batch_size (int): Number of experiences to sample.
            beta (float): Bias correction factor for sampling probability.

        Returns:
            batch (list): Sampled experiences.
            weights (list): Importance-sampling weights for each experience.
            indices (list): Indices of sampled experiences in the buffer.
        """
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        # Compute the probability of each experience
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample experiences according to the computed probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[idx] for idx in indices]

        # Compute importance-sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        return batch, weights, indices

    def update_priorities(self, indices, priorities):
        """
        Updates the priorities of sampled experiences based on TD error.

        Args:
            indices (list): Indices of experiences to update.
            priorities (list): New priority values for the experiences.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority