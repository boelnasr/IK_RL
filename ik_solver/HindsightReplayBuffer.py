import numpy as np
import random
from collections import deque, namedtuple
import torch

Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done', 
    'goal', 'achieved_goal', 'info'
])

class HindsightReplayBuffer:
    """Enhanced replay buffer with Hindsight Experience Replay for IK tasks"""
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, k_future=4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta = beta_start
        self.k_future = k_future  # Number of future goals to sample
        
        # Main buffer storage
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.episode_buffer = []  # Temporary storage for current episode
        self.pos = 0
        
        # Validation set management
        self.validation_buffer = []
        self.validation_ratio = 0.1
        self.validation_priorities = []
        
        # Statistics tracking
        self.success_rate = deque(maxlen=100)
        self.goal_statistics = {
            'position_errors': deque(maxlen=1000),
            'orientation_errors': deque(maxlen=1000),
            'successes': deque(maxlen=1000)
        }
        
        # Episode tracking
        self.episode_count = 0
        self.total_experiences = 0
        
    def add_experience_with_info(self, state, action, reward, next_state, done, info):
        """Add experience with automatic goal extraction for IK task."""
        try:
            # Extract goals - these should be end-effector poses for IK
            achieved_goal = self._extract_achieved_goal(info)
            desired_goal = self._extract_desired_goal(info)
            
            # Create experience
            exp = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                goal=desired_goal,
                achieved_goal=achieved_goal,
                info=info
            )
            
            self.episode_buffer.append(exp)
            
            # Process episode when done
            if done:
                self._process_episode()
                self.episode_count += 1
                
        except Exception as e:
            print(f"Error adding experience: {str(e)}")
            # Don't skip - try to recover
            if done and self.episode_buffer:
                self._process_episode()
                self.episode_count += 1
    
    def _extract_desired_goal(self, info):
        """Extract desired goal (target end-effector pose) from info."""
        # For IK task, desired goal is the target end-effector pose
        if 'desired_goal' in info:
            goal = info['desired_goal']
            if isinstance(goal, dict):
                position = np.array(goal.get('position', [0, 0, 0]))
                orientation = np.array(goal.get('orientation', [0, 0, 0, 1]))
                # Ensure orientation is quaternion (4 values)
                if len(orientation) == 3:  # Euler angles
                    orientation = np.array([0, 0, 0, 1])  # Default quaternion
                return np.concatenate([position, orientation])
            else:
                return np.array(goal)
        
        # Fallback: construct from target position and orientation
        if 'target_position' in info and 'target_orientation' in info:
            position = np.array(info['target_position'])
            orientation = np.array(info['target_orientation'])
            if len(orientation) == 3:  # Euler angles
                orientation = np.array([0, 0, 0, 1])  # Default quaternion
            return np.concatenate([position, orientation])
        
        raise ValueError("Could not extract desired goal from info")
    
    def _extract_achieved_goal(self, info):
        """Extract achieved goal (current end-effector pose) from info."""
        # For IK task, achieved goal is the current end-effector pose
        if 'achieved_goal' in info:
            goal = info['achieved_goal']
            if isinstance(goal, dict):
                position = np.array(goal.get('position', [0, 0, 0]))
                orientation = np.array(goal.get('orientation', [0, 0, 0, 1]))
                if len(orientation) == 3:  # Euler angles
                    orientation = np.array([0, 0, 0, 1])  # Default quaternion
                return np.concatenate([position, orientation])
            else:
                return np.array(goal)
        
        # Fallback: use end_effector position and orientation
        if 'end_effector_position' in info and 'end_effector_orientation' in info:
            position = np.array(info['end_effector_position'])
            orientation = np.array(info['end_effector_orientation'])
            if len(orientation) == 3:  # Euler angles
                orientation = np.array([0, 0, 0, 1])  # Default quaternion
            return np.concatenate([position, orientation])
        
        raise ValueError("Could not extract achieved goal from info")
    
    def _compute_reward(self, achieved_goal, desired_goal, info=None):
        """Compute reward based on end-effector pose error."""
        # Extract position and orientation components
        achieved_pos = achieved_goal[:3]
        achieved_ori = achieved_goal[3:7]  # Quaternion
        desired_pos = desired_goal[:3]
        desired_ori = desired_goal[3:7]
        
        # Compute distances
        pos_distance = np.linalg.norm(achieved_pos - desired_pos)
        
        # Quaternion distance (dot product)
        ori_distance = 1 - np.abs(np.dot(achieved_ori, desired_ori))
        
        # Get weights from info
        pos_weight = 0.7
        ori_weight = 0.3
        if info:
            pos_weight = info.get('position_weight', 0.7)
            ori_weight = info.get('orientation_weight', 0.3)
        
        # Sparse or dense reward
        use_sparse = info.get('use_sparse_reward', False) if info else False
        
        if use_sparse:
            threshold = info.get('success_threshold', 0.05) if info else 0.05
            success = pos_distance < threshold and ori_distance < threshold * 0.5
            reward = 0.0 if success else -1.0
        else:
            # Dense reward
            reward = -(pos_weight * pos_distance + ori_weight * ori_distance)
            
            # Scale reward
            if info and 'reward_scale' in info:
                reward *= info['reward_scale']
        
        # Track statistics
        self.goal_statistics['position_errors'].append(pos_distance)
        self.goal_statistics['orientation_errors'].append(ori_distance)
        self.goal_statistics['successes'].append(pos_distance < 0.05)
        
        return reward
    
    def _process_episode(self):
        """Process episode with HER strategy."""
        if not self.episode_buffer:
            return
        
        # Track success rate
        final_exp = self.episode_buffer[-1]
        achieved = final_exp.achieved_goal
        desired = final_exp.goal
        distance = np.linalg.norm(achieved[:3] - desired[:3])
        self.success_rate.append(distance < 0.05)
        
        # Store original episode
        self._store_episode(self.episode_buffer.copy())
        
        # Apply HER - sample future goals
        episode_length = len(self.episode_buffer)
        
        for t in range(episode_length):
            # Sample k future states as goals
            future_indices = list(range(t + 1, episode_length))
            if not future_indices:
                continue
            
            # Sample up to k_future goals
            k = min(self.k_future, len(future_indices))
            future_ids = np.random.choice(future_indices, k, replace=False)
            
            for future_id in future_ids:
                # Use achieved goal from future state as new goal
                future_goal = self.episode_buffer[future_id].achieved_goal
                
                # Create new episode with relabeled goal
                her_experience = self._create_her_experience(t, future_goal)
                if her_experience:
                    self._store_single_experience(her_experience)
        
        # Clear episode buffer
        self.episode_buffer = []
    
    def _create_her_experience(self, time_idx, new_goal):
        """Create a single HER experience with relabeled goal."""
        exp = self.episode_buffer[time_idx]
        
        # Compute new reward with relabeled goal
        new_reward = self._compute_reward(
            exp.achieved_goal,
            new_goal,
            exp.info
        )
        
        # Create new experience
        her_exp = Experience(
            state=exp.state,
            action=exp.action,
            reward=new_reward,
            next_state=exp.next_state,
            done=exp.done,
            goal=new_goal,
            achieved_goal=exp.achieved_goal,
            info={**exp.info, 'her_relabeled': True}
        )
        
        return her_exp
    
    def _store_episode(self, episode):
        """Store complete episode in buffer."""
        for exp in episode:
            self._store_single_experience(exp)
    
    def _store_single_experience(self, exp):
        """Store single experience with priority."""
        # Decide if this goes to validation buffer
        if random.random() < self.validation_ratio:
            self.validation_buffer.append(exp)
            self.validation_priorities.append(1.0)
        else:
            # Add to main buffer
            if len(self.buffer) < self.capacity:
                self.buffer.append(exp)
            else:
                self.buffer[self.pos] = exp
            
            # Set initial priority (max priority for new experiences)
            max_priority = self.priorities[:len(self.buffer)].max() if len(self.buffer) > 0 else 1.0
            self.priorities[self.pos] = max(max_priority, 1.0)
            
            self.pos = (self.pos + 1) % self.capacity
        
        self.total_experiences += 1
    
    def sample(self, batch_size, beta=None):
        """Sample a batch of experiences with prioritization."""
        if len(self.buffer) == 0:
            raise ValueError("Replay buffer is empty. Cannot sample experiences.")
        
        actual_batch_size = min(batch_size, len(self.buffer))
        
        # Use provided beta or instance beta
        if beta is None:
            beta = self.beta
        
        # Get active priorities
        if len(self.buffer) < self.capacity:
            priorities = self.priorities[:len(self.buffer)]
        else:
            priorities = self.priorities
        
        # Compute sampling probabilities
        priorities = priorities + 1e-6  # Avoid zero
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), actual_batch_size, p=probs)
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, weights, indices
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences."""
        priorities = np.array(priorities)
        # Clip priorities to avoid extreme values
        priorities = np.clip(priorities, 1e-6, 100.0)
        
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.buffer):
                self.priorities[idx] = priority
    
    def update_beta(self, beta):
        """Update beta for importance sampling."""
        self.beta = min(1.0, beta)
    
    def get_statistics(self):
        """Get comprehensive buffer statistics."""
        pos_errors = list(self.goal_statistics['position_errors'])
        ori_errors = list(self.goal_statistics['orientation_errors'])
        successes = list(self.goal_statistics['successes'])
        
        stats = {
            'buffer_size': len(self.buffer),
            'validation_size': len(self.validation_buffer),
            'total_experiences': self.total_experiences,
            'episodes_processed': self.episode_count,
            'success_rate': np.mean(self.success_rate) if self.success_rate else 0.0,
            'position_error_mean': np.mean(pos_errors) if pos_errors else 0.0,
            'position_error_std': np.std(pos_errors) if pos_errors else 0.0,
            'orientation_error_mean': np.mean(ori_errors) if ori_errors else 0.0,
            'recent_success_rate': np.mean(list(successes)[-100:]) if successes else 0.0,
            'priority_stats': {
                'mean': np.mean(self.priorities[:len(self.buffer)]) if self.buffer else 0.0,
                'std': np.std(self.priorities[:len(self.buffer)]) if self.buffer else 0.0,
                'max': np.max(self.priorities[:len(self.buffer)]) if self.buffer else 0.0,
                'min': np.min(self.priorities[:len(self.buffer)]) if self.buffer else 0.0
            },
            'beta': self.beta
        }
        return stats
    
    def clear(self):
        """Clear all buffers."""
        self.buffer.clear()
        self.validation_buffer.clear()
        self.episode_buffer.clear()
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0
        self.episode_count = 0
        self.total_experiences = 0

class ValidationManager:
    """Manages validation episodes and performance tracking"""
    def __init__(self, validation_frequency=10, validation_episodes=10):
        self.validation_frequency = validation_frequency
        self.validation_episodes = validation_episodes
        self.validation_history = []
        self.best_validation_score = float('-inf')
        self.best_model_state = None
    

    def should_validate(self, episode):
        """Check if validation should be performed"""
        return episode % self.validation_frequency == 0
    
    def validate(self, agent, env):
        """Run validation episodes"""
        validation_rewards = []
        validation_success = []
        
        for _ in range(self.validation_episodes):
            episode_reward = 0
            state = env.reset()
            done = False
            
            while not done:
                action, _ = agent.get_actions(state)
                next_state, reward, done, info = env.step(action)
                episode_reward += sum(reward)
                state = next_state
            
            validation_rewards.append(episode_reward)
            validation_success.append(info.get('success', False))
        
        # Compute validation metrics
        metrics = {
            'mean_reward': np.mean(validation_rewards),
            'std_reward': np.std(validation_rewards),
            'success_rate': np.mean(validation_success),
            'min_reward': np.min(validation_rewards),
            'max_reward': np.max(validation_rewards)
        }
        
        self.validation_history.append(metrics)
        
        # Update best model if necessary
        if metrics['mean_reward'] > self.best_validation_score:
            self.best_validation_score = metrics['mean_reward']
            return True, metrics
        
        return False, metrics