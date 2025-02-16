import numpy as np
import torch
from collections import deque, namedtuple
import random

Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done', 
    'goal', 'achieved_goal', 'info'
])
class HindsightReplayBuffer:
    """Enhanced replay buffer with Hindsight Experience Replay"""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, k_future=4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
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
        self.goal_statistics = {}


    def add_experience_with_info(self, state, action, reward, next_state, done, info):
        """Add experience with automatic goal extraction."""
        try:
            achieved_goal = self._extract_achieved_goal(state, info)
            desired_goal = self._extract_desired_goal(state, info)
        except ValueError as e:
            print(f"Error in goal extraction: {str(e)}")
            return  # Skip adding experience if goal extraction fails

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

        if done:
            self._process_episode()




    def _extract_desired_goal(self, state, info):
        """Extract desired goal from state/info"""
        desired_goal = None
        
        if info and 'desired_goal' in info:
            desired_goal = info['desired_goal']
        elif hasattr(state, 'desired_goal'):
            desired_goal = state.desired_goal
        elif 'target_position' in info:
            desired_goal = np.concatenate([
                info['target_position'],
                info.get('target_orientation', np.zeros(3))
            ])
        elif isinstance(state, dict) and 'target_joint_angles' in state:
            desired_goal = state['target_joint_angles'].flatten()
        
        if desired_goal is None:
            raise ValueError("Could not extract desired goal from state/info")
            
        return desired_goal

    def _compute_reward(self, achieved_goal, desired_goal, info=None):
        """Compute reward based on achieved and desired goals"""
        if info and info.get('use_sparse_reward', True):
            distance = np.linalg.norm(achieved_goal - desired_goal)
            threshold = info.get('success_threshold', 0.05)
            return -float(distance > threshold)
        
        # Dense reward
        if len(achieved_goal) >= 6:  # If includes orientation
            pos_distance = np.linalg.norm(achieved_goal[:3] - desired_goal[:3])
            ori_distance = np.linalg.norm(achieved_goal[3:] - desired_goal[3:])
            
            pos_weight = info.get('position_weight', 0.7) if info else 0.7
            ori_weight = info.get('orientation_weight', 0.3) if info else 0.3
            
            reward = -(pos_weight * pos_distance + ori_weight * ori_distance)
        else:
            distance = np.linalg.norm(achieved_goal - desired_goal)
            reward = -distance

        if info and 'reward_scale' in info:
            reward *= info['reward_scale']
        
        return reward

    def _process_episode(self):
        """Process episode with HER strategy"""
        if not self.episode_buffer:
            return
            
        # Store original episode
        self._store_episode(self.episode_buffer)
        
        # Apply future strategy
        episode_length = len(self.episode_buffer)
        for idx, exp in enumerate(self.episode_buffer):
            future_indices = range(idx + 1, episode_length)
            if not future_indices:
                continue
                
            k_samples = min(self.k_future, len(future_indices))
            future_ids = np.random.choice(future_indices, k_samples, replace=False)
            
            for future_id in future_ids:
                future_goal = self.episode_buffer[future_id].achieved_goal
                her_episode = self._create_hindsight_episode(future_goal)
                self._store_episode(her_episode)
        
        self.episode_buffer = []

    def _create_hindsight_episode(self, goal):
        """Create new episode with hindsight goal"""
        her_episode = []
        for exp in self.episode_buffer:
            new_reward = self._compute_reward(
                exp.achieved_goal, 
                goal,
                exp.info
            )
            
            her_exp = Experience(
                state=exp.state,
                action=exp.action,
                reward=new_reward,
                next_state=exp.next_state,
                done=exp.done,
                goal=goal,
                achieved_goal=exp.achieved_goal,
                info=exp.info
            )
            her_episode.append(her_exp)
            
        return her_episode

    def _store_episode(self, episode):
        """Store episode in main or validation buffer."""
        is_validation = random.random() < self.validation_ratio
        if is_validation:
            self.validation_buffer.extend(episode)
            self.validation_priorities.extend([1.0] * len(episode))
        else:
            for exp in episode:
                if len(self.buffer) < self.capacity:
                    self.buffer.append(exp)
                else:
                    self.buffer[self.pos] = exp
                self.priorities[self.pos] = max(self.priorities.max(), 1.0)
                self.pos = (self.pos + 1) % self.capacity
        print(f"Buffer size: {len(self.buffer)}, Validation buffer size: {len(self.validation_buffer)}")


    def sample(self, batch_size, beta=0.4):
        """Sample a batch of experiences with prioritization."""
        if len(self.buffer) == 0:
            raise ValueError("Replay buffer is empty. Cannot sample experiences.")
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)  # Adjust batch size if insufficient samples

        priorities = self.priorities[:len(self.buffer)] if len(self.buffer) < self.capacity else self.priorities

        # Handle case where all priorities are zero
        if np.sum(priorities) == 0:
            priorities += 1e-6  # Small value to prevent division by zero

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        experiences = [self.buffer[idx] for idx in indices]

        return experiences, weights, indices


    def sample_validation(self, batch_size):
        """Sample from validation buffer"""
        if len(self.validation_buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.validation_buffer), batch_size)
        experiences = [self.validation_buffer[idx] for idx in indices]
        weights = np.ones(batch_size)
        
        return experiences, weights, indices
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampling"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def get_statistics(self):
        """Get buffer statistics"""
        stats = {
            'buffer_size': len(self.buffer),
            'validation_size': len(self.validation_buffer),
            'success_rate': np.mean(self.success_rate) if self.success_rate else 0.0,
            'priority_stats': {
                'mean': np.mean(self.priorities[:len(self.buffer)]),
                'std': np.std(self.priorities[:len(self.buffer)]),
                'max': np.max(self.priorities[:len(self.buffer)]),
                'min': np.min(self.priorities[:len(self.buffer)])
            },
            'goal_statistics': self.goal_statistics
        }
        return stats
    

    def _extract_desired_goal(self, state, info):
        """Extract desired goal with multiple fallback options."""
        try:
            # Primary method: direct desired_goal from info
            if info and 'desired_goal' in info:
                goal = info['desired_goal']
                if isinstance(goal, dict) and 'position' in goal and 'orientation' in goal:
                    return np.concatenate([
                        np.array(goal['position']),
                        np.array(goal['orientation'])
                    ])

            # Secondary method: separate target fields
            if info and 'target_position' in info and 'target_orientation' in info:
                return np.concatenate([
                    np.array(info['target_position']),
                    np.array(info['target_orientation'])
                ])

            # Tertiary method: from state dictionary
            if isinstance(state, dict):
                if 'target_pose' in state:
                    target = state['target_pose']
                    return np.concatenate([
                        np.array(target['position']),
                        np.array(target['orientation'])
                    ])
                elif 'target_position' in state and 'target_orientation' in state:
                    return np.concatenate([
                        np.array(state['target_position']),
                        np.array(state['target_orientation'])
                    ])

            # Final fallback: use errors to reconstruct goal
            if info and 'position_error' in info and 'orientation_error' in info:
                return np.concatenate([
                    np.array(info['position_error']),
                    np.array(info['orientation_error'])
                ])

            # If we get here, log the contents and raise error
            print("Failed to extract goal from:")
            print("State:", state)
            print("Info:", info)
            raise ValueError("Could not extract desired goal from state/info")

        except Exception as e:
            print(f"Error in goal extraction: {str(e)}")
            print("State:", state)
            print("Info:", info)
            raise


    def extract_achieved_goal(self, state, info):
        """Extract achieved goal from state/info"""
        try:
            # Try to get from info first
            if 'achieved_goal' in info:
                return info['achieved_goal']
            elif 'end_effector_position' in info and 'end_effector_orientation' in info:
                return np.concatenate([
                    info['end_effector_position'],
                    info['end_effector_orientation']
                ])
            elif isinstance(state, dict) and 'joint_angle' in state:
                return np.concatenate([
                    state['joint_angle'].flatten(),
                    state.get('position_error', np.zeros(3)).flatten(),
                    state.get('orientation_error', np.zeros(3)).flatten()
                ])
            
            # Fallback: try to extract from state
            if hasattr(state, 'achieved_goal'):
                return state.achieved_goal
            
            raise ValueError("Could not extract achieved goal from state/info")
            
        except Exception as e:
            raise ValueError(f"Error extracting achieved goal: {str(e)}")
    def _extract_achieved_goal(self, state, info):
        """Extract achieved goal from state/info with fallback."""
        achieved_goal = None
        
        # Attempt to extract from known fields in info and state
        if info and 'achieved_goal' in info:
            achieved_goal = info['achieved_goal']
        elif 'end_effector_position' in info:
            achieved_goal = np.concatenate([
                info['end_effector_position'],
                info.get('end_effector_orientation', np.zeros(3))
            ])
        elif isinstance(state, list) and 'joint_angle' in state[0]:
            # Assumes `state` is a list of dicts for each joint
            achieved_goal = np.concatenate([
                np.array([joint['joint_angle'].flatten()[0] for joint in state]),
                np.array(state[0].get('position_error', np.zeros(3)).flatten()),
                np.array(state[0].get('orientation_error', np.zeros(3)).flatten())
            ])
        
        # Check if extraction was successful; raise an error if not
        if achieved_goal is None:
            print("State Type:", type(state))
            print("State Contents:", state)
            print("Info Type:", type(info))
            print("Info Contents:", info)
            raise ValueError("Could not extract achieved goal from state/info")
        
        return achieved_goal





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