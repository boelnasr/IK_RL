import numpy as np
import random
from collections import deque, namedtuple
import torch
import copy
import pybullet as p

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
            state_copy = self._clone_observations(state)
            next_state_copy = self._clone_observations(next_state)

            achieved_goal = self._extract_achieved_goal(info)
            desired_goal = self._extract_desired_goal(info)
            prev_achieved_goal = self._extract_prev_achieved_goal(info, achieved_goal)

            info_dict = dict(info) if info else {}
            info_dict['prev_achieved_goal'] = prev_achieved_goal

            exp = Experience(
                state=state_copy,
                action=action,
                reward=reward,
                next_state=next_state_copy,
                done=done,
                goal=desired_goal,
                achieved_goal=achieved_goal,
                info=info_dict
            )
            
            self.episode_buffer.append(exp)
            
            # Process episode when done
            if done:
                self._process_episode()
                self.episode_count += 1
                
        except Exception as e:
            print(f"Error adding experience: {str(e)}")
            import traceback
            traceback.print_exc()
            # Don't skip - try to recover with minimal data
            if done and self.episode_buffer:
                self._process_episode()
                self.episode_count += 1
    
    @staticmethod
    def _clone_observations(observations):
        """Lightweight copy of observation dicts."""
        if observations is None:
            return None
        cloned = []
        for joint_obs in observations:
            joint_clone = {}
            for key, value in joint_obs.items():
                if isinstance(value, np.ndarray):
                    joint_clone[key] = value.copy()
                else:
                    joint_clone[key] = np.array(value, dtype=np.float32) if isinstance(value, (list, tuple)) else value
            cloned.append(joint_clone)
        return cloned

    def _extract_desired_goal(self, info):
        """Extract desired goal (target end-effector pose) from info."""
        if not info:
            return np.zeros(7, dtype=np.float32)

        desired = info.get('desired_goal')
        if desired is not None:
            return np.asarray(desired, dtype=np.float32)

        target_pos = info.get('target_position')
        target_ori = info.get('target_orientation')
        if target_pos is not None and target_ori is not None:
            return np.concatenate([np.asarray(target_pos, dtype=np.float32),
                                   np.asarray(target_ori, dtype=np.float32)])

        target_angles = info.get('target_angles')
        if target_angles is not None:
            return np.asarray(target_angles, dtype=np.float32)

        return np.zeros(7, dtype=np.float32)
    
    def _extract_achieved_goal(self, info):
        """Extract achieved goal (current end-effector pose) from info."""
        if not info:
            return np.zeros(7, dtype=np.float32)

        achieved = info.get('achieved_goal')
        if achieved is not None:
            return np.asarray(achieved, dtype=np.float32)

        if 'position_error' in info and 'orientation_error' in info:
            return np.concatenate([
                np.asarray(info['position_error'], dtype=np.float32),
                np.asarray(info['orientation_error'], dtype=np.float32)
            ])

        joint_errors = info.get('joint_errors')
        if joint_errors is not None:
            return np.asarray(joint_errors, dtype=np.float32)

        mean_error = info.get('mean_joint_error')
        if mean_error is not None:
            num_joints = len(joint_errors) if isinstance(joint_errors, (list, np.ndarray)) else 6
            return np.full(num_joints, mean_error, dtype=np.float32)

        return np.zeros(7, dtype=np.float32)

    def _extract_prev_achieved_goal(self, info, fallback):
        """Extract previous achieved goal (pre-step pose)."""
        if info and 'prev_achieved_goal' in info:
            return np.array(info['prev_achieved_goal'])
        return np.array(fallback)

    @staticmethod
    def _compute_orientation_error(current_quat, target_quat):
        """Compute orientation error as Euler angles between two quaternions."""
        current_quat = np.array(current_quat, dtype=np.float64)
        target_quat = np.array(target_quat, dtype=np.float64)

        if current_quat.shape[0] < 4 or target_quat.shape[0] < 4:
            # Fallback to simple subtraction if quaternion data is incomplete
            size = min(len(current_quat), len(target_quat), 3)
            return (target_quat[:size] - current_quat[:size]).astype(np.float32)

        # Normalize quaternions to avoid drift
        current_quat = current_quat / max(np.linalg.norm(current_quat), 1e-8)
        target_quat = target_quat / max(np.linalg.norm(target_quat), 1e-8)

        current_inv = [current_quat[0], -current_quat[1], -current_quat[2], -current_quat[3]]
        diff_quat = p.multiplyTransforms([0, 0, 0], current_inv, [0, 0, 0], target_quat.tolist())[1]
        diff_euler = p.getEulerFromQuaternion(diff_quat)
        return np.array(diff_euler, dtype=np.float32)

    def _relabel_observations(self, observations, achieved_goal, desired_goal):
        """Adjust observation dictionaries to reflect a new desired goal."""
        if observations is None:
            return None

        achieved_goal = np.array(achieved_goal)
        desired_goal = np.array(desired_goal)

        if achieved_goal.shape[0] < 3 or desired_goal.shape[0] < 3:
            return copy.deepcopy(observations)

        achieved_pos = achieved_goal[:3]
        desired_pos = desired_goal[:3]
        pos_error = (achieved_pos - desired_pos).astype(np.float32)

        if achieved_goal.shape[0] >= 7 and desired_goal.shape[0] >= 7:
            achieved_ori = achieved_goal[3:7]
            desired_ori = desired_goal[3:7]
            ori_error = self._compute_orientation_error(achieved_ori, desired_ori)
        else:
            ori_error = np.zeros(3, dtype=np.float32)

        relabeled = []
        for joint_state in observations:
            relabeled_joint = {
                'joint_angle': np.array(joint_state['joint_angle'], dtype=np.float32),
                'position_error': pos_error.copy(),
                'orientation_error': ori_error.copy()
            }
            relabeled.append(relabeled_joint)
        return relabeled
    
    def _compute_reward(self, achieved_goal, desired_goal, info=None):
        """Compute reward based on end-effector pose error."""
        try:
            achieved_goal = np.array(achieved_goal)
            desired_goal = np.array(desired_goal)
            
            # Handle the case where both goals are end-effector poses (position + orientation)
            if len(achieved_goal) >= 6 and len(desired_goal) >= 6:
                # Extract position and orientation components
                achieved_pos = achieved_goal[:3]
                achieved_ori = achieved_goal[3:7] if len(achieved_goal) >= 7 else achieved_goal[3:6]
                desired_pos = desired_goal[:3]
                desired_ori = desired_goal[3:7] if len(desired_goal) >= 7 else desired_goal[3:6]
                
                # Compute distances
                pos_distance = np.linalg.norm(achieved_pos - desired_pos)
                
                # For orientation, use simple L2 distance (works for both quaternions and euler)
                ori_distance = np.linalg.norm(achieved_ori - desired_ori)
                
                # Get weights and thresholds from info
                pos_weight = 0.7
                ori_weight = 0.3
                pos_threshold = 0.02  # 2cm
                ori_threshold = 0.1   # ~5.7 degrees
                
                if info:
                    pos_weight = info.get('position_weight', 0.7)
                    ori_weight = info.get('orientation_weight', 0.3)
                    pos_threshold = info.get('position_threshold', 0.02)
                    ori_threshold = info.get('orientation_threshold', 0.1)
                
                # Combined distance
                total_distance = pos_weight * pos_distance + ori_weight * ori_distance
                
                # Sparse reward based on thresholds
                pos_success = pos_distance < pos_threshold
                ori_success = ori_distance < ori_threshold
                overall_success = pos_success and ori_success
                
                if overall_success:
                    reward = 0.0  # Success
                else:
                    reward = -1.0  # Failure
                
                # Add dense component for better learning
                dense_component = -total_distance * 0.1
                reward += dense_component
                
                # Track statistics
                self.goal_statistics['position_errors'].append(pos_distance)
                self.goal_statistics['orientation_errors'].append(ori_distance)
                self.goal_statistics['successes'].append(overall_success)
                
                return reward
                
            else:
                # Handle other goal formats (joint angles, etc.)
                distance = np.linalg.norm(achieved_goal - desired_goal)
                
                # Get threshold from info
                threshold = 0.05
                if info:
                    threshold = info.get('joint_threshold', 0.05)
                    threshold = info.get('success_threshold', threshold)
                
                # Sparse + dense reward
                if distance < threshold:
                    reward = 0.0
                else:
                    reward = -1.0
                
                reward += -distance * 0.1  # Dense component
                
                # Track statistics
                self.goal_statistics['position_errors'].append(distance)
                self.goal_statistics['successes'].append(distance < threshold)
                
                return reward
                
        except Exception as e:
            print(f"Error computing reward: {str(e)}")
            return -1.0  # Safe fallback
    
    def _process_episode(self):
        """Process episode with HER strategy."""
        if not self.episode_buffer:
            return
        
        try:
            # Track success rate
            final_exp = self.episode_buffer[-1]
            if final_exp.achieved_goal is not None and final_exp.goal is not None:
                distance = np.linalg.norm(final_exp.achieved_goal - final_exp.goal)
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
                if k > 0:
                    future_ids = np.random.choice(future_indices, k, replace=False)
                    
                    for future_id in future_ids:
                        # Use achieved goal from future state as new goal
                        future_goal = self.episode_buffer[future_id].achieved_goal
                        
                        if future_goal is not None:
                            # Create new episode with relabeled goal
                            her_experience = self._create_her_experience(t, future_goal)
                            if her_experience:
                                self._store_single_experience(her_experience)
            
        except Exception as e:
            print(f"Error processing episode: {str(e)}")
        
        # Clear episode buffer
        self.episode_buffer = []
    
    def _create_her_experience(self, time_idx, new_goal):
        """Create a single HER experience with relabeled goal."""
        try:
            exp = self.episode_buffer[time_idx]
            
            prev_pose = exp.info.get('prev_achieved_goal') if exp.info else None
            prev_pose = np.array(prev_pose) if prev_pose is not None else exp.achieved_goal

            # Relabel observations with the new goal
            relabeled_state = self._relabel_observations(exp.state, prev_pose, new_goal)
            relabeled_next_state = self._relabel_observations(exp.next_state, exp.achieved_goal, new_goal)

            # Compute new reward with relabeled goal
            new_reward = self._compute_reward(
                exp.achieved_goal,
                new_goal,
                exp.info
            )
            
            # Create new experience
            new_info = copy.deepcopy(exp.info) if exp.info else {}
            new_info['her_relabeled'] = True
            new_info['desired_goal'] = np.array(new_goal).tolist()
            new_info['prev_achieved_goal'] = np.array(prev_pose).tolist()
            new_info['achieved_goal'] = np.array(exp.achieved_goal).tolist()

            her_exp = Experience(
                state=relabeled_state,
                action=exp.action,
                reward=new_reward,
                next_state=relabeled_next_state,
                done=exp.done,
                goal=new_goal,
                achieved_goal=exp.achieved_goal,
                info=new_info
            )
            
            return her_exp
            
        except Exception as e:
            print(f"Error creating HER experience: {str(e)}")
            return None
    
    def _store_episode(self, episode):
        """Store complete episode in buffer."""
        for exp in episode:
            self._store_single_experience(exp)
    
    def _store_single_experience(self, exp):
        """Store single experience with priority."""
        try:
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
            
        except Exception as e:
            print(f"Error storing experience: {str(e)}")
    
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
        try:
            priorities = np.array(priorities)
            # Clip priorities to avoid extreme values
            priorities = np.clip(priorities, 1e-6, 100.0)
            
            for idx, priority in zip(indices, priorities):
                if 0 <= idx < len(self.buffer):
                    self.priorities[idx] = priority
        except Exception as e:
            print(f"Error updating priorities: {str(e)}")
    
    def update_beta(self, beta):
        """Update beta for importance sampling."""
        self.beta = min(1.0, beta)
    
    def get_statistics(self):
        """Get comprehensive buffer statistics."""
        try:
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
        except Exception as e:
            print(f"Error getting statistics: {str(e)}")
            return {'error': str(e)}
    
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
            state, _ = env.reset()  # Gymnasium API returns (obs, info)
            done = False

            while not done:
                action, _ = agent.get_actions(state)
                next_state, reward, terminated, truncated, info = env.step(action)  # Gymnasium API
                done = terminated or truncated
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
