import pybullet as p
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def wrap_angle_to_pi(angle):
    """
    Wrap angle to the range [-π, π].
    
    Args:
        angle (float or np.array): Angle(s) to wrap
        
    Returns:
        float or np.array: Wrapped angle(s) in [-π, π]
    """
    # Use modulo operation and adjust to [-π, π]
    wrapped = np.fmod(angle + np.pi, 2 * np.pi) - np.pi
    
    # Handle the edge case where fmod might return -π instead of π
    wrapped = np.where(wrapped == -np.pi, np.pi, wrapped)
    
    return wrapped


def compute_position_error(current_position, target_position):
    """
    Enhanced position error calculation with robust handling.
    
    Args:
        current_position (np.array): Current position [x, y, z]
        target_position (np.array): Target position [x, y, z]
        
    Returns:
        float: Euclidean distance between positions
    """
    try:
        current_position = np.array(current_position, dtype=np.float64)
        target_position = np.array(target_position, dtype=np.float64)
        error = np.linalg.norm(current_position - target_position)
        return max(float(error), 1e-8)
    except Exception as e:
        logging.warning(f"Error in position calculation: {e}")
        return 1e-3


def compute_quaternion_distance(q1, q2):
    """
    Robust quaternion distance calculation with proper angle wrapping.
    Returns the shortest angular distance between two quaternions in [-π, π].
    
    Args:
        q1 (np.array): First quaternion [x, y, z, w]
        q2 (np.array): Second quaternion [x, y, z, w]
        
    Returns:
        float: Shortest angular distance in [-π, π]
    """
    try:
        q1 = np.array(q1, dtype=np.float64)
        q2 = np.array(q2, dtype=np.float64)
        
        # Normalize quaternions to handle numerical errors
        q1_norm = np.linalg.norm(q1)
        q2_norm = np.linalg.norm(q2)
        
        if q1_norm < 1e-8 or q2_norm < 1e-8:
            return 0.0
            
        q1 = q1 / q1_norm
        q2 = q2 / q2_norm
        
        # Compute dot product (can be negative due to quaternion double cover)
        dot_product = np.dot(q1, q2)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Handle quaternion double cover: choose the shorter rotation
        # If dot_product < 0, the angle is > π/2, so use -q2 for shorter path
        if dot_product < 0:
            dot_product = -dot_product
            
        # Avoid numerical issues near 1.0
        if dot_product > 0.9999:
            return 0.0
            
        # Compute angle and wrap to [-π, π]
        angle = 2 * np.arccos(dot_product)
        wrapped_angle = wrap_angle_to_pi(angle)
        
        return float(wrapped_angle)
        
    except Exception as e:
        logging.warning(f"Error in quaternion calculation: {e}")
        return 0.0


def compute_orientation_error_euler(current_orientation, target_orientation):
    """
    Compute orientation error in Euler angles, properly wrapped to [-π, π].
    
    Args:
        current_orientation (np.array): Current quaternion [x, y, z, w]
        target_orientation (np.array): Target quaternion [x, y, z, w]
        
    Returns:
        np.array: Orientation error for each axis (roll, pitch, yaw) in [-π, π]
    """
    try:
        # Convert quaternions to Euler angles
        current_euler = np.array(p.getEulerFromQuaternion(current_orientation))
        target_euler = np.array(p.getEulerFromQuaternion(target_orientation))
        
        # Compute angular differences
        euler_diff = current_euler - target_euler
        
        # Wrap each angle component to [-π, π]
        wrapped_diff = wrap_angle_to_pi(euler_diff)
        
        return wrapped_diff
        
    except Exception as e:
        logging.warning(f"Error in Euler orientation calculation: {e}")
        return np.zeros(3)


def compute_joint_angle_error(current_angle, target_angle):
    """
    Compute joint angle error properly wrapped to [-π, π].
    
    Args:
        current_angle (float): Current joint angle
        target_angle (float): Target joint angle
        
    Returns:
        float: Joint angle error in [-π, π]
    """
    try:
        # Compute raw difference
        error = current_angle - target_angle
        
        # Wrap to [-π, π] to find equivalent angle
        wrapped_error = wrap_angle_to_pi(error)
        
        return float(wrapped_error)
        
    except Exception as e:
        logging.warning(f"Error in joint angle calculation: {e}")
        return 0.0


def compute_overall_distance(current_position, target_position, current_orientation, target_orientation):
    """
    Weighted combination of position and orientation errors with proper angle wrapping.
    
    Args:
        current_position (np.array): Current position [x, y, z]
        target_position (np.array): Target position [x, y, z]
        current_orientation (np.array): Current quaternion [x, y, z, w]
        target_orientation (np.array): Target quaternion [x, y, z, w]
        
    Returns:
        float: Combined distance metric
    """
    try:
        position_error = compute_position_error(current_position, target_position)
        
        # Use the properly wrapped quaternion distance
        orientation_error = compute_quaternion_distance(current_orientation, target_orientation)
        
        # For overall distance, we want the magnitude, so take absolute value
        orientation_error = abs(orientation_error)
        
        # Weight position more heavily than orientation for most IK tasks
        overall_distance = 0.7 * position_error + 0.3 * orientation_error
        return max(float(overall_distance), 1e-8)
        
    except Exception as e:
        logging.warning(f"Error in overall distance calculation: {e}")
        return 1.0


def compute_reward(
    distance: float,
    begin_distance: float,
    prev_best: float,
    current_orientation: np.ndarray,
    target_orientation: np.ndarray,
    joint_errors: list,
    linear_weights: list,
    angular_weights: list,
    difficulties: list,
    episode_number: int,
    total_episodes: int,
    success_threshold: float = 0.01,
    time_penalty: float = -0.001,
    smoothing_window: int = 10,
    exploration_bonus: float = 0.01
) -> tuple:
    """
    Enhanced reward function with proper angle wrapping and robust error handling.
    
    Args:
        distance (float): Current distance to target
        begin_distance (float): Initial distance to target
        prev_best (float): Best distance achieved so far
        current_orientation (np.array): Current orientation quaternion
        target_orientation (np.array): Target orientation quaternion
        joint_errors (list): List of joint angle errors
        linear_weights (list): Linear Jacobian weights for each joint
        angular_weights (list): Angular Jacobian weights for each joint
        difficulties (list): Difficulty levels for each agent
        episode_number (int): Current episode number
        total_episodes (int): Total training episodes
        success_threshold (float): Threshold for success determination
        time_penalty (float): Per-step time penalty
        smoothing_window (int): Window size for error smoothing
        exploration_bonus (float): Bonus for exploration
        
    Returns:
        tuple: (final_rewards, joint_specific_rewards, new_best, success)
    """
    
    # ====== CRITICAL FIX 1: Input Validation ======
    try:
        # Validate and bound all inputs
        distance = max(float(distance), 1e-8)
        begin_distance = max(float(begin_distance), 1e-8)
        prev_best = max(float(prev_best), 1e-8)
        episode_number = max(int(episode_number), 1)
        total_episodes = max(int(total_episodes), 1)
        success_threshold = np.clip(float(success_threshold), 1e-6, 1.0)
        
        # Ensure we have valid joint errors
        num_joints = len(joint_errors)
        if num_joints == 0:
            logging.warning("No joint errors provided")
            return np.array([]), [], prev_best, False
            
        # Sanitize joint errors - CRITICAL: wrap angles to [-π, π] and take absolute value for magnitude
        joint_errors = [abs(wrap_angle_to_pi(float(e))) for e in joint_errors]
        
        # Validate and normalize weights
        if len(linear_weights) < num_joints:
            linear_weights = list(linear_weights) + [1.0/num_joints] * (num_joints - len(linear_weights))
        if len(angular_weights) < num_joints:
            angular_weights = list(angular_weights) + [1.0/num_joints] * (num_joints - len(angular_weights))
            
        linear_weights = np.array(linear_weights[:num_joints])
        angular_weights = np.array(angular_weights[:num_joints])
        
        # Ensure weights are positive and normalized
        linear_weights = np.abs(linear_weights)
        angular_weights = np.abs(angular_weights)
        linear_weights = linear_weights / (np.sum(linear_weights) + 1e-8)
        angular_weights = angular_weights / (np.sum(angular_weights) + 1e-8)
        
        # Validate difficulties
        if len(difficulties) < num_joints:
            difficulties = list(difficulties) + [1.0] * (num_joints - len(difficulties))
        difficulties = [np.clip(float(d), 0.1, 5.0) for d in difficulties[:num_joints]]
        
    except Exception as e:
        logging.error(f"Critical input validation error: {e}")
        # Return safe fallback
        fallback_size = max(len(joint_errors), 1)
        return np.zeros(fallback_size), [0.0] * fallback_size, prev_best, False

    # ====== CRITICAL FIX 2: Conservative Constants ======
    MAX_REWARD = 3.0        # Reduced from 10.0 for stability
    MIN_REWARD = -1.0       # Less severe penalties
    POSITION_SCALE = 0.3    # Reduced scaling factors
    ORIENTATION_SCALE = 0.2
    IMPROVEMENT_SCALE = 0.1
    ERROR_PENALTY_SCALE = 0.05  # Much gentler penalties
    SUCCESS_BONUS_BASE = 1.0    # Reasonable success bonus

    # ====== CRITICAL FIX 3: Simplified State Tracking ======
    try:
        # Initialize simplified tracking
        if not hasattr(compute_reward, 'agent_data'):
            compute_reward.agent_data = [{
                'error_history': deque(maxlen=smoothing_window),
                'best_error': float('inf'),
                'total_steps': 0
            } for _ in range(num_joints)]
        
        # Ensure we have data for all agents
        while len(compute_reward.agent_data) < num_joints:
            compute_reward.agent_data.append({
                'error_history': deque(maxlen=smoothing_window),
                'best_error': float('inf'),
                'total_steps': 0
            })
            
    except Exception as e:
        logging.error(f"State tracking initialization failed: {e}")
        # Continue with fallback computation

    # ====== CRITICAL FIX 4: Robust Global Metrics ======
    try:
        # Use the properly wrapped quaternion distance
        orientation_error = compute_quaternion_distance(current_orientation, target_orientation)
        # For distance metrics, we want the magnitude
        orientation_error_magnitude = abs(orientation_error)
        orientation_error_magnitude = np.clip(orientation_error_magnitude, 0.0, np.pi)
        
        episode_progress = np.clip(episode_number / total_episodes, 0.1, 1.0)
        
        # Bounded normalization
        norm_distance = np.clip(distance / (begin_distance + 1e-8), 0.0, 5.0)
        norm_orientation = np.clip(orientation_error_magnitude / np.pi, 0.0, 1.0)
        
    except Exception as e:
        logging.warning(f"Global metrics calculation error: {e}")
        norm_distance = 1.0
        norm_orientation = 1.0
        episode_progress = 0.5

    # ====== CRITICAL FIX 5: Individual Agent Rewards with Bounds ======
    final_rewards = np.zeros(num_joints)
    joint_specific_rewards = []
    
    for agent_idx in range(num_joints):
        try:
            current_error = joint_errors[agent_idx]
            agent_difficulty = difficulties[agent_idx]
            
            # Update tracking safely
            if hasattr(compute_reward, 'agent_data') and agent_idx < len(compute_reward.agent_data):
                agent_data = compute_reward.agent_data[agent_idx]
                agent_data['error_history'].append(current_error)
                agent_data['total_steps'] += 1
                
                # Simple improvement calculation
                improvement = 0.0
                if len(agent_data['error_history']) >= 3:
                    recent = list(agent_data['error_history'])
                    mid = len(recent) // 2
                    old_avg = np.mean(recent[:mid]) if mid > 0 else current_error
                    new_avg = np.mean(recent[mid:])
                    
                    if old_avg > 1e-8:
                        improvement = (old_avg - new_avg) / old_avg
                        improvement = np.clip(improvement, -1.0, 1.0)
            else:
                improvement = 0.0

            # ====== BOUNDED REWARD COMPONENTS ======
            
            # 1. Position contribution (bounded)
            position_reward = POSITION_SCALE * (1.0 - norm_distance) * linear_weights[agent_idx]
            position_reward = np.clip(position_reward, 0.0, 0.5)
            
            # 2. Orientation contribution (bounded)
            orientation_reward = ORIENTATION_SCALE * (1.0 - norm_orientation) * angular_weights[agent_idx]
            orientation_reward = np.clip(orientation_reward, 0.0, 0.3)
            
            # 3. Individual performance (bounded)
            normalized_error = np.clip(current_error / np.pi, 0.0, 1.0)
            performance_reward = 0.2 * (1.0 - normalized_error)
            
            # 4. Improvement reward (bounded)
            improvement_reward = IMPROVEMENT_SCALE * np.tanh(improvement) * episode_progress
            improvement_reward = np.clip(improvement_reward, -0.1, 0.1)
            
            # 5. Gentle error penalty (bounded)
            error_penalty = -ERROR_PENALTY_SCALE * normalized_error
            error_penalty = np.clip(error_penalty, -0.1, 0.0)
            
            # 6. Success bonus (controlled)
            success_bonus = 0.0
            if current_error <= success_threshold:
                success_bonus = SUCCESS_BONUS_BASE * episode_progress
                success_bonus = np.clip(success_bonus, 0.0, 1.0)
            
            # 7. Gentle time penalty
            scaled_time_penalty = time_penalty * 0.5 * (linear_weights[agent_idx] + angular_weights[agent_idx])
            scaled_time_penalty = np.clip(scaled_time_penalty, -0.01, 0.0)
            
            # ====== COMBINE WITH STRICT BOUNDS ======
            agent_reward = (
                position_reward +
                orientation_reward +
                performance_reward +
                improvement_reward +
                error_penalty +
                success_bonus +
                scaled_time_penalty
            )
            
            # Apply difficulty scaling (bounded)
            agent_reward *= np.clip(agent_difficulty, 0.5, 2.0)
            
            # Final bounds check
            agent_reward = np.clip(agent_reward, MIN_REWARD, MAX_REWARD)
            
            # Ensure finite value
            if not np.isfinite(agent_reward):
                agent_reward = 0.0
                
            final_rewards[agent_idx] = float(agent_reward)
            joint_specific_rewards.append(float(agent_reward))
            
        except Exception as e:
            logging.warning(f"Error calculating reward for agent {agent_idx}: {e}")
            # Safe fallback
            fallback_reward = -0.1
            final_rewards[agent_idx] = fallback_reward
            joint_specific_rewards.append(fallback_reward)

    # ====== CRITICAL FIX 6: Gentle Normalization ======
    try:
        # Only normalize if rewards are extremely varied
        reward_std = np.std(final_rewards)
        if reward_std > 2.0:  # Only if very high variance
            reward_mean = np.mean(final_rewards)
            final_rewards = (final_rewards - reward_mean) / (reward_std + 1e-8)
            final_rewards = np.clip(final_rewards, MIN_REWARD, MAX_REWARD)
            joint_specific_rewards = final_rewards.tolist()
            
    except Exception as e:
        logging.warning(f"Reward normalization error: {e}")

    # ====== CRITICAL FIX 7: Robust Success Determination ======
    try:
        individual_successes = np.array([e <= success_threshold for e in joint_errors])
        success = np.mean(individual_successes) >= 0.7
        new_best = min(prev_best, distance)
    except Exception:
        success = False
        new_best = prev_best

    return final_rewards, joint_specific_rewards, new_best, success


def adaptive_clip(rewards, stats):
    """
    IMPROVED: Much more conservative adaptive clipping.
    
    Args:
        rewards (np.array): Rewards to clip
        stats (dict): Statistics for adaptive clipping
        
    Returns:
        np.array: Clipped rewards
    """
    try:
        rewards = np.array(rewards)
        if len(rewards) == 0:
            return rewards
            
        # Handle non-finite values
        finite_mask = np.isfinite(rewards)
        if not np.any(finite_mask):
            return np.zeros_like(rewards)
            
        finite_rewards = rewards[finite_mask]
        curr_mean = np.mean(finite_rewards)
        curr_std = np.std(finite_rewards) + 1e-8
        
        # CRITICAL: More conservative updates
        alpha = 0.95  # Slower adaptation
        stats['running_mean'] = alpha * stats.get('running_mean', curr_mean) + (1 - alpha) * curr_mean
        stats['running_std'] = alpha * stats.get('running_std', curr_std) + (1 - alpha) * curr_std
        
        # CRITICAL: Gentler clipping bounds
        clip_factor = 1.0  # Much smaller clipping range
        lower_bound = stats['running_mean'] - clip_factor * stats['running_std']
        upper_bound = stats['running_mean'] + clip_factor * stats['running_std']
        
        # Apply conservative bounds
        lower_bound = max(lower_bound, -2.0)  # Never clip below -2
        upper_bound = min(upper_bound, 5.0)   # Never clip above 5
        
        clipped_rewards = np.copy(rewards)
        clipped_rewards[finite_mask] = np.clip(finite_rewards, lower_bound, upper_bound)
        
        return clipped_rewards
        
    except Exception as e:
        logging.warning(f"Adaptive clipping error: {e}")
        return np.clip(rewards, -2.0, 5.0)


# ====== JACOBIAN COMPUTATION FUNCTIONS ======

def compute_jacobian_linear(robot_id, joint_indices, joint_angles):
    """
    Robust linear Jacobian computation with comprehensive error handling.
    
    Args:
        robot_id (int): PyBullet robot ID
        joint_indices (list): List of joint indices
        joint_angles (list): List of joint angles
        
    Returns:
        np.array: Linear Jacobian matrix (3 x n)
    """
    try:
        num_joints = len(joint_indices)
        if num_joints == 0:
            return np.zeros((3, 1))
            
        J_linear = np.zeros((3, num_joints))
        
        ee_state = p.getLinkState(robot_id, joint_indices[-1])
        ee_pos = np.array(ee_state[4], dtype=np.float64)
        
        for i, joint_idx in enumerate(joint_indices):
            try:
                joint_info = p.getJointInfo(robot_id, joint_idx)
                joint_state = p.getLinkState(robot_id, joint_idx)
                
                joint_pos = np.array(joint_state[4], dtype=np.float64)
                joint_axis = np.array(joint_info[13], dtype=np.float64)
                
                # Robust axis normalization
                axis_norm = np.linalg.norm(joint_axis)
                if axis_norm > 1e-8:
                    joint_axis = joint_axis / axis_norm
                else:
                    joint_axis = np.array([0, 0, 1], dtype=np.float64)
                
                r = ee_pos - joint_pos
                cross_product = np.cross(joint_axis, r)
                
                # Ensure finite values
                if np.all(np.isfinite(cross_product)):
                    J_linear[:, i] = cross_product
                else:
                    J_linear[:, i] = [0, 0, 0]
                    
            except Exception as e:
                logging.warning(f"Error computing Jacobian for joint {i}: {e}")
                J_linear[:, i] = [0, 0, 0]
        
        return J_linear
        
    except Exception as e:
        logging.error(f"Jacobian linear computation failed: {e}")
        return np.eye(3, len(joint_indices) if joint_indices else 1)


def compute_jacobian_angular(robot_id, joint_indices, joint_angles):
    """
    Robust angular Jacobian computation with comprehensive error handling.
    
    Args:
        robot_id (int): PyBullet robot ID
        joint_indices (list): List of joint indices
        joint_angles (list): List of joint angles
        
    Returns:
        np.array: Angular Jacobian matrix (3 x n)
    """
    try:
        num_joints = len(joint_indices)
        if num_joints == 0:
            return np.zeros((3, 1))
            
        J_angular = np.zeros((3, num_joints))
        
        for i, joint_idx in enumerate(joint_indices):
            try:
                joint_info = p.getJointInfo(robot_id, joint_idx)
                joint_state = p.getLinkState(robot_id, joint_idx)
                
                joint_axis = np.array(joint_info[13], dtype=np.float64)
                joint_orientation = np.array(joint_state[5], dtype=np.float64)
                
                # Robust normalization
                axis_norm = np.linalg.norm(joint_axis)
                if axis_norm > 1e-8:
                    joint_axis = joint_axis / axis_norm
                else:
                    joint_axis = np.array([0, 0, 1], dtype=np.float64)
                
                R = quaternion_to_rotation_matrix(joint_orientation)
                result = R @ joint_axis
                
                if np.all(np.isfinite(result)):
                    J_angular[:, i] = result
                else:
                    J_angular[:, i] = [0, 0, 1]
                    
            except Exception as e:
                logging.warning(f"Error computing angular Jacobian for joint {i}: {e}")
                J_angular[:, i] = [0, 0, 1]
        
        return J_angular
        
    except Exception as e:
        logging.error(f"Jacobian angular computation failed: {e}")
        return np.eye(3, len(joint_indices) if joint_indices else 1)


def quaternion_to_rotation_matrix(q):
    """
    Robust quaternion to rotation matrix conversion.
    
    Args:
        q (np.array): Quaternion [x, y, z, w]
        
    Returns:
        np.array: 3x3 rotation matrix
    """
    try:
        q = np.array(q, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        
        if q_norm < 1e-8:
            return np.eye(3, dtype=np.float64)
            
        q = q / q_norm
        x, y, z, w = q
        
        R = np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
            [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
            [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ], dtype=np.float64)
        
        # Ensure orthogonality (numerical stability)
        if not np.allclose(np.linalg.det(R), 1.0, atol=1e-6):
            logging.warning("Non-orthogonal rotation matrix detected")
            return np.eye(3, dtype=np.float64)
            
        return R
        
    except Exception as e:
        logging.warning(f"Quaternion conversion failed: {e}")
        return np.eye(3, dtype=np.float64)


def assign_joint_weights(jacobian_linear, jacobian_angular):
    """
    Robust weight assignment with proper handling of edge cases.
    
    Args:
        jacobian_linear (np.array): Linear Jacobian matrix
        jacobian_angular (np.array): Angular Jacobian matrix
        
    Returns:
        tuple: (linear_weights, angular_weights)
    """
    try:
        if jacobian_linear.size == 0 or jacobian_angular.size == 0:
            n_joints = max(jacobian_linear.shape[1] if jacobian_linear.size > 0 else 1,
                          jacobian_angular.shape[1] if jacobian_angular.size > 0 else 1)
            uniform = np.ones(n_joints) / n_joints
            return uniform, uniform
            
        linear_weights = np.linalg.norm(jacobian_linear, axis=0)
        angular_weights = np.linalg.norm(jacobian_angular, axis=0)
        
        # Handle zero or near-zero weights
        linear_sum = np.sum(linear_weights)
        angular_sum = np.sum(angular_weights)
        
        if linear_sum < 1e-8:
            linear_weights = np.ones(len(linear_weights)) / len(linear_weights)
        else:
            linear_weights = linear_weights / linear_sum
            
        if angular_sum < 1e-8:
            angular_weights = np.ones(len(angular_weights)) / len(angular_weights)
        else:
            angular_weights = angular_weights / angular_sum
        
        # Ensure weights are finite
        linear_weights = np.where(np.isfinite(linear_weights), linear_weights, 1.0/len(linear_weights))
        angular_weights = np.where(np.isfinite(angular_weights), angular_weights, 1.0/len(angular_weights))
        
        return linear_weights, angular_weights
        
    except Exception as e:
        logging.error(f"Joint weight assignment failed: {e}")
        n = jacobian_linear.shape[1] if len(jacobian_linear.shape) > 1 else 1
        uniform = np.ones(n) / n
        return uniform, uniform


# ====== HELPER FUNCTIONS ======

def compute_weighted_joint_rewards(joint_errors, linear_weights, angular_weights, progress_reward):
    """
    Simplified weighted reward computation with proper bounds.
    
    Args:
        joint_errors (list): List of joint errors
        linear_weights (np.array): Linear movement weights
        angular_weights (np.array): Angular movement weights
        progress_reward (float): Progress-based reward
        
    Returns:
        np.array: Weighted joint rewards
    """
    try:
        combined_weights = (linear_weights + angular_weights) / 2.0
        combined_weights = combined_weights / (np.sum(combined_weights) + 1e-8)
        
        # Simple inverse error weighting
        safe_errors = np.maximum(np.array(joint_errors), 1e-6)
        inverse_errors = 1.0 / safe_errors
        inverse_errors = inverse_errors / (np.sum(inverse_errors) + 1e-8)
        
        rewards = np.clip(progress_reward * combined_weights * inverse_errors, -1.0, 2.0)
        return rewards
        
    except Exception:
        return np.zeros(len(joint_errors))


def compute_shaped_distance(distance, begin_distance):
    """
    Safe distance shaping with bounds checking.
    
    Args:
        distance (float): Current distance
        begin_distance (float): Initial distance
        
    Returns:
        tuple: (shaped_distance, shaped_begin_distance)
    """
    try:
        safe_dist = max(float(distance), 1e-8)
        safe_begin = max(float(begin_distance), 1e-8)
        return np.log(safe_dist + 1e-6), np.log(safe_begin + 1e-6)
    except Exception:
        return 0.0, 1.0


def compute_progress_reward(shaped_dist, shaped_begin, scale=1.0):
    """
    Bounded progress reward calculation with proper error handling.
    
    Args:
        shaped_dist (float): Current shaped distance
        shaped_begin (float): Initial shaped distance
        scale (float): Reward scaling factor
        
    Returns:
        float: Progress reward
    """
    try:
        progress = shaped_begin - shaped_dist
        if progress > 0:
            reward = scale * (np.exp(np.clip(progress, -5, 5)) - 1)
        else:
            reward = scale * progress * 0.3
        return np.clip(reward, -5.0, 5.0)
    except Exception:
        return 0.0


def compute_orientation_bonus(quaternion_distance):
    """
    Safe orientation bonus with proper angle handling.
    
    Args:
        quaternion_distance (float): Angular distance in [-π, π]
        
    Returns:
        float: Orientation bonus between 0 and 1
    """
    try:
        # Take absolute value since we want magnitude for bonus calculation
        safe_dist = abs(quaternion_distance)
        safe_dist = np.clip(safe_dist, 0, np.pi)
        
        # Exponential decay: closer to target = higher bonus
        bonus = np.exp(-safe_dist)
        return float(bonus)
    except Exception:
        return 0.0


def inverse_scaled_rewards(scaled_rewards, epsilon=1e-6):
    """
    Safe inverse computation with bounds checking.
    
    Args:
        scaled_rewards (list): Scaled rewards to invert
        epsilon (float): Small value to avoid division by zero
        
    Returns:
        list: Inverse of the scaled rewards
    """
    try:
        result = []
        for reward in scaled_rewards:
            if abs(reward) < epsilon:
                result.append(1.0 / epsilon)
            else:
                result.append(1.0 / reward)
        return result
    except Exception:
        return [1.0] * len(scaled_rewards)


# ====== UTILITY FUNCTIONS FOR DEBUGGING ======

def validate_reward_function_inputs(
    distance, begin_distance, prev_best, current_orientation, target_orientation,
    joint_errors, linear_weights, angular_weights, difficulties
):
    """
    Validate all inputs to the reward function for debugging purposes.
    
    Args:
        All reward function parameters
        
    Returns:
        dict: Validation results with warnings and fixes applied
    """
    validation_results = {
        'warnings': [],
        'fixes_applied': [],
        'is_valid': True
    }
    
    try:
        # Check distance values
        if not np.isfinite(distance) or distance < 0:
            validation_results['warnings'].append(f"Invalid distance: {distance}")
            validation_results['is_valid'] = False
            
        if not np.isfinite(begin_distance) or begin_distance <= 0:
            validation_results['warnings'].append(f"Invalid begin_distance: {begin_distance}")
            validation_results['is_valid'] = False
            
        # Check quaternions
        current_quat = np.array(current_orientation)
        target_quat = np.array(target_orientation)
        
        if len(current_quat) != 4 or not np.all(np.isfinite(current_quat)):
            validation_results['warnings'].append("Invalid current_orientation quaternion")
            validation_results['is_valid'] = False
            
        if len(target_quat) != 4 or not np.all(np.isfinite(target_quat)):
            validation_results['warnings'].append("Invalid target_orientation quaternion")
            validation_results['is_valid'] = False
            
        # Check joint errors
        if not joint_errors or len(joint_errors) == 0:
            validation_results['warnings'].append("Empty joint_errors list")
            validation_results['is_valid'] = False
        else:
            for i, error in enumerate(joint_errors):
                if not np.isfinite(error):
                    validation_results['warnings'].append(f"Non-finite joint error at index {i}: {error}")
                    validation_results['is_valid'] = False
                    
        # Check weights
        if len(linear_weights) != len(joint_errors):
            validation_results['warnings'].append(
                f"Linear weights length ({len(linear_weights)}) != joint errors length ({len(joint_errors)})"
            )
            
        if len(angular_weights) != len(joint_errors):
            validation_results['warnings'].append(
                f"Angular weights length ({len(angular_weights)}) != joint errors length ({len(joint_errors)})"
            )
            
        # Check difficulties
        if len(difficulties) != len(joint_errors):
            validation_results['warnings'].append(
                f"Difficulties length ({len(difficulties)}) != joint errors length ({len(joint_errors)})"
            )
            
    except Exception as e:
        validation_results['warnings'].append(f"Validation error: {str(e)}")
        validation_results['is_valid'] = False
        
    return validation_results


def log_reward_statistics(rewards, episode_number, log_frequency=100):
    """
    Log reward statistics for monitoring training stability.
    
    Args:
        rewards (list or np.array): Reward values
        episode_number (int): Current episode number
        log_frequency (int): How often to log statistics
    """
    if episode_number % log_frequency == 0:
        try:
            rewards_array = np.array(rewards)
            
            stats = {
                'episode': episode_number,
                'mean': np.mean(rewards_array),
                'std': np.std(rewards_array),
                'min': np.min(rewards_array),
                'max': np.max(rewards_array),
                'median': np.median(rewards_array),
                'finite_count': np.sum(np.isfinite(rewards_array)),
                'total_count': len(rewards_array)
            }
            
            logging.info(f"Episode {episode_number} Reward Stats:")
            logging.info(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            logging.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            logging.info(f"  Median: {stats['median']:.4f}")
            logging.info(f"  Finite values: {stats['finite_count']}/{stats['total_count']}")
            
            # Warning checks
            if stats['std'] > 3.0:
                logging.warning("⚠️  High reward variance detected!")
                
            if stats['finite_count'] < stats['total_count']:
                logging.error("🚨 Non-finite rewards detected!")
                
            if abs(stats['mean']) > 5.0:
                logging.warning("⚠️  Extreme reward mean detected!")
                
        except Exception as e:
            logging.error(f"Error in reward statistics logging: {e}")


def test_reward_function():
    """
    Test the reward function with various edge cases to ensure robustness.
    """
    print("Testing reward function robustness...")
    
    # Test case 1: Normal inputs
    try:
        rewards, _, _, _ = compute_reward(
            distance=0.5,
            begin_distance=1.0,
            prev_best=0.6,
            current_orientation=[0, 0, 0, 1],
            target_orientation=[0, 0, 0.1, 0.995],
            joint_errors=[0.1, 0.2, 0.15],
            linear_weights=[0.3, 0.4, 0.3],
            angular_weights=[0.2, 0.5, 0.3],
            difficulties=[1.0, 1.5, 1.2],
            episode_number=100,
            total_episodes=1000
        )
        print("✅ Normal inputs test passed")
        print(f"   Rewards: {rewards}")
        
    except Exception as e:
        print(f"❌ Normal inputs test failed: {e}")
    
    # Test case 2: Edge case inputs
    try:
        rewards, _, _, _ = compute_reward(
            distance=0.0,  # Zero distance
            begin_distance=1e-10,  # Very small begin distance
            prev_best=float('inf'),  # Infinite prev_best
            current_orientation=[0, 0, 0, 0],  # Zero quaternion
            target_orientation=[1, 1, 1, 1],  # Unnormalized quaternion
            joint_errors=[float('inf'), float('-inf'), float('nan')],  # Bad joint errors
            linear_weights=[],  # Empty weights
            angular_weights=[0, 0, 0],  # Zero weights
            difficulties=[-1, 100],  # Bad difficulties
            episode_number=0,
            total_episodes=0
        )
        print("✅ Edge case inputs test passed")
        print(f"   Rewards: {rewards}")
        
    except Exception as e:
        print(f"❌ Edge case inputs test failed: {e}")
    
    # Test case 3: Large angle wrapping
    try:
        # Test large angles that need wrapping
        large_angle = 10 * np.pi  # 10π radians
        wrapped = wrap_angle_to_pi(large_angle)
        print(f"✅ Angle wrapping test: {large_angle:.3f} → {wrapped:.3f}")
        
        # Test quaternion distance with equivalent rotations
        q1 = [0, 0, 0, 1]    # No rotation
        q2 = [0, 0, 0, -1]   # Same rotation (quaternion double cover)
        dist = compute_quaternion_distance(q1, q2)
        print(f"✅ Quaternion double cover test: distance = {dist:.6f}")
        
    except Exception as e:
        print(f"❌ Angle wrapping test failed: {e}")
    
    print("Reward function testing completed!")


# ====== PERFORMANCE MONITORING ======

class RewardFunctionMonitor:
    """
    Monitor for tracking reward function performance and detecting issues.
    """
    
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.reward_history = deque(maxlen=window_size)
        self.computation_times = deque(maxlen=window_size)
        self.error_count = 0
        self.warning_count = 0
        
    def log_reward_call(self, rewards, computation_time, had_error=False, had_warning=False):
        """Log a reward function call for monitoring."""
        self.reward_history.extend(rewards if isinstance(rewards, list) else [rewards])
        self.computation_times.append(computation_time)
        
        if had_error:
            self.error_count += 1
        if had_warning:
            self.warning_count += 1
            
    def get_statistics(self):
        """Get monitoring statistics."""
        if not self.reward_history:
            return {}
            
        rewards_array = np.array(self.reward_history)
        
        return {
            'reward_stats': {
                'mean': np.mean(rewards_array),
                'std': np.std(rewards_array),
                'min': np.min(rewards_array),
                'max': np.max(rewards_array),
                'finite_ratio': np.sum(np.isfinite(rewards_array)) / len(rewards_array)
            },
            'performance_stats': {
                'avg_computation_time': np.mean(self.computation_times),
                'max_computation_time': np.max(self.computation_times),
                'error_rate': self.error_count / len(self.computation_times),
                'warning_rate': self.warning_count / len(self.computation_times)
            }
        }
        
    def check_health(self):
        """Check if the reward function is healthy."""
        stats = self.get_statistics()
        
        if not stats:
            return True, []
            
        issues = []
        
        # Check reward health
        if stats['reward_stats']['finite_ratio'] < 0.95:
            issues.append("High rate of non-finite rewards")
            
        if stats['reward_stats']['std'] > 5.0:
            issues.append("Very high reward variance")
            
        # Check performance health
        if stats['performance_stats']['error_rate'] > 0.01:
            issues.append("High error rate in reward computation")
            
        if stats['performance_stats']['avg_computation_time'] > 0.01:
            issues.append("Slow reward computation")
            
        return len(issues) == 0, issues


# ====== MODULE INITIALIZATION ======

# Global monitor instance
_reward_monitor = RewardFunctionMonitor()

def get_reward_monitor():
    """Get the global reward function monitor."""
    return _reward_monitor


# Run self-test if module is executed directly
if __name__ == "__main__":
    test_reward_function()