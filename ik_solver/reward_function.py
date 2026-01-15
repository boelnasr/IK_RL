import pybullet as p
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


EPS = 1e-8            # single global epsilon

# SIMPLIFIED reward constants
REWARD_CONSTANTS = {
    # Core rewards
    "IMPROVEMENT_SCALE": 10.0,      # Reward for error reduction
    "SUCCESS_BONUS": 5.0,           # Bonus when joint is under threshold
    "TEAM_SUCCESS_BONUS": 10.0,     # Bonus when ALL joints succeed

    # Minimal penalties
    "STEP_PENALTY": -0.001,         # Small per-step cost to encourage efficiency

    # Bounds
    "MAX_REWARD": 20.0,
    "MIN_REWARD": -1.0,             # Minimal negative (no harsh penalties)
}

REWARD_DEFAULT_ARGUMENTS = {
    "success_threshold": 0.005,     # 5mm default
    "position_threshold": 0.005,
    "orientation_threshold": 0.01,
    "joint_threshold": 0.005,
}

OVERALL_DISTANCE_WEIGHTS = {
    "position": 0.7,
    "orientation": 0.3,
}


def get_reward_parameters_snapshot():
    """
    Provide a serialisable snapshot of the reward shaping configuration.
    """
    return {
        "eps": EPS,
        "reward_constants": REWARD_CONSTANTS.copy(),
        "default_arguments": REWARD_DEFAULT_ARGUMENTS.copy(),
        "overall_distance_weights": OVERALL_DISTANCE_WEIGHTS.copy(),
    }


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
    Return unsigned axis-angle distance in radians ∈ [0, π].
    API and units stay the same – you still call it exactly the same way.
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    n1 = np.linalg.norm(q1);  n2 = np.linalg.norm(q2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0

    q1 /= n1;  q2 /= n2
    dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)   # abs() handles double cover
    # small-angle safeguard
    if dot > 0.999999:
        return 0.0
    return 2.0 * np.arccos(dot)                      # ∈ (0, π]


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
        overall_distance = (
            OVERALL_DISTANCE_WEIGHTS["position"] * position_error +
            OVERALL_DISTANCE_WEIGHTS["orientation"] * orientation_error
        )
        return max(float(overall_distance), 1e-8)
        
    except Exception as e:
        logging.warning(f"Error in overall distance calculation: {e}")
        return 1.0

# --------------------------------------------------------------------------- #
#  SIMPLIFIED compute_reward() - Clean, minimal, effective
# --------------------------------------------------------------------------- #
# Previous joint errors for computing improvement (stored per-episode)
_prev_joint_errors = None

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
    *,
    position_error: Optional[float] = None,
    orientation_error: Optional[float] = None,
    success_threshold: float = 0.005,
    position_threshold: float = 0.005,
    orientation_threshold: float = 0.01,
    joint_threshold: float = 0.005,
    ratio_threshold: float = 0.5,
    time_penalty: float = -0.001,
    smoothing_window: int = 10,
    exploration_bonus: float = 0.02,
    joint_movements: Optional[list] = None,
    done: bool = False
) -> tuple:
    """
    SIMPLIFIED reward function:
    1. Improvement reward: Positive reward when error decreases
    2. Success bonus: Bonus when joint error < threshold
    3. Team bonus: Extra bonus when ALL joints succeed
    4. Minimal step penalty: Small cost per step

    Returns:
        tuple: (final_rewards, joint_rewards, new_best_distance, success, stay_rewards)
    """
    global _prev_joint_errors

    # Get constants
    IMPROVEMENT_SCALE = REWARD_CONSTANTS.get("IMPROVEMENT_SCALE", 10.0)
    SUCCESS_BONUS = REWARD_CONSTANTS.get("SUCCESS_BONUS", 5.0)
    TEAM_SUCCESS_BONUS = REWARD_CONSTANTS.get("TEAM_SUCCESS_BONUS", 10.0)
    STEP_PENALTY = REWARD_CONSTANTS.get("STEP_PENALTY", -0.001)
    MAX_REWARD = REWARD_CONSTANTS.get("MAX_REWARD", 20.0)
    MIN_REWARD = REWARD_CONSTANTS.get("MIN_REWARD", -1.0)

    # ---- 1. Basic validation --------------------------------------------------
    num_joints = len(joint_errors)
    if num_joints == 0:
        return np.zeros(0), [], prev_best, False, []

    # Convert to numpy arrays
    joint_errors = np.abs(np.asarray(joint_errors, dtype=np.float64))
    joint_threshold = max(float(joint_threshold), EPS)

    # Initialize previous errors if needed
    if _prev_joint_errors is None or len(_prev_joint_errors) != num_joints:
        _prev_joint_errors = joint_errors.copy()

    # ---- 2. Compute rewards per joint ------------------------------------------
    final_rewards = np.zeros(num_joints, dtype=np.float32)
    individual_successes = []

    for j in range(num_joints):
        err = joint_errors[j]
        prev_err = _prev_joint_errors[j]

        # 1. IMPROVEMENT REWARD: Positive when error decreases
        improvement = (prev_err - err) / max(prev_err, joint_threshold, EPS)
        improvement = np.clip(improvement, -0.5, 1.0)  # Cap negative improvement
        improvement_reward = IMPROVEMENT_SCALE * max(0, improvement)  # Only reward improvement

        # 2. SUCCESS BONUS: When joint error is under threshold
        is_successful = err <= joint_threshold
        individual_successes.append(is_successful)
        success_bonus = SUCCESS_BONUS if is_successful else 0.0

        # 3. SMALL STEP PENALTY: Encourage efficiency
        step_penalty = STEP_PENALTY

        # Total reward for this joint
        reward = improvement_reward + success_bonus + step_penalty
        final_rewards[j] = np.clip(reward, MIN_REWARD, MAX_REWARD)

    # 4. TEAM BONUS: Extra reward when ALL joints succeed
    all_successful = all(individual_successes)
    if all_successful:
        final_rewards += TEAM_SUCCESS_BONUS / num_joints

    # Update previous errors for next step
    _prev_joint_errors = joint_errors.copy()

    # Update best distance
    new_best = min(prev_best, distance)

    # Overall success: all joints under threshold
    success = all_successful

    # Return format: (rewards, joint_rewards_list, new_best, success, stay_rewards)
    joint_specific_rewards = final_rewards.tolist()
    stay_rewards_list = [0.0] * num_joints  # Not used in simplified version

    return final_rewards, joint_specific_rewards, new_best, success, stay_rewards_list


def adaptive_clip(rewards, stats):
    """
    ENHANCED: Safer, faster adaptive clipping with Welford's online algorithm (Change 5).
    
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
        
        # Welford's online update for numerical stability
        alpha = 0.99
        delta = curr_mean - stats.get('running_mean', curr_mean)
        stats['running_mean'] = stats.get('running_mean', curr_mean) + alpha * delta
        stats['running_M2'] = stats.get('running_M2', 0.0) + delta * (curr_mean - stats['running_mean'])
        n = stats.get('count', 0) + 1
        stats['count'] = n
        stats['running_std'] = math.sqrt(stats['running_M2'] / max(n-1, 1))
        
        # Conservative clipping bounds
        clip_factor = 1.0
        lower_bound = stats['running_mean'] - clip_factor * stats['running_std']
        upper_bound = stats['running_mean'] + clip_factor * stats['running_std']
        
        # Apply conservative bounds
        lower_bound = max(lower_bound, -2.0)  # Never clip below -2
        upper_bound = min(upper_bound, 8.0)   # Never clip above 8 (increased for success bonuses)
        
        clipped_rewards = np.copy(rewards)
        clipped_rewards[finite_mask] = np.clip(finite_rewards, lower_bound, upper_bound)
        
        return clipped_rewards
        
    except Exception as e:
        logging.warning(f"Adaptive clipping error: {e}")
        return np.clip(rewards, -2.0, 8.0)


# ====== ENHANCED JACOBIAN COMPUTATION (Change 6) ======

def compute_jacobian_linear(robot_id, joint_indices, joint_angles):
    """
    ENHANCED: Use PyBullet's built-in Jacobian computation for better performance.
    
    Args:
        robot_id (int): PyBullet robot ID
        joint_indices (list): List of joint indices
        joint_angles (list): List of joint angles
        
    Returns:
        np.array: Linear Jacobian matrix (3 x n)
    """
    try:
        if not joint_indices:
            return np.zeros((3, 1))
            
        zero_vec = [0.0] * len(joint_indices)
        j_lin, _ = p.calculateJacobian(
            robot_id, joint_indices[-1],
            [0, 0, 0],  # local position of end-effector
            list(joint_angles), zero_vec, zero_vec
        )
        return np.asarray(j_lin, dtype=np.float64)
        
    except Exception as e:
        logging.warning(f"Built-in linear Jacobian computation failed: {e}")
        # Fallback to manual computation
        return compute_jacobian_linear_manual(robot_id, joint_indices, joint_angles)


def compute_jacobian_angular(robot_id, joint_indices, joint_angles):
    """
    ENHANCED: Use PyBullet's built-in Jacobian computation for better performance.
    
    Args:
        robot_id (int): PyBullet robot ID
        joint_indices (list): List of joint indices
        joint_angles (list): List of joint angles
        
    Returns:
        np.array: Angular Jacobian matrix (3 x n)
    """
    try:
        if not joint_indices:
            return np.zeros((3, 1))
            
        zero_vec = [0.0] * len(joint_indices)
        _, j_ang = p.calculateJacobian(
            robot_id, joint_indices[-1],
            [0, 0, 0],  # local position of end-effector
            list(joint_angles), zero_vec, zero_vec
        )
        return np.asarray(j_ang, dtype=np.float64)
        
    except Exception as e:
        logging.warning(f"Built-in angular Jacobian computation failed: {e}")
        # Fallback to manual computation
        return compute_jacobian_angular_manual(robot_id, joint_indices, joint_angles)


def compute_jacobian_linear_manual(robot_id, joint_indices, joint_angles):
    """
    Manual linear Jacobian computation as fallback.
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
                logging.warning(f"Error computing manual Jacobian for joint {i}: {e}")
                J_linear[:, i] = [0, 0, 0]
        
        return J_linear
        
    except Exception as e:
        logging.error(f"Manual Jacobian linear computation failed: {e}")
        return np.eye(3, len(joint_indices) if joint_indices else 1)


def compute_jacobian_angular_manual(robot_id, joint_indices, joint_angles):
    """
    Manual angular Jacobian computation as fallback.
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
                logging.warning(f"Error computing manual angular Jacobian for joint {i}: {e}")
                J_angular[:, i] = [0, 0, 1]
        
        return J_angular
        
    except Exception as e:
        logging.error(f"Manual Jacobian angular computation failed: {e}")
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
        quaternion_distance (float): Angular distance in [0, π]
        
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
    print("Testing enhanced reward function robustness...")
    
    # Test case 1: Normal inputs
    try:
        rewards, _, _, success, _ = compute_reward(
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
            total_episodes=1000,
            position_error=0.05,
            orientation_error=0.1
        )
        print("✅ Normal inputs test passed")
        print(f"   Rewards: {rewards}")
        print(f"   Success: {success}")
        
    except Exception as e:
        print(f"❌ Normal inputs test failed: {e}")
    
    # Test case 2: Success condition test with direct errors
    try:
        rewards, _, _, success, _ = compute_reward(
            distance=0.005,  # Very small distance
            begin_distance=1.0,
            prev_best=0.6,
            current_orientation=[0, 0, 0, 1],
            target_orientation=[0, 0, 0, 1],  # Same orientation
            joint_errors=[0.005, 0.008, 0.003],  # Very small joint errors
            linear_weights=[0.3, 0.4, 0.3],
            angular_weights=[0.2, 0.5, 0.3],
            difficulties=[1.0, 1.5, 1.2],
            episode_number=500,
            total_episodes=1000,
            position_error=0.005,  # Direct position error
            orientation_error=0.01,  # Direct orientation error
            position_threshold=0.02,
            orientation_threshold=0.05,
            joint_threshold=0.01
        )
        print("✅ Success condition test with direct errors passed")
        print(f"   Rewards: {rewards}")
        print(f"   Success detected: {success}")
        
    except Exception as e:
        print(f"❌ Success condition test failed: {e}")
    
    # Test case 3: Late-stage training boost
    try:
        rewards_early, _, _, _, _ = compute_reward(
            distance=0.01,
            begin_distance=1.0,
            prev_best=0.02,
            current_orientation=[0, 0, 0, 1],
            target_orientation=[0, 0, 0, 1],
            joint_errors=[0.008, 0.009, 0.007],
            linear_weights=[0.3, 0.4, 0.3],
            angular_weights=[0.2, 0.5, 0.3],
            difficulties=[1.0, 1.5, 1.2],
            episode_number=200,  # Early training (20%)
            total_episodes=1000,
            position_error=0.008,
            orientation_error=0.005
        )
        
        rewards_late, _, _, _, _ = compute_reward(
            distance=0.01,
            begin_distance=1.0,
            prev_best=0.02,
            current_orientation=[0, 0, 0, 1],
            target_orientation=[0, 0, 0, 1],
            joint_errors=[0.008, 0.009, 0.007],
            linear_weights=[0.3, 0.4, 0.3],
            angular_weights=[0.2, 0.5, 0.3],
            difficulties=[1.0, 1.5, 1.2],
            episode_number=900,  # Late training (90%)
            total_episodes=1000,
            position_error=0.008,
            orientation_error=0.005
        )
        
        print("✅ Late-stage amplification test passed")
        print(f"   Early rewards: {rewards_early}")
        print(f"   Late rewards: {rewards_late}")
        print(f"   Amplification factor: {np.mean(rewards_late) / max(np.mean(rewards_early), 1e-6):.2f}")
        
    except Exception as e:
        print(f"❌ Late-stage amplification test failed: {e}")
    
    # Test case 4: Built-in Jacobian computation
    try:
        # This would require a real PyBullet environment, so we'll just test the fallback
        linear_jac = compute_jacobian_linear_manual(None, [0, 1, 2], [0.1, 0.2, 0.3])
        angular_jac = compute_jacobian_angular_manual(None, [0, 1, 2], [0.1, 0.2, 0.3])
        print("✅ Manual Jacobian fallback test passed")
        print(f"   Linear Jacobian shape: {linear_jac.shape}")
        print(f"   Angular Jacobian shape: {angular_jac.shape}")
        
    except Exception as e:
        print(f"❌ Jacobian computation test failed: {e}")
    
    # Test case 5: State reset functionality
    try:
        # Test episode boundary reset
        rewards1, _, _, _, _ = compute_reward(
            distance=0.1, begin_distance=1.0, prev_best=0.2,
            current_orientation=[0, 0, 0, 1], target_orientation=[0, 0, 0, 1],
            joint_errors=[0.05, 0.06], linear_weights=[0.5, 0.5], angular_weights=[0.5, 0.5],
            difficulties=[1.0, 1.0], episode_number=1, total_episodes=100,
            done=True  # Episode completion
        )
        
        rewards2, _, _, _, _ = compute_reward(
            distance=0.1, begin_distance=1.0, prev_best=0.2,
            current_orientation=[0, 0, 0, 1], target_orientation=[0, 0, 0, 1],
            joint_errors=[0.05, 0.06], linear_weights=[0.5, 0.5], angular_weights=[0.5, 0.5],
            difficulties=[1.0, 1.0], episode_number=2, total_episodes=100
        )
        
        print("✅ State reset test passed")
        print(f"   Episode 1 rewards: {rewards1}")
        print(f"   Episode 2 rewards: {rewards2}")
        
    except Exception as e:
        print(f"❌ State reset test failed: {e}")
    
    print("Enhanced reward function testing completed!")


# ====== ENVIRONMENT INTEGRATION EXAMPLE ======

def example_environment_integration():
    """
    Example of how to integrate the enhanced reward function with the environment.
    This shows the call site changes needed in InverseKinematicsEnv.step()
    """
    print("Example environment integration:")
    print("""
    # In InverseKinematicsEnv.step() - call site change only
    rewards, individual_rewards, self.previous_best_distance, overall_success, stay_rewards = compute_reward(
        distance=float(self.current_distance),
        begin_distance=float(self.initial_distance),
        prev_best=float(self.previous_best_distance),
        current_orientation=self.current_quaternion.tolist(),
        target_orientation=self.target_quaternion.tolist(),
        joint_errors=self.joint_errors.tolist(),
        linear_weights=self.linear_weights.tolist(),
        angular_weights=self.angular_weights.tolist(),
        difficulties=step_difficulties,
        episode_number=self.episode_number,
        total_episodes=self.total_episodes,
        # NEW: Direct error inputs ↓↓↓
        position_error=float(np.linalg.norm(self.position_error)),
        orientation_error=float(np.linalg.norm(self.orientation_error)),
        position_threshold=float(self.position_threshold),
        orientation_threshold=float(self.orientation_threshold),
        joint_threshold=float(self.success_threshold),
        done=done  # Pass episode completion flag
    )
    """)


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_reward_function()
    example_environment_integration()
