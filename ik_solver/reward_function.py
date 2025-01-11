import pybullet as p
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
import math
def compute_position_error(current_position, target_position):
    """
    Enhanced position error calculation with debugging.
    """
    error = np.linalg.norm(current_position - target_position)
    # print(f"Debug - Position Error: {error:.6f}")
    # print(f"Debug - Current Position: {current_position}")
    # print(f"Debug - Target Position: {target_position}")
    return max(error, 1e-6)  # Ensure non-zero error


def compute_quaternion_distance(q1, q2):
    """
    Compute the distance between two quaternions.

    Args:
        q1 (np.array): First quaternion.
        q2 (np.array): Second quaternion.

    Returns:
        float: Quaternion distance.
    """
    q1 = np.array(q1)
    q2 = np.array(q2)
    dot_product = np.abs(np.dot(q1, q2))
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = 2 * np.arccos(dot_product)
    return angle




def compute_overall_distance(current_position, target_position, current_orientation, target_orientation):
    """
    Computes the overall distance combining Euclidean distance for position and quaternion distance for orientation.

    Args:
        current_position (np.array): Current position of the end-effector.
        target_position (np.array): Target position of the end-effector.
        current_orientation (np.array): Current orientation quaternion of the end-effector.
        target_orientation (np.array): Target orientation quaternion of the end-effector.

    Returns:
        float: The overall distance metric.
    """
    # Euclidean distance between positions
    euclidean_distance = compute_position_error(current_position, target_position)

    # Quaternion distance between orientations
    quaternion_distance = compute_quaternion_distance(current_orientation, target_orientation)

    # Combine the Euclidean and Quaternion distances
    overall_distance = euclidean_distance + quaternion_distance
    return overall_distance

# def compute_reward(distance, begin_distance, prev_best, current_orientation, target_orientation, 
#                    joint_errors, linear_weights, angular_weights,
#                    success_threshold=0.006, time_penalty=-1.0):
#     """
#     Enhanced reward function that incorporates joint errors, progress reward, orientation bonus, 
#     and penalties to encourage efficient movement and precision.
    
#     Args:
#         distance (float): Current distance to target.
#         begin_distance (float): Initial distance to target.
#         prev_best (float): Best distance achieved so far.
#         current_orientation (np.array): Current end-effector orientation quaternion.
#         target_orientation (np.array): Target orientation quaternion.
#         joint_errors (list): List of errors for each joint.
#         linear_weights (np.array): Weights for each joint based on linear movement.
#         angular_weights (np.array): Weights for each joint based on angular movement.
#         success_threshold (float): Distance threshold for success.
#         time_penalty (float): Small penalty per timestep to encourage efficiency.
    
#     Returns:
#         tuple: (reward, prev_best, success)
#     """
#     success = False
    
#     # Shape the distance metrics
#     shaped_dist, shaped_begin = compute_shaped_distance(distance, begin_distance)
    
#     # Compute progress reward
#     progress_reward = compute_progress_reward(shaped_dist, shaped_begin)
    
#     # Compute orientation reward
#     quaternion_distance = compute_quaternion_distance(current_orientation, target_orientation)
#     orientation_reward = compute_orientation_bonus(quaternion_distance)
    
#     # Calculate joint rewards using joint errors and weights
#     joint_rewards = compute_weighted_joint_rewards(joint_errors, linear_weights, angular_weights, progress_reward)
#     joint_reward_contribution = np.sum(joint_rewards)  # Aggregate contribution from joint rewards

#     # Combine rewards with time penalty and joint contributions
#     reward = progress_reward + orientation_reward + joint_reward_contribution + time_penalty
    
#     # Update the previous best distance if improved
#     if distance < prev_best:
#         prev_best = distance

#     # Check for success and add a success bonus
#     if distance <= success_threshold:
#         reward += 2000.0 + (success_threshold - distance) * 1000
#         success = True
    
#     # Apply penalty if moving away from the target
#     if distance > prev_best:
#         reward = prev_best - distance  # Negative value
    
#     # Subtract joint error penalty
#     joint_error_array = np.array(joint_errors)
#     total_joint_weight = linear_weights + angular_weights
#     total_joint_weight /= np.sum(total_joint_weight) + 1e-8

#     weighted_joint_errors = joint_error_array * total_joint_weight
#     joint_error_penalty = np.sum(weighted_joint_errors)
#     reward -= joint_error_penalty
    
#     # Clip the reward to ensure numerical stability
#     reward = np.clip(reward, -1, 100)
    
#     return reward, prev_best, success

# def compute_reward(distance, begin_distance, prev_best, current_orientation, target_orientation, 
#                    joint_errors, linear_weights, angular_weights,
#                    success_threshold=0.006, time_penalty=-1):
#     """
#     Enhanced reward function that incorporates joint errors and returns individual joint rewards.
    
#     Args:
#         distance (float): Current distance to target.
#         begin_distance (float): Initial distance to target.
#         prev_best (float): Best distance achieved so far.
#         current_orientation (np.array): Current end-effector orientation quaternion.
#         target_orientation (np.array): Target orientation quaternion.
#         joint_errors (list): List of errors for each joint.
#         linear_weights (np.array): Weights for each joint based on linear movement.
#         angular_weights (np.array): Weights for each joint based on angular movement.
#         success_threshold (float): Distance threshold for success.
#         time_penalty (float): Small penalty per timestep to encourage efficiency.
    
#     Returns:
#         tuple: (reward, individual_rewards, prev_best, success)
#     """
#     success = False
    
#     # Shape the distance metrics
#     shaped_dist, shaped_begin = compute_shaped_distance(distance, begin_distance)
    
#     # Compute progress reward
#     progress_reward = compute_progress_reward(shaped_dist, shaped_begin)
    
#     # Compute orientation reward
#     quaternion_distance = compute_quaternion_distance(current_orientation, target_orientation)
#     orientation_reward = compute_orientation_bonus(quaternion_distance)
    
#     # Calculate joint rewards using joint errors and weights
#     joint_rewards = compute_weighted_joint_rewards(
#         joint_errors, linear_weights, angular_weights, progress_reward
#     )
#     joint_reward_contribution = np.sum(joint_rewards)  # Aggregate contribution from joint rewards

#     # Combine rewards with time penalty and joint contributions
#     reward = (
#         0.1 * progress_reward +
#         0.1 * orientation_reward +
#         0.1 * joint_reward_contribution +
#         time_penalty
#     )
    
#     # Check for success
#     if distance <= success_threshold:
#         # Exponential bonus for precision
#         precision_bonus = 2000 * np.exp(-distance / (0.1 * success_threshold))
#         reward += precision_bonus
#         success = True

#     # Update previous best if we've improved
#     if distance < prev_best:
#         prev_best = distance
#     elif distance > prev_best:
#         # The agent moved away from the target, apply punishment
#         reward -= (distance - prev_best) * 10  # Adjust the multiplier as needed

#     # Add smoothing to prevent sharp reward changes
#     reward = np.clip(reward, -1, 5000)
    
#     # Return individual joint rewards along with other values
#     return reward, joint_rewards, prev_best, success

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
    time_penalty: float = -0.1,
    smoothing_window: int = 10,
    exploration_bonus: float = 0.1
) -> tuple:
    """
    Computes the reward for inverse kinematics tasks with unique difficulties for each agent.
    Args:
        difficulties (list): List of difficulty levels for each agent.
    Returns:
        tuple: (final_rewards, joint_specific_rewards, new_best, success)
    """
    # Constants
    MAX_REWARD = 10.0
    MIN_REWARD = -10.0
    POSITION_SCALE = 0.6
    ORIENTATION_SCALE = 0.4
    IMPROVEMENT_SCALE = 0.3
    ERROR_PENALTY_SCALE = 0.3
    NOVELTY_BASE = 0.5

    num_joints = len(joint_errors)

    # Initialize stats tracking
    if not hasattr(compute_reward, 'reward_stats'):
        compute_reward.reward_stats = {
            'running_mean': 0.0,
            'running_std': 1.0,
            'history': deque(maxlen=100),
            'alpha': 0.95  # Exponential moving average factor
        }

    if not hasattr(compute_reward, 'distance_history'):
        compute_reward.distance_history = deque(maxlen=smoothing_window)
        compute_reward.min_distance = float('inf')
        compute_reward.joint_histories = [deque(maxlen=smoothing_window) for _ in range(num_joints)]
        compute_reward.prev_joint_errors = [float('inf')] * num_joints

    # Update minimum distance and history
    compute_reward.min_distance = min(compute_reward.min_distance, distance)
    compute_reward.distance_history.append(distance)

    # Normalize metrics
    norm_distance = distance / (begin_distance + 1e-8)
    orientation_diff = compute_quaternion_distance(current_orientation, target_orientation)
    norm_quaternion_distance = orientation_diff / np.pi

    joint_specific_rewards = []
    final_rewards = np.zeros(num_joints)

    # Reward scaling factor based on episode progression
    episode_progress = max(0.1, episode_number / total_episodes)

    # Calculate per-agent rewards
    for i in range(num_joints):
        # Retrieve agent-specific difficulty
        agent_difficulty = difficulties[i]

        # Update joint error history and improvement calculation
        compute_reward.joint_histories[i].append(joint_errors[i])
        smoothed_error = max(np.median(list(compute_reward.joint_histories[i])), 1e-6)
        prev_error = max(compute_reward.prev_joint_errors[i], 1e-6)
        improvement = 0.0
        if math.isfinite(prev_error) and math.isfinite(smoothed_error) and episode_number >= 0.1 * total_episodes:
            improvement = np.clip((prev_error - smoothed_error) / (prev_error + 1e-8), -1.0, 1.0)
        compute_reward.prev_joint_errors[i] = smoothed_error

        # Scale factors by agent difficulty
        SUCCESS_BONUS_BASE = 5 * agent_difficulty
        task_difficulty = 0.2 + 0.3 * agent_difficulty

        # Position and orientation rewards
        position_reward = POSITION_SCALE * (1.0 - norm_distance) * linear_weights[i] * agent_difficulty
        orientation_reward = ORIENTATION_SCALE * (1.0 - norm_quaternion_distance) * angular_weights[i] * agent_difficulty

        # Improvement reward
        progress_reward = IMPROVEMENT_SCALE * np.tanh(improvement) * episode_progress

        # Error penalty
        normalized_error = np.clip(joint_errors[i] / np.pi, 0.0, 1.0)
        combined_weight = 0.5 * (linear_weights[i] + angular_weights[i])
        error_penalty = -normalized_error * combined_weight * task_difficulty * ERROR_PENALTY_SCALE

        # Novelty bonus
        state_key = tuple(np.round([distance, smoothed_error], 3))
        novelty_bonus = 0
        if episode_number >= 0.1 * total_episodes:
            novelty_bonus = exploration_bonus * (NOVELTY_BASE / np.sqrt(
                compute_reward.reward_stats['history'].count(state_key) + 1
            )) * episode_progress

        # Success bonus
        success_bonus = 0
        if joint_errors[i] <= success_threshold and episode_number > 1:
            success_bonus = SUCCESS_BONUS_BASE * (1.0 + agent_difficulty) * episode_progress

        # Combine components
        joint_reward = (
            position_reward +
            orientation_reward +
            progress_reward +
            error_penalty +
            novelty_bonus +
            success_bonus +
            time_penalty * task_difficulty
        )
        joint_reward = np.clip(joint_reward, MIN_REWARD, MAX_REWARD)
        joint_specific_rewards.append(joint_reward)
        final_rewards[i] = joint_reward

    # Adaptive clipping
    final_rewards = adaptive_clip(final_rewards, compute_reward.reward_stats)

    # Determine success
    successes = np.array(joint_errors) <= success_threshold
    success = np.all(successes)
    new_best = min(prev_best, compute_reward.min_distance)
    
    return final_rewards, joint_specific_rewards, new_best, success
 

def adaptive_clip(rewards, stats):
    """
    Clip rewards adaptively based on running mean and standard deviation.

    Args:
        rewards (np.array): Rewards to clip.
        stats (dict): Dictionary containing running mean and std.

    Returns:
        np.array: Clipped rewards.
    """
    curr_mean = np.mean(rewards)
    curr_std = np.std(rewards) + 1e-8

    # Update running mean and std
    stats['running_mean'] = stats['alpha'] * stats['running_mean'] + (1 - stats['alpha']) * curr_mean
    stats['running_std'] = stats['alpha'] * stats['running_std'] + (1 - stats['alpha']) * curr_std

    # Clip rewards adaptively
    return np.clip(rewards, stats['running_mean'] - 0.2 * stats['running_std'], stats['running_mean'] + 0.2 * stats['running_std'])

def compute_jacobian_linear(robot_id, joint_indices, joint_angles):
    """
    Computes the linear Jacobian for the end-effector analytically.
    
    Args:
        robot_id (int): The ID of the robot in the PyBullet simulation.
        joint_indices (list): The indices of the joints to consider.
        joint_angles (list): The angles of the joints to compute the Jacobian.
    
    Returns:
        np.array: Linear Jacobian matrix (3 x n).
    """
    num_joints = len(joint_indices)
    J_linear = np.zeros((3, num_joints))
    
    # Get end-effector position and joint poses
    ee_state = p.getLinkState(robot_id, joint_indices[-1])
    ee_pos = np.array(ee_state[4])  # Use worldLinkFramePosition
    
    # Compute Jacobian column by column
    for i, joint_idx in enumerate(joint_indices):
        # Get joint state
        joint_info = p.getJointInfo(robot_id, joint_idx)
        joint_state = p.getLinkState(robot_id, joint_idx)
        
        # Get joint position and axis in world frame
        joint_pos = np.array(joint_state[4])  # Use worldLinkFramePosition
        joint_axis = np.array(joint_info[13])  # Joint axis in world frame
        
        # Normalize joint axis
        joint_axis = joint_axis / (np.linalg.norm(joint_axis) + 1e-8)
        
        # For revolute joints: J = z × (p - p_i)
        # where z is joint axis, p is end-effector position, p_i is joint position
        r = ee_pos - joint_pos
        J_linear[:, i] = np.cross(joint_axis, r)
    
    return J_linear

def compute_jacobian_angular(robot_id, joint_indices, joint_angles):
    """
    Computes the angular Jacobian for the end-effector analytically.
    
    Args:
        robot_id (int): The ID of the robot in the PyBullet simulation.
        joint_indices (list): The indices of the joints to consider.
        joint_angles (list): The angles of the joints to compute the Jacobian.
    
    Returns:
        np.array: Angular Jacobian matrix (3 x n).
    """
    num_joints = len(joint_indices)
    J_angular = np.zeros((3, num_joints))
    
    # Get current transforms
    ee_state = p.getLinkState(robot_id, joint_indices[-1])
    ee_orientation = np.array(ee_state[5])  # World frame orientation
    
    # Compute Jacobian column by column
    for i, joint_idx in enumerate(joint_indices):
        # Get joint info
        joint_info = p.getJointInfo(robot_id, joint_idx)
        joint_state = p.getLinkState(robot_id, joint_idx)
        
        # Get joint axis in world frame
        joint_axis = np.array(joint_info[13])
        
        # Get joint orientation
        joint_orientation = np.array(joint_state[5])
        
        # Normalize joint axis
        joint_axis = joint_axis / (np.linalg.norm(joint_axis) + 1e-8)
        
        # For revolute joints: Angular Jacobian column is the joint rotation axis
        # transformed to the world frame
        R = quaternion_to_rotation_matrix(joint_orientation)
        J_angular[:, i] = R @ joint_axis
    
    return J_angular

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q (np.array): Quaternion [x, y, z, w].
        
    Returns:
        np.array: 3x3 rotation matrix.
    """
    x, y, z, w = q
    
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    return R


def assign_joint_weights(jacobian_linear, jacobian_angular):
    """
    Assign weights to joints based on their contributions to linear and angular movements.

    Args:
        jacobian_linear (np.array): The linear Jacobian matrix.
        jacobian_angular (np.array): The angular Jacobian matrix.

    Returns:
        tuple: Linear and angular weights for each joint.
    """
    # Compute the norm of each column (joint) to get its contribution to the linear and angular movements
    linear_weights = np.linalg.norm(jacobian_linear, axis=0)
    angular_weights = np.linalg.norm(jacobian_angular, axis=0)

    # Normalize the weights so that they sum to 1
    linear_weights /= (np.sum(linear_weights) + 1e-8)
    angular_weights /= (np.sum(angular_weights) + 1e-8)

    return linear_weights, angular_weights

def compute_weighted_joint_rewards(joint_errors, linear_weights, angular_weights, progress_reward):
    """
    Compute weighted joint rewards based on joint errors and movement weights.

    Args:
        joint_errors (list): List of errors for each joint.
        linear_weights (np.array): Linear movement weights for joints.
        angular_weights (np.array): Angular movement weights for joints.
        progress_reward (float): Reward based on progress towards the target.

    Returns:
        np.array: Array of individual joint rewards.
    """
    # Combine linear and angular weights
    combined_weights = linear_weights + angular_weights
    combined_weights /= np.sum(combined_weights) + 1e-8  # Normalize weights

    # Inverse of joint errors to reward lower errors more
    inverse_errors = 1.0 / (np.array(joint_errors) + 1e-6)
    normalized_inverse_errors = inverse_errors / (np.sum(inverse_errors) + 1e-8)

    
    # Calculate joint rewards proportional to progress and weights
    joint_rewards = progress_reward * combined_weights * normalized_inverse_errors

    return joint_rewards


def compute_shaped_distance(distance, begin_distance):
    """
    Shape the distance metrics to enhance sensitivity near the target.

    Args:
        distance (float): Current distance to the target.
        begin_distance (float): Initial distance to the target.

    Returns:
        tuple: (shaped_distance, shaped_begin_distance)
    """
    shaped_distance = np.log(distance + 1e-6)
    shaped_begin_distance = np.log(begin_distance + 1e-6)
    return shaped_distance, shaped_begin_distance

def compute_progress_reward(shaped_dist, shaped_begin, scale=1.0):
    """
    Computes reward based on progress toward the goal.
    
    Args:
        shaped_dist (float): Current shaped distance
        shaped_begin (float): Initial shaped distance
        scale (float): Reward scaling factor
    
    Returns:
        float: Progress reward
    """
    progress = shaped_begin - shaped_dist
    
    # Exponential reward scaling based on progress
    if progress > 0:
        reward = scale * (np.exp(progress) - 1)
    else:
        # Smaller penalty for negative progress to avoid discouraging exploration
        reward = scale * progress * 0.5
    
    return reward


def compute_orientation_bonus(quaternion_distance):
    """
    Compute the orientation bonus based on quaternion distance.

    Args:
        quaternion_distance (float): Distance between current and target orientations.

    Returns:
        float: Orientation bonus.
    """
    orientation_bonus = np.exp(-quaternion_distance)
    return orientation_bonus

def inverse_scaled_rewards(scaled_rewards, epsilon=1e-6):
    """
    Compute the inverse of scaled rewards, avoiding division by zero.

    Args:
        scaled_rewards (list): Scaled rewards to invert.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        list: Inverse of the scaled rewards.
    """
    inverse_rewards = []
    for reward in scaled_rewards:
        if reward == 0:
            inverse_rewards.append(1 / epsilon)  # Avoid division by zero
        else:
            inverse_rewards.append(1 / reward)
    return inverse_rewards



