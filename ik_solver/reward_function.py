import numpy as np

def compute_position_error(current_position, target_position):
    """
    Computes the Euclidean distance (L2 norm) between the current and target positions.

    Args:
        current_position (np.array): The current position of the end-effector.
        target_position (np.array): The target position of the end-effector.

    Returns:
        float: The position error.
    """
    return np.linalg.norm(current_position - target_position)

def compute_orientation_error(current_orientation, target_orientation):
    """
    Computes the angular difference between the current and target orientations using quaternions.

    Args:
        current_orientation (np.array): The current orientation quaternion.
        target_orientation (np.array): The target orientation quaternion.

    Returns:
        float: The orientation error in radians.
    """
    dot_product = np.clip(np.abs(np.dot(current_orientation, target_orientation)), -1.0, 1.0)
    return 2 * np.arccos(dot_product)

def compute_joint_error(current_joint_angles, target_joint_angles):
    """
    Computes the absolute difference between the current and target joint angles.

    Args:
        current_joint_angles (np.array): Current joint angles.
        target_joint_angles (np.array): Target joint angles.

    Returns:
        np.array: The joint angle errors.
    """
    return np.abs(current_joint_angles - target_joint_angles)

def smooth_reward(new_reward, previous_reward, smoothing_factor=0.9):
    """
    Applies exponential smoothing to the reward to reduce fluctuations.

    Args:
        new_reward (float): The new computed reward.
        previous_reward (float): The previous smoothed reward.
        smoothing_factor (float): Weight for the previous reward (default is 0.9).

    Returns:
        float: Smoothed reward.
    """
    return smoothing_factor * previous_reward + (1 - smoothing_factor) * new_reward

def compute_joint_limit_penalty(joint_angles, joint_limits):
    """
    Computes a penalty for joint limit violations.

    Args:
        joint_angles (np.array): Current joint angles.
        joint_limits (list): List of tuples representing joint limits (min, max) for each joint.

    Returns:
        float: The total penalty for joint limit violations.
    """
    penalty = 0.0
    for angle, (min_limit, max_limit) in zip(joint_angles, joint_limits):
        if angle < min_limit or angle > max_limit:
            penalty += 0.5  # Arbitrary penalty value for joint limit violations
    return penalty


def compute_combined_reward(
    current_position,
    target_position,
    current_orientation,
    target_orientation,
    current_joint_angles,
    target_joint_angles,
    joint_limits,
    previous_joint_angles,
    iteration,
    position_weight=0,
    orientation_weight=0,
    joint_weight=1,
    smoothness_weight=0,
    joint_limit_penalty_weight=0,
    success_bonus=5.0,
    max_reward=1.0,
    min_reward=0.0
):
    """
    Computes the total reward by combining position, orientation, and joint errors,
    and applies penalties for joint limit violations and large joint movements.

    Returns:
        tuple: (total_reward, position_error, orientation_error, joint_error, success)
    """
    # Compute errors
    position_error = compute_position_error(current_position, target_position)
    orientation_error = compute_orientation_error(current_orientation, target_orientation)
    joint_error = compute_joint_error(current_joint_angles, target_joint_angles)

    # Calculate negative errors as rewards
    position_reward = -position_error
    orientation_reward = -orientation_error
    joint_reward = -np.mean(joint_error)

    # Smoothness penalty
    if previous_joint_angles is not None:
        smoothness_penalty = np.mean(np.abs(current_joint_angles - previous_joint_angles))
    else:
        smoothness_penalty = 0.0

    # Compute joint limit penalty
    joint_limit_penalty = 0.0
    for angle, (min_limit, max_limit) in zip(current_joint_angles, joint_limits):
        if angle < min_limit:
            penalty = joint_limit_penalty_weight * (min_limit - angle)
            joint_limit_penalty += penalty
        elif angle > max_limit:
            penalty = joint_limit_penalty_weight * (angle - max_limit)
            joint_limit_penalty += penalty

    # Total reward computation
    total_reward = (
        position_weight * position_reward +
        orientation_weight * orientation_reward +
        joint_weight * joint_reward -
        smoothness_weight * smoothness_penalty -
        joint_limit_penalty  # Subtract joint limit penalty
    )

    # Success condition with thresholds
    position_threshold = 0.05  # Adjust as needed
    orientation_threshold = 0.2  # Adjust as needed
    joint_threshold = 0.05  # Adjust as needed

    success = (
        position_error < position_threshold and
        orientation_error < orientation_threshold and
        np.all(joint_error < joint_threshold)
    )

    # Apply success bonus
    if success:
        total_reward += success_bonus

    # Clip total reward
    total_reward = np.clip(total_reward, min_reward, max_reward)

    return total_reward, position_error, orientation_error, joint_error, success



def calculate_joint_contribution(joint_idx, position_error, orientation_error):
    """
    Calculates the contribution of a specific joint based on the position and orientation errors.

    Args:
        joint_idx (int): Index of the joint.
        position_error (np.array): The position error of the end-effector.
        orientation_error (np.array): The orientation error of the end-effector.

    Returns:
        float: Contribution of the joint to the overall error.
    """
    contribution = -np.linalg.norm(position_error) - np.linalg.norm(orientation_error)
    return contribution


def calculate_joint_limit_penalty(joint_idx, joint_angles, joint_limits):
    """
    Calculates a penalty if a specific joint is near its limit.

    Args:
        joint_idx (int): Index of the joint.
        joint_angles (np.array): Array of current joint angles.
        joint_limits (list of tuples): List of joint limits [(lower_limit, upper_limit)] for each joint.

    Returns:
        float: Penalty if the joint is near its limits.
    """
    joint_angle = joint_angles[joint_idx]
    lower_limit, upper_limit = joint_limits[joint_idx]
    
    # Penalty if the joint angle is outside its limit
    penalty = max(0, joint_angle - upper_limit) + max(0, lower_limit - joint_angle)
    return penalty


def calculate_joint_smoothness(joint_idx, joint_angles, previous_joint_angles):
    """
    Calculates a smoothness penalty for a specific joint, penalizing large changes in joint angle.

    Args:
        joint_idx (int): Index of the joint.
        joint_angles (np.array): Array of current joint angles.
        previous_joint_angles (np.array): Array of previous joint angles.

    Returns:
        float: Smoothness penalty based on the difference between the current and previous joint angles.
    """
    if previous_joint_angles is None:
        return 0  # No penalty for the first step

    current_angle = joint_angles[joint_idx]
    previous_angle = previous_joint_angles[joint_idx]
    
    smoothness_penalty = np.abs(current_angle - previous_angle)
    return smoothness_penalty


