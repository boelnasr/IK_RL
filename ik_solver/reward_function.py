import numpy as np
import pybullet as p

def compute_position_error(current_position, target_position):
    """
    Enhanced position error calculation with debugging.
    """
    error = np.linalg.norm(current_position - target_position)
    # print(f"Debug - Position Error: {error:.6f}")
    # print(f"Debug - Current Position: {current_position}")
    # print(f"Debug - Target Position: {target_position}")
    return max(error, 1e-6)  # Ensure non-zero error


def compute_quaternion_distance(current_orientation, target_orientation):
    """
    Compute quaternion distance with improved numerical stability.
    
    Args:
        current_orientation (np.array): Current orientation quaternion
        target_orientation (np.array): Target orientation quaternion
    
    Returns:
        float: Angular distance in radians
    """
    # Ensure inputs are numpy arrays
    current_orientation = np.array(current_orientation)
    target_orientation = np.array(target_orientation)
    
    # Compute dot product with improved numerical stability
    dot_product = np.clip(
        np.abs(np.dot(current_orientation, target_orientation)),
        -1.0 + 1e-7,
        1.0 - 1e-7
    )
    
    # Convert to angle
    distance = 2 * np.arccos(dot_product)
    
    return max(distance, 1e-7)  # Ensure non-zero distance

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
#     reward = np.clip(reward, -50, 50)
    
#     return reward, prev_best, success



def compute_reward(distance, begin_distance, prev_best, current_orientation, target_orientation, 
                   joint_errors, linear_weights, angular_weights,
                   success_threshold=0.006, time_penalty=-0.1):
    """
    Enhanced reward function with logarithmic shaping and improvement tracking.
    
    Args:
        distance (float): Current distance to target.
        begin_distance (float): Initial distance to target.
        prev_best (float): Best distance achieved so far.
        current_orientation (np.array): Current end-effector orientation quaternion.
        target_orientation (np.array): Target orientation quaternion.
        joint_errors (list): List of errors for each joint.
        linear_weights (np.array): Weights for each joint based on linear movement.
        angular_weights (np.array): Weights for each joint based on angular movement.
        success_threshold (float): Distance threshold for success.
        time_penalty (float): Small penalty per timestep to encourage efficiency.
    
    Returns:
        tuple: (reward, prev_best, success)
    """
    success = False
    epsilon = 1e-6
    
    # Shape the distance metrics logarithmically
    shaped_dist, shaped_begin = compute_shaped_distance(distance, begin_distance)
    
    # Compute progress reward with logarithmic scaling
    progress_reward = compute_progress_reward(shaped_dist, shaped_begin)
    
    # Compute orientation reward
    quaternion_distance = compute_quaternion_distance(current_orientation, target_orientation)
    orientation_reward = compute_orientation_bonus(quaternion_distance)
    
    # Calculate improvement with logarithmic scaling
    if distance < prev_best:
        # Logarithmic improvement bonus
        improvement = -np.log(np.maximum(distance / prev_best, epsilon))
        prev_best = distance
    else:
        # Logarithmic penalty for moving away
        improvement = -np.log1p((distance - prev_best) / begin_distance)
    
    # Ensure arrays and normalize weights
    joint_errors = np.array(joint_errors)
    linear_weights = np.array(linear_weights)
    angular_weights = np.array(angular_weights)
    
    # Logarithmic error scaling for joints
    log_joint_errors = -np.log(np.maximum(joint_errors, epsilon))
    max_log_error = np.max(np.abs(log_joint_errors))
    normalized_joint_errors = log_joint_errors / (max_log_error + epsilon)
    
    # Combine weights
    combined_weights = np.maximum(linear_weights + angular_weights, epsilon)
    combined_weights /= (np.sum(combined_weights) + epsilon)
    
    # Calculate base reward components
    progress_contribution = progress_reward * np.mean(combined_weights)
    orientation_contribution = orientation_reward * np.mean(combined_weights)
    improvement_contribution = improvement * np.mean(combined_weights)
    
    # Joint efficiency (logarithmically scaled)
    efficiency = np.mean(normalized_joint_errors * np.log1p(1 / (joint_errors + epsilon)))
    
    # Combine all components
    reward = (
        0.3 * progress_contribution +
        0.3 * orientation_contribution +
        0.2 * improvement_contribution +
        0.2 * efficiency
    )
    
    # Success bonus with logarithmic precision scaling
    if distance <= success_threshold:
        precision_ratio = distance / success_threshold
        precision_bonus = -20.0 * np.log(precision_ratio)
        
        # Additional bonus for efficient joints
        if np.mean(joint_errors) < np.max(joint_errors):
            efficiency_bonus = -10.0 * np.log(np.mean(joint_errors) / np.max(joint_errors))
            precision_bonus += efficiency_bonus
            
        reward += precision_bonus
        success = True
    
    # Time penalty with logarithmic decay
    progress_ratio = distance / begin_distance
    adaptive_time_penalty = time_penalty * np.log1p(progress_ratio)
    reward += adaptive_time_penalty
    
    # Logarithmic reward scaling
    if reward > 0:
        reward = np.log1p(reward)
    else:
        reward = -np.log1p(-reward)
    
    # Clip rewards with logarithmic bounds
    if not success:
        reward = np.clip(reward, -5.0, 5.0)
    else:
        reward = np.clip(reward, -5.0, 10.0)
    
    # Debug information
    if np.random.random() < 0.01:  # Print occasionally
        print("\n=== Reward Computation Debug ===")
        print(f"Distance: {distance:.6f}, Begin: {begin_distance:.6f}")
        print(f"Improvement: {improvement:.4f}")
        print(f"Progress Reward: {progress_reward:.4f}")
        print(f"Orientation Reward: {orientation_reward:.4f}")
        print(f"Joint Errors: {joint_errors}")
        print(f"Combined Reward: {reward:.4f}")
        print(f"Success Threshold: {success_threshold}")
        print(f"Success: {success}")

    return reward, prev_best, success

def compute_jacobian_linear(robot_id, joint_indices, joint_angles):
    """
    Computes the linear Jacobian for the end-effector.

    Args:
        robot_id (int): The ID of the robot in the PyBullet simulation.
        joint_indices (list): The indices of the joints to consider.
        joint_angles (list): The angles of the joints to compute the Jacobian.

    Returns:
        np.array: Linear Jacobian matrix.
    """
    zero_vec = [0.0] * len(joint_indices)
    jacobian_linear, _ = p.calculateJacobian(
        bodyUniqueId=robot_id,
        linkIndex=joint_indices[-1],
        localPosition=[0, 0, 0],
        objPositions=joint_angles,
        objVelocities=zero_vec,
        objAccelerations=zero_vec
    )
    return np.array(jacobian_linear)

def compute_jacobian_angular(robot_id, joint_indices, joint_angles):
    """
    Computes the angular Jacobian for the end-effector.

    Args:
        robot_id (int): The ID of the robot in the PyBullet simulation.
        joint_indices (list): The indices of the joints to consider.
        joint_angles (list): The angles of the joints to compute the Jacobian.

    Returns:
        np.array: Angular Jacobian matrix.
    """
    zero_vec = [0.0] * len(joint_indices)
    _, jacobian_angular = p.calculateJacobian(
        bodyUniqueId=robot_id,
        linkIndex=joint_indices[-1],
        localPosition=[0, 0, 0],
        objPositions=joint_angles,
        objVelocities=zero_vec,
        objAccelerations=zero_vec
    )
    return np.array(jacobian_angular)

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

def compute_weighted_joint_rewards(joint_errors, linear_weights, angular_weights, overall_reward):
    """
    Enhanced joint reward calculation with debugging and minimum values.
    """
    epsilon = 1e-6
    
    # Print debug information
    #print("Debug - Joint Errors:", joint_errors)
    #print("Debug - Overall Reward:", overall_reward)
    
    # Ensure joint errors aren't zero
    joint_errors = np.array(joint_errors)
    joint_errors = np.maximum(np.abs(joint_errors), epsilon)
    
    # Calculate error norm with minimum value
    joint_error_norms = max(np.linalg.norm(joint_errors), epsilon)
    normalized_joint_errors = joint_errors / joint_error_norms
    
    #print("Debug - Normalized Joint Errors:", normalized_joint_errors)
    
    # Combine weights with minimum values
    combined_weights = np.maximum(linear_weights + angular_weights, epsilon)
    
    # Calculate rewards with minimum value
    joint_rewards = overall_reward * combined_weights * normalized_joint_errors*10
    
    # Ensure at least some small reward/penalty
    joint_rewards = np.where(joint_rewards == 0, 
                           np.sign(overall_reward) * epsilon,
                           joint_rewards)
    
    #print("Debug - Final Joint Rewards:", joint_rewards)
    
    return joint_rewards

def compute_shaped_distance(distance, begin_distance, max_distance=1.0):
    """
    Shapes the distance metric using a smoothed normalization.
    
    Args:
        distance (float): Current distance to target
        begin_distance (float): Initial distance to target
        max_distance (float): Maximum expected distance for normalization
    
    Returns:
        float: Shaped distance metric between 0 and 1
    """
    # Normalize distances to [0,1] range with smooth clipping
    norm_dist = np.clip(distance / max_distance, 0, 1)
    norm_begin = np.clip(begin_distance / max_distance, 0, 1)
    
    # Apply smooth shaping function
    shaped_dist = 1 - np.exp(-2 * norm_dist)
    shaped_begin = 1 - np.exp(-2 * norm_begin)
    
    return shaped_dist, shaped_begin

def compute_progress_reward(shaped_dist, shaped_begin, scale=1000.0):
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

def compute_orientation_bonus(quaternion_distance, max_angle=np.pi):
    """
    Computes additional reward for good orientation alignment.
    
    Args:
        quaternion_distance (float): Angular distance between current and target orientation
        max_angle (float): Maximum expected angular distance
    
    Returns:
        float: Orientation bonus reward
    """
    # Normalize and shape the orientation reward
    norm_angle = np.clip(quaternion_distance / max_angle, 0, 1)
    orientation_bonus = np.exp(-3 * norm_angle) - 0.05
    return max(orientation_bonus, 0) 

