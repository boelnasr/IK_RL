import numpy as np
import pybullet as p

def compute_position_error(current_position, target_position):
    """
    Enhanced position error calculation with debugging.
    """
    error = np.linalg.norm(current_position - target_position)
    print(f"Debug - Position Error: {error:.6f}")
    print(f"Debug - Current Position: {current_position}")
    print(f"Debug - Target Position: {target_position}")
    return max(error, 1e-6)  # Ensure non-zero error

def compute_quaternion_distance(current_orientation, target_orientation):
    """
    Enhanced quaternion distance calculation with debugging.
    """
    # Ensure inputs are numpy arrays
    current_orientation = np.array(current_orientation)
    target_orientation = np.array(target_orientation)
    
    print(f"Debug - Current Orientation: {current_orientation}")
    print(f"Debug - Target Orientation: {target_orientation}")
    
    # Compute dot product with safety checks
    dot_product = np.clip(np.dot(current_orientation, target_orientation), -1.0 + 1e-6, 1.0 - 1e-6)
    distance = 2 * np.arccos(np.abs(dot_product))
    
    print(f"Debug - Quaternion Distance: {distance:.6f}")
    return max(distance, 1e-6)  # Ensure non-zero distance

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


def compute_reward(distance, begin_distance, prev_best, success_threshold=0.006):
    """
    Computes the reward or punishment based on the current distance.

    Args:
        distance (float): The current overall distance.
        begin_distance (float): The initial overall distance at the start of the iteration.
        prev_best (float): The closest distance achieved so far in the iteration.
        success_threshold (float): The distance threshold for success.

    Returns:
        tuple:
            reward (float): The computed reward or punishment.
            prev_best (float): Updated closest distance.
            success (bool): Indicates if the target was reached.
    """
    success = False

    if distance > prev_best:
        # The agent moved away from the target, apply punishment
        reward = prev_best - distance  # Negative value
    else:
        # The agent moved closer to the target, provide reward
        reward = abs(begin_distance - distance)/10
        prev_best = distance  # Update prev_best

    # Check if the agent has reached the target
    if distance <= success_threshold:
        reward = 2000 + (success_threshold - distance)*1000
        success = True

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
    print("Debug - Joint Errors:", joint_errors)
    print("Debug - Overall Reward:", overall_reward)
    
    # Ensure joint errors aren't zero
    joint_errors = np.array(joint_errors)
    joint_errors = np.maximum(np.abs(joint_errors), epsilon)
    
    # Calculate error norm with minimum value
    joint_error_norms = max(np.linalg.norm(joint_errors), epsilon)
    normalized_joint_errors = joint_errors / joint_error_norms
    
    print("Debug - Normalized Joint Errors:", normalized_joint_errors)
    
    # Combine weights with minimum values
    combined_weights = np.maximum(linear_weights + angular_weights, epsilon)
    
    # Calculate rewards with minimum value
    joint_rewards = overall_reward * combined_weights * normalized_joint_errors
    
    # Ensure at least some small reward/penalty
    joint_rewards = np.where(joint_rewards == 0, 
                           np.sign(overall_reward) * epsilon,
                           joint_rewards)
    
    print("Debug - Final Joint Rewards:", joint_rewards)
    
    return joint_rewards