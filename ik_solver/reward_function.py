import numpy as np
import pybullet as p

def compute_position_error(current_position, target_position):
    """
    Computes the Euclidean distance between the current and target positions.

    Args:
        current_position (np.array): The current position of the end-effector.
        target_position (np.array): The target position of the end-effector.

    Returns:
        float: The Euclidean distance.
    """
    return np.linalg.norm(current_position - target_position)

def compute_quaternion_distance(current_orientation, target_orientation):
    """
    Computes the quaternion distance between two quaternions.

    Args:
        current_orientation (np.array): Current orientation quaternion [w, x, y, z].
        target_orientation (np.array): Target orientation quaternion [w, x, y, z].

    Returns:
        float: Quaternion distance in radians.
    """
    # Compute the dot product of the quaternions
    dot_product = np.clip(np.dot(current_orientation, target_orientation), -1.0, 1.0)
    
    # Calculate the angular distance using arccos of the dot product
    distance = 2 * np.arccos(np.abs(dot_product))
    return distance

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
        reward = begin_distance - distance
        prev_best = distance  # Update prev_best

    # Check if the agent has reached the target
    if distance <= success_threshold:
        reward = 2.0 + (success_threshold - distance) * 1000
        success = True

    return reward, prev_best, success

<<<<<<< HEAD
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
=======
>>>>>>> IK_RL/main
