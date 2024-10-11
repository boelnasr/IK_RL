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

def compute_jacobian_linear(robot_id, joint_indices, joint_angles, delta=1e-5):
    """
    Computes the linear Jacobian for the robot's end-effector using numerical differentiation.
    """
    link_index = joint_indices[-1]  # End-effector link index

    # Define the forward kinematics function for the specific robot configuration
    def forward_kinematics(joint_angles):
        return forward_kinematics_pyb(robot_id, joint_angles, link_index)
    
    # Compute the Jacobian using numerical differentiation
    jacobian = numerical_jacobian(forward_kinematics, joint_angles, delta)

    # Return only the linear part of the Jacobian (first 3 rows)
    return jacobian[0:3, :]
def compute_jacobian_angular(robot_id, joint_indices, joint_angles, delta=1e-5):
    """
    Computes the angular Jacobian for the robot's end-effector using numerical differentiation.
    """
    link_index = joint_indices[-1]  # End-effector link index

    # Define the forward kinematics function for the specific robot configuration
    def forward_kinematics(joint_angles):
        return forward_kinematics_pyb(robot_id, joint_angles, link_index)
    
    # Compute the Jacobian using numerical differentiation
    jacobian = numerical_jacobian(forward_kinematics, joint_angles, delta)

    # Return only the angular part of the Jacobian (last 3 rows)
    return jacobian[3:6, :]

def numerical_jacobian(robot_forward_kinematics, joint_angles, delta=1e-5):
    """
    Computes the Jacobian matrix using numerical differentiation.
    """
    num_joints = len(joint_angles)
    jacobian = np.zeros((6, num_joints))

    # Get the current end-effector pose (position and orientation)
    current_position, current_orientation = robot_forward_kinematics(joint_angles)

    # Iterate over each joint to compute the partial derivatives
    for i in range(num_joints):
        # Create a perturbed joint angle array
        perturbed_joint_angles = np.copy(joint_angles)
        perturbed_joint_angles[i] += delta

        # Get the perturbed end-effector pose
        perturbed_position, perturbed_orientation = robot_forward_kinematics(perturbed_joint_angles)

        # Compute numerical derivatives for position and orientation
        position_derivative = (perturbed_position - current_position) / delta
        orientation_derivative = (perturbed_orientation - current_orientation) / delta

        # Assign the derivatives to the corresponding column in the Jacobian
        jacobian[0:3, i] = position_derivative  # Linear part
        jacobian[3:6, i] = orientation_derivative  # Angular part

    return jacobian


def forward_kinematics_pyb(robot_id, joint_angles, link_index):
    """
    Compute the end-effector position and orientation using PyBullet's kinematics.
    """
    for i, angle in enumerate(joint_angles):
        p.resetJointState(robot_id, i, angle)
    link_state = p.getLinkState(robot_id, link_index)
    position = np.array(link_state[4])
    orientation = np.array(p.getEulerFromQuaternion(link_state[5]))
    return position, orientation

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
    Compute individual joint rewards based on their contribution and the overall reward.

    Args:
        joint_errors (list): Errors for each joint.
        linear_weights (np.array): Linear weights for each joint.
        angular_weights (np.array): Angular weights for each joint.
        overall_reward (float): The overall reward for the step.

    Returns:
        list: Individual rewards for each joint.
    """
    # Normalize the joint errors to compute their relative contribution
    joint_error_norms = np.linalg.norm(joint_errors)
    normalized_joint_errors = joint_errors / (joint_error_norms + 1e-8)

    # Compute individual rewards by considering the weighted errors
    joint_rewards = overall_reward * (linear_weights + angular_weights) * normalized_joint_errors
    return joint_rewards
