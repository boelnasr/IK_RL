import numpy as np
import pybullet as p
from ManipulaPy.utils import VecToso3, MatrixExp3, MatrixLog3  # Assuming you use ManipulaPy for Lie algebra utilities

def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles (roll, pitch, yaw).
    
    Args:
        R (np.array): A 3x3 rotation matrix.
    
    Returns:
        np.array: A 3-element array representing the Euler angles (roll, pitch, yaw) in radians.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])

def euler_angles_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.
    
    Args:
        roll (float): Rotation around the x-axis in radians.
        pitch (float): Rotation around the y-axis in radians.
        yaw (float): Rotation around the z-axis in radians.
    
    Returns:
        np.array: A 3x3 rotation matrix.
    """
    # Rotation matrix around the X-axis (roll)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    # Rotation matrix around the Y-axis (pitch)
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    # Rotation matrix around the Z-axis (yaw)
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def quaternion_to_euler(quaternion):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        quaternion (np.array): A 4-element array representing a quaternion (x, y, z, w).
    
    Returns:
        np.array: A 3-element array representing the Euler angles (roll, pitch, yaw) in radians.
    """
    return np.array(p.getEulerFromQuaternion(quaternion))

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion.
    
    Args:
        roll (float): Rotation around the x-axis in radians.
        pitch (float): Rotation around the y-axis in radians.
        yaw (float): Rotation around the z-axis in radians.
    
    Returns:
        np.array: A 4-element array representing a quaternion (x, y, z, w).
    """
    return np.array(p.getQuaternionFromEuler([roll, pitch, yaw]))

def skew_symmetric(v):
    """
    Convert a 3D vector into a skew-symmetric matrix.
    
    Args:
        v (np.array): A 3D vector.
    
    Returns:
        np.array: A 3x3 skew-symmetric matrix.
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def matrix_exp6(se3mat):
    """
    Compute the matrix exponential of an se(3) matrix.
    
    Args:
        se3mat (np.array): A 4x4 matrix from the se(3) Lie algebra.
    
    Returns:
        np.array: A 4x4 transformation matrix.
    """
    omega = np.array([se3mat[2, 1], se3mat[0, 2], se3mat[1, 0]])
    v = se3mat[0:3, 3]
    omega_norm = np.linalg.norm(omega)

    if omega_norm < 1e-6:
        return np.eye(4) + se3mat
    else:
        omega_skew = skew_symmetric(omega)
        R = MatrixExp3(omega_skew)
        omega_skew_squared = np.dot(omega_skew, omega_skew)
        G = (np.eye(3) + ((1 - np.cos(omega_norm)) / omega_norm**2) * omega_skew +
             ((omega_norm - np.sin(omega_norm)) / omega_norm**3) * omega_skew_squared)
        p = np.dot(G, v)
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = p
        return T

def matrix_log6(T):
    """
    Compute the matrix logarithm of a 4x4 transformation matrix from SE(3).
    
    Args:
        T (np.array): A 4x4 transformation matrix.
    
    Returns:
        np.array: A 4x4 matrix in the se(3) Lie algebra.
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    omega_skew = MatrixLog3(R)
    omega = np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])
    omega_norm = np.linalg.norm(omega)

    if omega_norm < 1e-6:
        return np.vstack((np.hstack((np.zeros((3, 3)), p.reshape(3, 1))), [0, 0, 0, 0]))
    else:
        G_inv = (np.eye(3) - omega_skew / 2 +
                 (1 - omega_norm * np.sin(omega_norm) / (2 * (1 - np.cos(omega_norm)))) / omega_norm**2 * np.dot(omega_skew, omega_skew))
        v = np.dot(G_inv, p)
        return np.vstack((np.hstack((omega_skew, v.reshape(3, 1))), [0, 0, 0, 0]))

def vec_to_se3(V):
    """
    Converts a 6D spatial velocity vector to an se(3) matrix.
    
    Args:
        V (np.array): A 6D spatial velocity vector.
    
    Returns:
        np.array: A 4x4 matrix in the se(3) Lie algebra.
    """
    return np.vstack((np.hstack((VecToso3(V[:3]), V[3:].reshape(3, 1))), [0, 0, 0, 0]))

def extract_rotation_translation(T):
    """
    Extract the rotation matrix and translation vector from a 4x4 transformation matrix.
    
    Args:
        T (np.array): A 4x4 transformation matrix.
    
    Returns:
        tuple: A tuple containing the 3x3 rotation matrix and 3D translation vector.
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    return R, p

def generate_dataset(num_samples, joint_limits, robot_id, joint_indices):
    """
    Generates a dataset of random joint angles and corresponding end-effector poses.

    Args:
        num_samples (int): Number of samples to generate.
        joint_limits (list): List of tuples representing joint limits (min, max) for each joint.
        robot_id (int): PyBullet ID of the robot.
        joint_indices (list): List of joint indices.

    Returns:
        tuple: A tuple containing two arrays:
            - end_effector_poses: An array of end-effector poses (position + orientation).
            - joint_angles: An array of corresponding joint angles.
    """
    joint_angles_list = []
    end_effector_poses_list = []

    for _ in range(num_samples):
        # Randomly sample joint angles within joint limits
        joint_angles = np.array([
            np.random.uniform(low, high) for (low, high) in joint_limits
        ])

        # Set the robot to these joint angles
        for idx, angle in zip(joint_indices, joint_angles):
            p.resetJointState(robot_id, idx, angle)

        # Get the end-effector pose
        end_effector_state = p.getLinkState(robot_id, joint_indices[-1])
        position = np.array(end_effector_state[4])
        orientation = np.array(end_effector_state[5])  # Quaternion

        # Store the data
        joint_angles_list.append(joint_angles)
        end_effector_poses_list.append(np.concatenate([position, orientation]))

    return np.array(end_effector_poses_list), np.array(joint_angles_list)

def normalize_observations(obs, obs_mean, obs_std):
    """
    Normalize observations based on provided mean and standard deviation.
    
    Args:
        obs (np.array): The observation to normalize.
        obs_mean (np.array): The mean of the observations.
        obs_std (np.array): The standard deviation of the observations.
    
    Returns:
        np.array: The normalized observation.
    """
    return (obs - obs_mean) / (obs_std + 1e-8)
