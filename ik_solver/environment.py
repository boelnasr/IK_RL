import gym
import pybullet as p
import pybullet_data
from typing import Dict, List, Tuple    
import numpy as np
import logging
from config import config
from .reward_function import (
    compute_position_error,         # To compute position errors between current and target positions
    compute_quaternion_distance,    # To compute quaternion distance between current and target orientations
    compute_overall_distance,       # To compute the combined distance metric
    compute_reward,                 # To compute the overall reward based on the distance metrics
    compute_jacobian_linear,        # To compute the linear Jacobian matrix for the end-effector
    compute_jacobian_angular,        # To compute the angular Jacobian matrix for the end-effector
    assign_joint_weights          # To assign weights to linear and angular Jacobians
)
import os

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InverseKinematicsEnv(gym.Env):
    """
    Custom environment for inverse kinematics control of a robotic arm in PyBullet.
    Each joint is controlled by an individual agent.
    """
    def __init__(self, robot_name="kuka_iiwa", sim_timestep=1./240.):
        super(InverseKinematicsEnv, self).__init__()
        self.episode_number = 0
        self.total_episodes = config.get('num_episodes', 2000)
        self.max_episode_steps = config.get('max_episode_steps', 5000)
        # Define minimum and maximum success thresholds
        self.min_success_threshold = 0.01  # Minimum threshold (more strict)
        self.max_success_threshold = 0.1   # Maximum threshold (more lenient)

        # Initialize success threshold to the maximum value
        self.success_threshold = self.max_success_threshold

        # PyBullet setup
        self.physics_client = p.connect(p.DIRECT)  # Use p.DIRECT for headless simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # PyBullet data path
        p.setGravity(0, 0, -9.8)  # Set gravity for the simulation
        self.sim_timestep = sim_timestep
        p.setTimeStep(self.sim_timestep)

        # Load the selected robot
        if robot_name == "kuka_iiwa":
            self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
        elif robot_name == "ur5":
            # Provide the path to the UR5 URDF file
            ur5_urdf_path = os.path.join("ur5/ur5.urdf")
            ur5_urdf_directory = os.path.dirname(ur5_urdf_path)
            p.setAdditionalSearchPath(ur5_urdf_directory)  # Add UR5 model path
            self.robot_id = p.loadURDF(ur5_urdf_path, useFixedBase=True)
        elif robot_name == "xarm":
            xarm_urdf_path = os.path.join("xarm/xarm6_robot.urdf")
            xarm_urdf_directory = os.path.dirname(xarm_urdf_path)
            p.setAdditionalSearchPath(xarm_urdf_directory)  # Add XArm model path
            self.robot_id = p.loadURDF(xarm_urdf_path, useFixedBase=True)
        else:
            raise ValueError(f"Robot {robot_name} is not available in PyBullet data.")

        # Initialize joint information
        self.joint_indices = []
        self.joint_limits = []

        # Filter out fixed joints and include only revolute joints
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                if lower_limit > upper_limit:
                    # Handle continuous joints
                    lower_limit = -np.pi
                    upper_limit = np.pi
                self.joint_limits.append((lower_limit, upper_limit))
                print(f"Active Joint {i}: {joint_name}, Type: {joint_type}, Limits: {lower_limit}, {upper_limit}")
            else:
                print(f"Skipped Joint {i}: {joint_name}, Type: {joint_type}")

        self.num_joints = len(self.joint_indices)
        print(f"Number of active joints in the robot: {self.num_joints}")

        # Track previous joint angles
        self.previous_joint_angles = None  # Will be initialized in reset

        # Define action and observation spaces per agent
        self.action_spaces = []
        self.observation_spaces = []
        for idx in self.joint_indices:
            i = self.joint_indices.index(idx)
            # Action space for each joint (scalar)
            action_low = np.array([self.joint_limits[i][0]], dtype=np.float32)
            action_high = np.array([self.joint_limits[i][1]], dtype=np.float32)
            action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
            self.action_spaces.append(action_space)

            # Observation space for each agent
            obs_space = gym.spaces.Dict({
                'joint_angle': gym.spaces.Box(low=action_low, high=action_high, shape=(1,), dtype=np.float32),
                'position_error': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                'orientation_error': gym.spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32),
                # Include other relevant observations if needed
            })
            self.observation_spaces.append(obs_space)

        # Combined action and observation spaces (for compatibility)
        self.action_space = gym.spaces.Tuple(self.action_spaces)
        self.observation_space = gym.spaces.Tuple(self.observation_spaces)

        # Initialize step counter and thresholds
        self.current_step = 0
        self.position_threshold = 0.1  # Define appropriate value
        self.orientation_threshold = 0.1  # Define appropriate value in radians

        print(f"Number of joints in the robot: {self.num_joints}")

    def update_success_threshold(self):
        """
        Updates the success threshold in intervals:
        - Decreases at regular intervals for the first 2/3 of the episodes.
        - Remains constant at the minimum threshold for the last 1/3 of the episodes.
        """
        total_episodes = self.total_episodes
        max_threshold = self.max_success_threshold
        min_threshold = self.min_success_threshold
        current_episode = self.episode_number

        # Define intervals and decrease logic
        decrease_phase_end = int(3 * total_episodes / 2)  # First 2/3 episodes
        interval_count = 10  # Number of intervals for decreasing
        episodes_per_interval = decrease_phase_end // interval_count

        if current_episode <= decrease_phase_end:
            # Determine the current interval
            current_interval = current_episode // episodes_per_interval

            # Avoid exceeding the number of intervals
            if current_interval >= interval_count:
                current_interval = interval_count - 1

            # Calculate the threshold based on the interval
            self.success_threshold = max_threshold - (
                (max_threshold - min_threshold) * current_interval / (interval_count - 1)
            )
        else:
            # After 2/3 episodes, hold constant at the minimum threshold
            self.success_threshold = min_threshold

        return self.success_threshold
    def compute_forward_kinematics(self, joint_angles):
        """
        Computes the end-effector's position and orientation based on joint angles.

        Args:
            joint_angles (array): Current joint angles.

        Returns:
            array: End-effector pose [x, y, z, qx, qy, qz, qw].
        """
        # Update the joint states
        for i, angle in zip(self.joint_indices, joint_angles):
            p.resetJointState(self.robot_id, i, angle)
        
        # Get the state of the end-effector link
        link_state = p.getLinkState(
            self.robot_id,
            self.joint_indices[-1],
            computeForwardKinematics=True
        )
        
        position = link_state[0]  # (x, y, z)
        orientation = link_state[1]  # (qx, qy, qz, qw)
        
        return np.concatenate([position, orientation])
    
    def reset(self, difficulty=1.0):
        """
        Resets the environment and initializes all necessary state tracking variables.
        """
        # Adjust success threshold based on episode progress
        #episode_progress = self.episode_number / self.total_episodes
        #self.success_threshold = self.update_success_threshold()+
        self.success_threshold = 0.01
        #print(f"Success Threshold: {self.success_threshold}")
        #self.success_threshold = 0.01
        self.current_difficulty = difficulty

        # Reset joint angles to random values within limits
        self.joint_angles = np.array([
            np.random.uniform(limit[0], limit[1]) for limit in self.joint_limits
        ])
        for i, angle in zip(self.joint_indices, self.joint_angles):
            p.resetJointState(self.robot_id, i, angle)

        # Track previous joint angles
        self.previous_joint_angles = np.copy(self.joint_angles)

        # Generate random target pose
        self.target_position = np.random.uniform([0.1, -0.5, 0.1], [0.5, 0.5, 0.5])
        random_euler = np.random.uniform(-np.pi, np.pi, 3)
        self.target_orientation = np.array(p.getQuaternionFromEuler(random_euler))

        # Get initial end-effector state
        self.current_position, self.current_orientation = self.get_end_effector_pose()
        
        # Store quaternions for reward calculation
        self.current_quaternion = self.current_orientation
        self.target_quaternion = self.target_orientation

        # Initialize distance tracking
        self.current_distance = compute_overall_distance(
            current_position=self.current_position,
            target_position=self.target_position,
            current_orientation=self.current_quaternion,
            target_orientation=self.target_quaternion
        )
        self.initial_distance = self.current_distance
        self.previous_best_distance = self.current_distance

        # Reset step counter and episode tracking
        self.current_step = 0
        self.episode_number += 1

        # Visualize target (optional)
        # if hasattr(self, 'target_marker'):
        #     p.removeBody(self.target_marker)
        # self.target_marker = p.loadURDF(
        #     "sphere_small.urdf",
        #     self.target_position,
        #     globalScaling=0.05,
        #     useFixedBase=True
        # )

        # Calculate initial errors
        self.position_error = self.current_position - self.target_position
        self.orientation_error = self.compute_orientation_difference(
            self.current_orientation, self.target_orientation
        )

        return self.get_all_agent_observations()


    def step(self, actions):
        """
        Executes one environment step with error handling, success computation, and safety checks.
        """
        try:
            # Apply clipped actions to joints
            for i, action in enumerate(actions):
                action = np.clip(float(action), self.joint_limits[i][0], self.joint_limits[i][1])
                try:
                    p.setJointMotorControl2(
                        self.robot_id, self.joint_indices[i], p.POSITION_CONTROL, targetPosition=action
                    )
                except Exception as e:
                    logging.warning(f"Failed to set joint {i} target position: {e}")

            # Step simulation
            try:
                p.stepSimulation()
                self.current_step += 1
            except Exception as e:
                logging.error(f"Error stepping simulation: {e}")
                raise

            # Update joint states
            self.joint_angles = np.array([
                p.getJointState(self.robot_id, i)[0] for i in self.joint_indices
            ], dtype=np.float32)

            # Get end-effector pose
            self.current_position, self.current_orientation = self.get_end_effector_pose()
            self.current_quaternion = np.array(self.current_orientation, dtype=np.float32)

            # Compute errors
            self.position_error = self.current_position - self.target_position
            self.orientation_error = self.compute_orientation_difference(
                self.current_orientation, self.target_orientation
            )
            self.current_distance = compute_overall_distance(
                self.current_position, self.target_position,
                self.current_orientation, self.target_orientation
            )
            self.joint_errors = np.abs(self.joint_angles - self.previous_joint_angles)

            # Compute Jacobians and weights
            jacobian_linear = compute_jacobian_linear(self.robot_id, self.joint_indices, self.joint_angles)
            jacobian_angular = compute_jacobian_angular(self.robot_id, self.joint_indices, self.joint_angles)
            self.linear_weights, self.angular_weights = assign_joint_weights(jacobian_linear, jacobian_angular)

            # Compute rewards
            try:
                rewards, individual_rewards, self.previous_best_distance, overall_success = compute_reward(
                    distance=float(self.current_distance),
                    begin_distance=float(self.initial_distance),
                    prev_best=float(self.previous_best_distance),
                    current_orientation=self.current_quaternion.tolist(),
                    target_orientation=self.target_quaternion.tolist(),
                    joint_errors=self.joint_errors.tolist(),
                    linear_weights=self.linear_weights.tolist(),
                    angular_weights=self.angular_weights.tolist(),
                    episode_number=self.episode_number,
                    total_episodes=self.total_episodes,
                    success_threshold=float(self.success_threshold)
                )
            except Exception as e:
                logging.error(f"Reward computation error: {e}")
                rewards, individual_rewards, overall_success = -1.0, [-1.0] * len(self.joint_indices), False

            # Determine success for each agent and overall success rate
            agent_successes = [self.is_agent_success(i, self.joint_errors) for i in range(self.num_joints)]
            success_rate = sum(agent_successes) / self.num_joints
            overall_success = success_rate == 1.0  # Overall success if all agents succeed
            individual_rewards= np.clip(individual_rewards, -20.0, 1200.0)
            

            # Check if episode is done
            done = self.current_step >= self.max_episode_steps or overall_success
            self.previous_joint_angles = np.copy(self.joint_angles)

            # Construct info dictionary
            info = {
                'success_per_joint': agent_successes,
                'overall_success_rate': success_rate,
                'current_distance': float(self.current_distance),
                'initial_distance': float(self.initial_distance),
                'best_distance': float(self.previous_best_distance),
                'position_error': float(np.linalg.norm(self.position_error)),
                'orientation_error': float(np.linalg.norm(self.orientation_error)),
                'mean_joint_error': float(np.mean(self.joint_errors)),
                'step': self.current_step,
                'individual_rewards': individual_rewards,
            }

            return self.get_all_agent_observations(), rewards.tolist(), done, info

        except Exception as e:
            logging.error(f"Critical error in step: {e}")
            return (
                self.get_all_agent_observations(),
                [-1.0] * len(self.joint_indices),
                True,
                {'error': str(e)}
            )


    def get_all_agent_observations(self):
        """
        Retrieves observations for all agents.

        Returns:
            list: A list of observations for each agent.
        """
        observations = []
        for i in range(self.num_joints):
            obs = self.get_agent_observation(i)
            observations.append(obs)
        return observations



    def get_agent_observation(self, joint_index):
        """
        Retrieves the observation for a specific agent controlling a joint.

        Args:
            joint_index (int): The index of the joint (agent).

        Returns:
            dict: The observation for the agent.
        """
        joint_angle = np.array([self.joint_angles[joint_index]], dtype=np.float32)
        obs = {
            'joint_angle': joint_angle,
            'position_error': self.position_error.astype(np.float32),
            'orientation_error': self.orientation_error.astype(np.float32),
        }
        return obs

    def get_current_pose(self):
        """
        Retrieves the current end-effector pose (position and orientation).

        Returns:
            tuple: A 3D position (np.array) and a quaternion orientation (np.array).
        """
        end_effector_state = p.getLinkState(self.robot_id, self.joint_indices[-1])
        current_position = np.array(end_effector_state[4])
        current_orientation = np.array(end_effector_state[5])  # Quaternion
        return current_position, current_orientation

    def compute_orientation_difference(self, current_orientation, target_orientation):
        """
        Computes the difference between the current and target orientations in terms of Euler angles.

        Args:
            current_orientation (np.array): The current orientation quaternion.
            target_orientation (np.array): The target orientation quaternion.

        Returns:
            np.array: The difference between the orientations in Euler angles.
        """
        current_inv = [current_orientation[0], -current_orientation[1], -current_orientation[2], -current_orientation[3]]
        diff_quat = p.multiplyTransforms([0, 0, 0], current_inv, [0, 0, 0], target_orientation)[1]
        diff_euler = p.getEulerFromQuaternion(diff_quat)
        return np.array(diff_euler)

    def render(self, mode='human'):
        """ PyBullet automatically handles rendering in GUI mode. """
        pass

    def close(self):
        """ Closes the environment and disconnects from the PyBullet simulation. """
        if p.isConnected():
            p.disconnect()


    def safe_min(arr, default=np.inf):
        """
        Safely computes the minimum value of an array. If the array is empty or contains NaNs, returns a default value.
        """
        if len(arr) == 0:
            return default
        arr = np.array(arr)
        if np.isnan(arr).any():
            return default
        return np.min(arr)

    def is_success(self, joint_errors):
        """
        Check if the current state is successful based on the mean of joint errors and variable success threshold.
        """
        # Calculate the mean of joint errors
        mean_joint_error = np.mean(joint_errors)

        # Determine if the mean joint error is below the current success threshold
        mean_position_error = np.nanmin(self.position_error)
        mean_orientation_error = np.nanmin(self.orientation_error)
        success = mean_joint_error < self.success_threshold and mean_position_error < self.success_threshold and mean_orientation_error < self.success_threshold

        # Log the success criteria
        #print(f"Mean Joint Error: {mean_joint_error:.6f} (Threshold: {self.success_threshold:.4f}), Success: {success}")

        return success

    def is_agent_success(self, joint_index, joint_errors):
        """
        Determines if an agent (joint) is successful based on its joint error.

        Args:
            joint_index (int): Index of the joint (agent).
            joint_errors (np.array): Array of joint errors for each joint.

        Returns:
            bool: True if the joint's error is below the error threshold, False otherwise.
        """
        # Define success criteria for joint error
        error_threshold = 0.01  # Define a threshold for joint error

        # Get the specific joint error
        joint_error = joint_errors[joint_index]
                # Determine if the mean joint error is below the current success threshold
        mean_position_error = np.nanmin(self.position_error)
        mean_orientation_error = np.nanmin(self.orientation_error)
        joint_success = joint_error < self.success_threshold and mean_position_error < self.success_threshold and mean_orientation_error < self.success_threshold
        # Determine success based on joint error
        # joint_success = np.nanmean(joint_error) < error_threshold

        # Log details about the success criterion
        print(f"Joint {joint_index} - Joint Error: {joint_error:.6f} (Threshold: {error_threshold}), Success: {joint_success}")

        return joint_success
    
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

    def get_end_effector_pose(self):
        """
        Retrieve the current position and orientation of the end-effector.
        """
        end_effector_state = p.getLinkState(self.robot_id, self.joint_indices[-1])
        current_position = np.array(end_effector_state[4])
        current_orientation = np.array(end_effector_state[5])
        return current_position, current_orientation

    def get_target_pose(self):
        """
        Retrieve the target position and orientation for the end-effector.
        """
        return self.target_position, self.target_orientation
    
    def get_euler_angles(self):
        """
        Retrieves the Euler angles of the end-effector's current orientation.
        
        Returns:
            np.array: The Euler angles (roll, pitch, yaw) of the end-effector.
        """
        _, orientation_quat = self.get_end_effector_pose()
        euler_angles = np.array(p.getEulerFromQuaternion(orientation_quat))
        return euler_angles

    def set_joint_positions(self, joint_positions: List[float]):
        """
        Set the robot's joint positions to the specified values.

        Args:
            joint_positions (List[float]): List of joint angles to set.
        """
        assert len(joint_positions) == self.num_joints, \
            f"Expected {self.num_joints} joint positions, but got {len(joint_positions)}"

        for i, joint_index in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_index, targetValue=joint_positions[i])

    def reset_to_target(self, target_position: List[float], target_orientation: List[float]):
        """
        Reset the robot to a configuration that corresponds to the given target position and orientation.
        
        Args:
            target_position (List[float]): Target position [x, y, z] for the end-effector.
            target_orientation (List[float]): Target orientation [x, y, z, w] as a quaternion for the end-effector.
        """
        # Compute the joint angles using inverse kinematics
        target_joint_angles = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.joint_indices[-1],
            targetPosition=target_position,
            targetOrientation=target_orientation,
            lowerLimits=[limit[0] for limit in self.joint_limits],
            upperLimits=[limit[1] for limit in self.joint_limits],
            jointRanges=[limit[1] - limit[0] for limit in self.joint_limits],
            restPoses=[0.0] * self.num_joints
        )
        
        # Set the computed joint angles
        for i, joint_index in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_index, targetValue=target_joint_angles[i])

        # Update the internal state of the environment, if necessary
        self.target_position = target_position
        self.target_orientation = target_orientation
    def reset_to_joint_positions(self, joint_positions):
        """
        Resets the robot to the specified joint positions.

        Args:
            joint_positions (list or np.ndarray): Target joint angles for resetting.
        """
        try:
            if len(joint_positions) != self.num_joints:
                raise ValueError(f"Expected {self.num_joints} joint positions, but got {len(joint_positions)}")

            for i, joint_index in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, joint_index, joint_positions[i])

            # Update the simulation to reflect the new joint positions
            p.stepSimulation()

        except Exception as e:
            raise RuntimeError(f"Error resetting joint positions: {e}")

