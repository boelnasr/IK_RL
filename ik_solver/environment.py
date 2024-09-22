import gym
import pybullet as p
import pybullet_data
import numpy as np
import logging

from .reward_function import (
    compute_position_error,
    compute_cosine_distance,
    compute_overall_distance,
    compute_reward
)
# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class InverseKinematicsEnv(gym.Env):
    """
    Custom environment for inverse kinematics control of a robotic arm in PyBullet.
    Each joint is controlled by an individual agent.
    """
    def __init__(self, robot_name="kuka_iiwa", sim_timestep=1./240., max_episode_steps=1000):
        """
        Initializes the environment and PyBullet simulation.

        Args:
            robot_name (str): The robot's name to load (e.g., "kuka_iiwa").
            sim_timestep (float): Simulation time step for PyBullet.
            max_episode_steps (int): Maximum number of steps per episode.
        """
        super(InverseKinematicsEnv, self).__init__()

        # PyBullet setup
        self.physics_client = p.connect(p.GUI)  # Use p.DIRECT for headless simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # PyBullet data path
        p.setGravity(0, 0, -9.8)  # Set gravity for the simulation
        self.sim_timestep = sim_timestep
        p.setTimeStep(self.sim_timestep)

        # Load the selected robot
        if robot_name == "kuka_iiwa":
            self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)  # Assign robot_id here
        else:
            raise ValueError(f"Robot {robot_name} is not available in PyBullet data.")

        # Initialize joint information
        self.num_joints = p.getNumJoints(self.robot_id)  # Ensure robot_id is assigned before this line
        self.joint_indices = list(range(self.num_joints))
        print(f"Number of joints in the robot: {self.num_joints}")

        # Retrieve joint limits for all joints
        self.joint_limits = []
        for i in self.joint_indices:
            joint_info = p.getJointInfo(self.robot_id, i)
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            self.joint_limits.append((lower_limit, upper_limit))

        # Track previous joint angles
        self.previous_joint_angles = None  # Will be initialized in reset

        # Define action and observation spaces per agent
        self.action_spaces = []
        self.observation_spaces = []
        for i in range(self.num_joints):
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
        self.max_episode_steps = max_episode_steps
        self.position_threshold = 0.01  # Define appropriate value
        self.orientation_threshold = 0.1  # Define appropriate value in radians
        # Set the success threshold for the distance
        self.success_threshold = 0.01  # Adjust this value as needed
        print(f"Number of joints in the robot: {self.num_joints}")


    def reset(self):
        """
        Resets the environment by setting the robot's joint angles to random values
        and generating a new random target pose for the end-effector.

        Returns:
            list: A list of observations for each agent.
        """
        # Reset joint angles to random values within joint limits
        self.joint_angles = np.array([
            np.random.uniform(limit[0], limit[1]) for limit in self.joint_limits
        ])
        for i, angle in zip(self.joint_indices, self.joint_angles):
            p.resetJointState(self.robot_id, i, angle)

        # Track the initial joint angles as the previous joint angles
        self.previous_joint_angles = np.copy(self.joint_angles)

        # Generate a new random target pose for the end-effector
        # Ensure the target is within reachable workspace
        self.target_position = np.random.uniform(
            [0.1, -0.5, 0.1], [0.5, 0.5, 0.5]
        )
        # Random target orientation as a quaternion
        random_euler = np.random.uniform(-np.pi, np.pi, 3)
        self.target_orientation = p.getQuaternionFromEuler(random_euler)

        # Optionally, visualize the target position in the simulation
        if hasattr(self, 'target_marker'):
            p.removeBody(self.target_marker)
        self.target_marker = p.loadURDF(
            "sphere_small.urdf",
            self.target_position,
            globalScaling=0.05,
            useFixedBase=True
        )

        # Reset step counter
        self.current_step = 0

        # Get the current end-effector pose
        self.current_position, self.current_orientation = self.get_current_pose()

        # Compute initial position and orientation errors
        self.position_error = self.current_position - self.target_position
        self.orientation_error = self.compute_orientation_difference(
            self.current_orientation, self.target_orientation
        )

        # Compute the initial overall distance
        self.begin_distance = compute_overall_distance(
            current_position=self.current_position,
            target_position=self.target_position,
            current_orientation=self.current_orientation,
            target_orientation=self.target_orientation
        )

        # Initialize prev_best with a large value or begin_distance
        self.prev_best = float('inf')  # or self.begin_distance

        # Return the initial observations for all agents
        return self.get_all_agent_observations()

    def step(self, actions):
        # Apply the actions and step the simulation
        for i, action in enumerate(actions):
            action = np.clip(action, self.joint_limits[i][0], self.joint_limits[i][1])
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=action)

        p.stepSimulation()

        # Update current step
        self.current_step += 1

        # Get the current state
        self.joint_angles = np.array([p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)])
        current_position, current_orientation = self.get_current_pose()
        self.position_error = current_position - self.target_position
        self.orientation_error = self.compute_orientation_difference(
            current_orientation, self.target_orientation
        )

        # Compute the overall distance
        euclidean_distance = compute_position_error(current_position, self.target_position)
        cosine_distance = compute_cosine_distance(current_orientation, self.target_orientation)
        b = cosine_distance - 1
        distance = euclidean_distance * cosine_distance + b

        # Compute the overall reward
        if distance > self.prev_best:
            overall_reward = self.prev_best - distance  # Punishment
        else:
            overall_reward = self.begin_distance - distance  # Reward
            self.prev_best = distance

        if distance <= self.success_threshold:
            overall_reward = 2.0 + (self.success_threshold - distance) * 1000
            success = True
        else:
            success = False

        # Compute the Jacobian matrices
        zero_vec = [0.0] * self.num_joints
        jacobian_linear, jacobian_angular = p.calculateJacobian(
            bodyUniqueId=self.robot_id,
            linkIndex=self.joint_indices[-1],
            localPosition=[0, 0, 0],
            objPositions=self.joint_angles.tolist(),
            objVelocities=zero_vec,
            objAccelerations=zero_vec
        )

        jacobian_linear = np.array(jacobian_linear)  # Shape: (3, num_joints)
        jacobian_angular = np.array(jacobian_angular)  # Shape: (3, num_joints)

        # Compute normalized errors
        delta_position = current_position - self.target_position
        delta_position_norm = delta_position / (np.linalg.norm(delta_position) + 1e-8)

        delta_orientation = self.orientation_error
        delta_orientation_norm = delta_orientation / (np.linalg.norm(delta_orientation) + 1e-8)

        # Compute gradients
        grad_position = jacobian_linear.T @ delta_position_norm  # Shape: (num_joints,)
        grad_orientation = jacobian_angular.T @ delta_orientation_norm  # Shape: (num_joints,)

        grad_total = grad_position + grad_orientation
        grad_norm = np.linalg.norm(grad_total) + 1e-8

        # Assign individual rewards and compute joint errors
        rewards = []
        joint_errors = []
        for i in range(self.num_joints):
            joint_contribution = -grad_total[i] / grad_norm
            joint_reward = overall_reward * joint_contribution
            rewards.append(joint_reward)

            # Compute joint error (absolute difference in joint angles)
            joint_error = abs(self.joint_angles[i] - self.previous_joint_angles[i])
            joint_errors.append(joint_error)

            # Log the error and reward for each joint
            logging.info(f"Step {self.current_step}, Joint {i} - Error: {joint_error:.6f}, Reward: {joint_reward:.6f}")

        # Update previous joint angles
        self.previous_joint_angles = np.copy(self.joint_angles)

        # Update the done flag
        done = self.current_step >= self.max_episode_steps or success

        # Get observations
        observations = self.get_all_agent_observations()

        # Store joint errors for access in the training loop
        self.joint_errors = joint_errors

        return observations, rewards, done, {}
    def step(self, actions):
        # Apply the actions and step the simulation
        for i, action in enumerate(actions):
            action = np.clip(action, self.joint_limits[i][0], self.joint_limits[i][1])
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=action)

        p.stepSimulation()

        # Update current step
        self.current_step += 1

        # Get the current state
        self.joint_angles = np.array([p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)])
        current_position, current_orientation = self.get_current_pose()
        self.position_error = current_position - self.target_position
        self.orientation_error = self.compute_orientation_difference(
            current_orientation, self.target_orientation
        )

        # Compute the overall distance
        euclidean_distance = compute_position_error(current_position, self.target_position)
        cosine_distance = compute_cosine_distance(current_orientation, self.target_orientation)
        b = cosine_distance - 1
        distance = euclidean_distance * cosine_distance + b

        # Compute the overall reward
        if distance > self.prev_best:
            overall_reward = self.prev_best - distance  # Punishment
        else:
            overall_reward = self.begin_distance - distance  # Reward
            self.prev_best = distance

        if distance <= self.success_threshold:
            overall_reward = 2.0 + (self.success_threshold - distance) * 5000
            success = True
        else:
            success = False

        # Compute the Jacobian matrices
        zero_vec = [0.0] * self.num_joints
        jacobian_linear, jacobian_angular = p.calculateJacobian(
            bodyUniqueId=self.robot_id,
            linkIndex=self.joint_indices[-1],
            localPosition=[0, 0, 0],
            objPositions=self.joint_angles.tolist(),
            objVelocities=zero_vec,
            objAccelerations=zero_vec
        )

        jacobian_linear = np.array(jacobian_linear)  # Shape: (3, num_joints)
        jacobian_angular = np.array(jacobian_angular)  # Shape: (3, num_joints)

        # Compute normalized errors
        delta_position = current_position - self.target_position
        delta_position_norm = delta_position / (np.linalg.norm(delta_position) + 1e-8)

        delta_orientation = self.orientation_error
        delta_orientation_norm = delta_orientation / (np.linalg.norm(delta_orientation) + 1e-8)

        # Compute gradients
        grad_position = jacobian_linear.T @ delta_position_norm  # Shape: (num_joints,)
        grad_orientation = jacobian_angular.T @ delta_orientation_norm  # Shape: (num_joints,)

        grad_total = grad_position + grad_orientation
        grad_norm = np.linalg.norm(grad_total) + 1e-8

        # Assign individual rewards and compute joint errors
        rewards = []
        joint_errors = []
        for i in range(self.num_joints):
            joint_contribution = -grad_total[i] / grad_norm
            joint_reward = overall_reward * joint_contribution
            rewards.append(joint_reward)

            # Compute joint error (absolute difference in joint angles)
            joint_error = abs(self.joint_angles[i] - self.previous_joint_angles[i])
            joint_errors.append(joint_error)

            # Log the error and reward for each joint
            logging.info(f"Step {self.current_step}, Joint {i} - Error: {joint_error:.6f}, Reward: {joint_reward:.6f}")

        # Update previous joint angles
        self.previous_joint_angles = np.copy(self.joint_angles)

        # Update the done flag
        done = self.current_step >= self.max_episode_steps or success

        # Get observations
        observations = self.get_all_agent_observations()

        # Store joint errors for access in the training loop
        self.joint_errors = joint_errors

        return observations, rewards, done, {}


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

    # Optionally, include the compute_orientation_error method if used elsewhere
    def compute_orientation_error(self, current_orientation, target_orientation):
        """
        Computes the angular distance between the current and target orientations using quaternions.

        Args:
            current_orientation (np.array): The current orientation quaternion.
            target_orientation (np.array): The target orientation quaternion.

        Returns:
            float: The angular difference (in radians) between the two orientations.
        """
        dot_product = np.abs(np.dot(current_orientation, target_orientation))
        angle = 2 * np.arccos(np.clip(dot_product, -1.0, 1.0))
        return angle

    def is_success(self, position_error, orientation_error):
        """
        Check if the current state is successful, based on position and orientation error thresholds.
        
        Args:
            position_error (float): The current position error.
            orientation_error (float): The current orientation error.

        Returns:
            bool: True if both position and orientation errors are below the success thresholds.
        """
        position_threshold = 0.01  # Define your threshold for position error
        orientation_threshold = 0.1  # Define your threshold for orientation error (in radians)

    # Compute the norm of the errors
        position_error_norm = np.linalg.norm(position_error)
        orientation_error_norm = np.linalg.norm(orientation_error)
        return position_error_norm < position_threshold and orientation_error_norm < orientation_threshold