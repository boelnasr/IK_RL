import gym
import pybullet as p
import pybullet_data
from typing import Dict, List, Tuple    
import numpy as np
import logging
from config import config
from .exploration import ExplorationModule
from .curriculum import CurriculumManager
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
        self.robot_name = robot_name
        self.episode_number = 0
        self.total_episodes = config.get('num_episodes', 2000)
        # Support both parameter names for backwards compatibility
        self.max_episode_steps = config.get('max_episode_steps',
                                            config.get('max_steps_per_episode', 5000))

        # Validation: ensure episodes can actually run
        if self.max_episode_steps < 10:
            logging.warning(
                f"max_episode_steps is very low ({self.max_episode_steps})! "
                f"Episodes may terminate too early. Setting to minimum of 100."
            )
            self.max_episode_steps = 100

        logging.info(f"Environment initialized: max_episode_steps={self.max_episode_steps}")
        # FIXED: Use constant threshold of 0.01 (no decay)
        self.max_success_threshold = config.get('max_success_threshold', 0.01)
        self.min_success_threshold = config.get('min_success_threshold', 0.01)

        # Fixed threshold (no decay)
        self.success_threshold = 0.01
        self.curriculum_manager = CurriculumManager(
            initial_difficulty=0.5,             # IMPROVED: Start easier (was 1.0)
            max_difficulty=2.0,                 # Reasonable ceiling
            min_difficulty=0.3,                 # IMPROVED: Lower floor for struggling agents
            success_threshold=0.25,             # IMPROVED: Easier progression (was 0.35)
            window_size=100,                    # Long window for stable estimates
            difficulty_increment=0.02           # IMPROVED: Slower increments (was 0.03)
        )
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
        self.link_lengths = []
        for i in range(p.getNumJoints(self.robot_id)):
            # Get joint info
            joint_info = p.getJointInfo(self.robot_id, i)
            # Get link state
            link_state = p.getLinkState(self.robot_id, i)
            
            # Calculate link length from joint positions
            if i > 0:  # Skip the base link
                prev_joint_pos = p.getLinkState(self.robot_id, i-1)[0]
                current_joint_pos = link_state[0]
                link_length = np.linalg.norm(
                    np.array(current_joint_pos) - np.array(prev_joint_pos)
                )
                self.link_lengths.append(link_length)
        
        # Ensure we have at least one link length
        if not self.link_lengths:
            self.link_lengths = [0.2, 0.3, 0.2]  # Default values
            print("Warning: Using default link lengths")
        
        print(f"Calculated link lengths: {self.link_lengths}")
        
        # Calculate total reach for workspace validation
        self.total_reach = sum(self.link_lengths)
        self.min_reach = self.total_reach * 0.1  # 10% of total reach
        self.max_reach = self.total_reach * 0.9  # 90% of total reach

        # Define workspace bounds based on total reach
        self.workspace_bounds = {
            'x': (self.min_reach, self.max_reach),
            'y': (-self.max_reach/2, self.max_reach/2),
            'z': (0.1, self.max_reach * 0.8)
        }
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
        # Position error target is much tighter (0.1 mm) while orientation/joint thresholds stay at 0.01
        self.position_threshold = config.get('position_threshold', 1e-4)
        self.orientation_threshold = config.get('orientation_threshold', 0.01)
        self.success_grace_steps = config.get('success_grace_steps',10)
        self._success_grace_counter = None

        # Cache for expensive Jacobian computations
        self._jacobian_cache = {
            'angles': None,
            'linear': None,
            'angular': None
        }
        self.jacobian_update_tolerance = config.get('jacobian_update_tolerance', 1e-3)

        print(f"Number of joints in the robot: {self.num_joints}")



    def update_success_threshold(self):
        """
        Linearly decays the success-, position- and orientation-thresholds from
        `max_*_threshold` to `min_*_threshold` across roughly the first 90 % of the
        episodes. After that point the thresholds stay constant.

        ───────────────────────── decay ─────────────────────────── hold ───
        ep = 0                           0.9·total_episodes                  total
        succ_thresh = max  ────────────────▶  min  ──────────────────────────▶ min
        """
        # IMPROVED: Decay threshold from max (relaxed) to min (precise) over 90% of training
        progress = min(1.0, self.episode_number / (self.total_episodes * 0.9))
        self.success_threshold = (
            self.max_success_threshold * (1.0 - progress) +
            self.min_success_threshold * progress
        )

        return self.success_threshold


    def get_reward_logging_snapshot(self) -> dict:
        """
        Collect the environment-side parameters that influence reward shaping.
        """
        curriculum = {
            'max_difficulty': float(self.curriculum_manager.max_difficulty),
            'min_difficulty': float(self.curriculum_manager.min_difficulty),
            'success_threshold': float(self.curriculum_manager.success_threshold),
            'window_size': int(self.curriculum_manager.window_size),
            'difficulty_increment': float(self.curriculum_manager.difficulty_increment),
            'decay_rate': float(self.curriculum_manager.decay_rate),
        }
        return {
            'robot_name': self.robot_name,
            'num_joints': int(self.num_joints),
            'total_episodes': int(self.total_episodes),
            'max_episode_steps': int(self.max_episode_steps),
            'min_success_threshold': float(self.min_success_threshold),
            'max_success_threshold': float(self.max_success_threshold),
            'current_success_threshold': float(self.success_threshold),
            'position_threshold': float(self.position_threshold),
            'orientation_threshold': float(self.orientation_threshold),
            'jacobian_update_tolerance': float(self.jacobian_update_tolerance),
            'curriculum': curriculum,
        }


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
    
    def is_position_reachable(self, position):
        """
        Check if a position is reachable by the robot.
        
        Args:
            position: numpy array [x, y, z] of the target position
        
        Returns:
            bool: True if position is reachable, False otherwise
        """
        # Calculate distance from base to position
        distance_from_base = np.linalg.norm(position[:2])  # XY distance from base
        
        # Check if position is within workspace boundaries
        is_within_bounds = (
            self.workspace_bounds['x'][0] <= position[0] <= self.workspace_bounds['x'][1] and
            self.workspace_bounds['y'][0] <= position[1] <= self.workspace_bounds['y'][1] and
            self.workspace_bounds['z'][0] <= position[2] <= self.workspace_bounds['z'][1]
        )
        
        # Check if the point is within the reachable sphere
        is_within_reach = (
            self.min_reach <= distance_from_base <= self.max_reach
        )
        
        return is_within_bounds and is_within_reach

    def sample_valid_target_position(self, max_attempts=100):
        """
        Sample a valid target position within the robot's workspace.
        
        Args:
            max_attempts: Maximum number of sampling attempts
        
        Returns:
            numpy array: Valid target position or None if no valid position found
        """
        for _ in range(max_attempts):
            # Sample random position within bounds
            position = np.array([
                np.random.uniform(self.workspace_bounds['x'][0], self.workspace_bounds['x'][1]),
                np.random.uniform(self.workspace_bounds['y'][0], self.workspace_bounds['y'][1]),
                np.random.uniform(self.workspace_bounds['z'][0], self.workspace_bounds['z'][1])
            ])
            
            if self.is_position_reachable(position):
                return position
                
        print("Warning: Could not find valid target position")
        # Fallback to a known reachable position
        return np.array([
            (self.workspace_bounds['x'][0] + self.workspace_bounds['x'][1]) / 2,
            0,  # Middle of Y range
            (self.workspace_bounds['z'][0] + self.workspace_bounds['z'][1]) / 2
        ])

    def validate_orientation(self, position, orientation):
        """
        Validate if an orientation is feasible at a given position.
        
        Args:
            position: numpy array [x, y, z] of the target position
            orientation: numpy array [qx, qy, qz, qw] quaternion
        
        Returns:
            bool: True if orientation is feasible, False otherwise
        """
        # Convert quaternion to euler angles for easier checking
        euler = p.getEulerFromQuaternion(orientation)
        
        # Define reasonable angle limits for each axis
        angle_limits = {
            'roll': (-np.pi, np.pi),
            'pitch': (-np.pi/2, np.pi/2),  # More restrictive pitch
            'yaw': (-np.pi, np.pi)
        }
        
        # Check if angles are within limits
        if not (angle_limits['roll'][0] <= euler[0] <= angle_limits['roll'][1] and
                angle_limits['pitch'][0] <= euler[1] <= angle_limits['pitch'][1] and
                angle_limits['yaw'][0] <= euler[2] <= angle_limits['yaw'][1]):
            return False
        
        return True

    def sample_valid_orientation(self, position, max_attempts=50):
        """
        Sample a valid orientation for a given position.
        
        Args:
            position: numpy array [x, y, z] of the target position
            max_attempts: Maximum number of sampling attempts
        
        Returns:
            numpy array: Valid quaternion orientation or None if no valid orientation found
        """
        for _ in range(max_attempts):
            # Sample random euler angles
            euler = np.random.uniform(-np.pi, np.pi, 3)
            # Convert to quaternion
            quaternion = np.array(p.getQuaternionFromEuler(euler))
            
            if self.validate_orientation(position, quaternion):
                return quaternion
                
        return None  # No valid orientation found

    def reset(self, difficulties=1.0):
        """
        Reset environment with workspace validation.
        """
        self.success_threshold = self.update_success_threshold()
        self.current_difficulty = difficulties

        # Reset joint angles to random values within limits
        self.joint_angles = np.array([
            np.random.uniform(limit[0], limit[1]) for limit in self.joint_limits
        ])
        for i, angle in zip(self.joint_indices, self.joint_angles):
            p.resetJointState(self.robot_id, i, angle)

        # Track previous joint angles
        self.previous_joint_angles = np.copy(self.joint_angles)

        # Sample valid target position and orientation
        max_position_attempts = 100
        max_orientation_attempts = 50
        
        valid_position = self.sample_valid_target_position(max_position_attempts)
        if valid_position is None:
            raise RuntimeError("Could not find valid target position after maximum attempts")
        
        valid_orientation = self.sample_valid_orientation(valid_position, max_orientation_attempts)
        if valid_orientation is None:
            raise RuntimeError("Could not find valid target orientation after maximum attempts")

        # Set target pose
        self.target_position = valid_position
        self.target_orientation = valid_orientation

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
        self._success_grace_counter = None

        # Reset Jacobian cache
        self._jacobian_cache['angles'] = None
        self._jacobian_cache['linear'] = None
        self._jacobian_cache['angular'] = None

        # Initialize prev_achieved_goal for HER (same as current at reset)
        self._prev_achieved_goal = np.concatenate(
            [self.current_position.copy(), self.current_orientation.copy()]
        )

        # Calculate initial errors
        self.position_error = self.current_position - self.target_position
        self.orientation_error = self.compute_orientation_difference(
            self.current_orientation, self.target_orientation
        )

        return self.get_all_agent_observations()


    def step(self, actions):
        """
        Executes one environment step with proper joint error calculation based on 
        the difference between predicted and target joint angles.
        """
        try:
            # Store target angles from actions - these represent our intended joint positions
            target_joint_angles = np.array([
                np.clip(float(action), self.joint_limits[i][0], self.joint_limits[i][1])
                for i, action in enumerate(actions)
            ])

            # Apply actions to move the robot
            for i, target_angle in enumerate(target_joint_angles):
                p.setJointMotorControl2(
                    self.robot_id, 
                    self.joint_indices[i], 
                    p.POSITION_CONTROL, 
                    targetPosition=target_angle
                )
            p.stepSimulation()
            self.current_step += 1

            # Get the actual achieved joint angles after movement
            self.joint_angles = np.array([
                p.getJointState(self.robot_id, i)[0] for i in self.joint_indices
            ], dtype=np.float32)

            # Calculate joint errors as the difference between target and achieved angles
            # This tells us how far each joint is from where we wanted it to be
            self.joint_errors = np.abs(self.joint_angles - target_joint_angles)

            joint_movements = None
            if self.previous_joint_angles is not None:
                joint_movements = np.abs(self.joint_angles - self.previous_joint_angles)
            else:
                joint_movements = np.zeros_like(self.joint_errors)

            # Store previous achieved goal for HER (before updating current pose)
            self._prev_achieved_goal = np.concatenate(
                [self.current_position.copy(), self.current_orientation.copy()]
            ) if hasattr(self, 'current_position') else None

            # Update end effector state and compute task space errors
            self.current_position, self.current_orientation = self.get_end_effector_pose()
            self.current_quaternion = np.array(self.current_orientation, dtype=np.float32)

            # Calculate position and orientation errors in task space
            self.position_error = self.current_position - self.target_position
            self.orientation_error = self.compute_orientation_difference(
                self.current_orientation, self.target_orientation
            )
            self.current_distance = compute_overall_distance(
                self.current_position, self.target_position,
                self.current_orientation, self.target_orientation
            )

            # Calculate movement weights using Jacobians (reuse cache when motion is small)
            recompute_jacobian = True
            cached_angles = self._jacobian_cache['angles']
            if cached_angles is not None:
                angle_delta = np.linalg.norm(self.joint_angles - cached_angles, ord=2)
                if angle_delta < self.jacobian_update_tolerance:
                    recompute_jacobian = False
            if recompute_jacobian:
                jacobian_linear = compute_jacobian_linear(self.robot_id, self.joint_indices, self.joint_angles)
                jacobian_angular = compute_jacobian_angular(self.robot_id, self.joint_indices, self.joint_angles)
                self.linear_weights, self.angular_weights = assign_joint_weights(jacobian_linear, jacobian_angular)
                self._jacobian_cache['angles'] = self.joint_angles.copy()
                self._jacobian_cache['linear'] = np.array(self.linear_weights, dtype=np.float32)
                self._jacobian_cache['angular'] = np.array(self.angular_weights, dtype=np.float32)
            else:
                jacobian_linear = self._jacobian_cache['linear']
                jacobian_angular = self._jacobian_cache['angular']
                self.linear_weights = jacobian_linear.tolist()
                self.angular_weights = jacobian_angular.tolist()

            # Evaluate success and update curriculum for each joint
            agent_successes = []
            step_difficulties = []

            for i in range(self.num_joints):
                # Determine success based on how close we got to the target angle
                agent_success = self.is_agent_success(i, self.joint_errors)
                agent_successes.append(agent_success)

                # Update the curriculum based on achievement of target angles
                self.curriculum_manager.log_agent_success(i, agent_success)
                self.curriculum_manager.update_agent_difficulty(i)
                current_difficulty = self.curriculum_manager.get_agent_difficulty(i)
                step_difficulties.append(current_difficulty)

                # print(f"Joint {i}: Target={target_joint_angles[i]:.4f}, "
                #     f"Achieved={self.joint_angles[i]:.4f}, "
                #     f"Error={self.joint_errors[i]:.4f}, "
                #     f"Success={agent_success}, "
                #     f"Difficulty={current_difficulty:.3f}")

            # Track difficulty history
            if not hasattr(self, 'difficulties_history'):
                self.difficulties_history = [[] for _ in range(self.num_joints)]
            for i in range(self.num_joints):
                self.difficulties_history[i].append(step_difficulties[i])

            # Compute rewards with the corrected joint errors
            try:
                #  In InverseKinematicsEnv.step()  ── call site change only
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
                        # NEW ↓↓↓
                        position_error=float(np.linalg.norm(self.position_error)),
                        orientation_error=float(np.linalg.norm(self.orientation_error)),
                        position_threshold=float(self.position_threshold),
                        orientation_threshold=float(self.orientation_threshold),
                        joint_threshold=float(self.success_threshold),          # keep legacy link-up
                        joint_movements=joint_movements.tolist()
                )




            except Exception as e:
                logging.error(f"Reward computation error: {e}")
                rewards = np.array([-0.1] * self.num_joints)  # Smaller penalty for stability
                individual_rewards = [-0.1] * self.num_joints
                stay_rewards = [0.0] * self.num_joints
                overall_success = False

            # Track post-success grace window
            if overall_success:
                if self._success_grace_counter is None:
                    self._success_grace_counter = 0
                else:
                    self._success_grace_counter += 1
            else:
                self._success_grace_counter = None

            grace_done = (
                self._success_grace_counter is not None and
                self._success_grace_counter >= self.success_grace_steps
            )

            # Calculate success metrics
            success_rate = sum(agent_successes) / self.num_joints
            done = self.current_step >= self.max_episode_steps or grace_done

            # Debug early termination
            if done and self.current_step < 10:
                logging.warning(
                    f"Early episode termination at step {self.current_step}! "
                    f"Reason: {'grace_done' if grace_done else 'max_steps'}, "
                    f"grace_counter={self._success_grace_counter}, "
                    f"overall_success={overall_success}, "
                    f"max_episode_steps={self.max_episode_steps}"
                )

            position_error_norm = float(np.linalg.norm(self.position_error))
            orientation_error_norm = float(np.linalg.norm(self.orientation_error))
            mean_joint_error = float(np.mean(self.joint_errors))

            # Create streamlined info dictionary (only fields consumed downstream / HER)
            info = {
                'success_per_joint': agent_successes,
                'overall_success_rate': success_rate,
                'current_distance': float(self.current_distance),
                'best_distance': float(self.previous_best_distance),
                'position_error': position_error_norm,
                'orientation_error': orientation_error_norm,
                'mean_joint_error': mean_joint_error,
                'step': self.current_step,
                'agent_difficulties': step_difficulties,
                'joint_errors': self.joint_errors.tolist(),
                'stay_rewards': stay_rewards,
                'target_angles': target_joint_angles.tolist(),
                'episode_success': bool(overall_success),
                'grace_active': self._success_grace_counter is not None,
                'grace_steps': int(self._success_grace_counter) if self._success_grace_counter is not None else 0,
                'desired_goal': np.concatenate(
                    [self.target_position, self.target_orientation]
                ).tolist(),
                'achieved_goal': np.concatenate(
                    [self.current_position, self.current_orientation]
                ).tolist(),
                'prev_achieved_goal': self._prev_achieved_goal.tolist() if self._prev_achieved_goal is not None else np.concatenate(
                    [self.current_position, self.current_orientation]
                ).tolist(),
            }

            # Throttle verbose logging to avoid I/O bottlenecks.
            if self.current_step % 100 == 0 or done:
                logging.debug(
                    "Step %d | success=%.3f | mean_joint_error=%.4f | dist=%.4f | difficulties=%s",
                    self.current_step,
                    success_rate,
                    mean_joint_error,
                    self.current_distance,
                    step_difficulties
                )

            # Store current angles for next step
            self.previous_joint_angles = np.copy(self.joint_angles)
            return self.get_all_agent_observations(), rewards.tolist(), done, info

        except Exception as e:
            import traceback
            logging.error(f"Critical error in step: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            logging.error(
                f"Episode will terminate at step {self.current_step}. "
                f"Target: {self.target_position}, Current: {self.current_position}"
            )
            return (
                self.get_all_agent_observations(),
                [-0.1] * len(self.joint_indices),  # Smaller penalty for stability
                True,  # Terminate episode on critical errors
                {
                    'error': str(e),
                    'agent_difficulties': [0.0] * self.num_joints,
                    'step': self.current_step if hasattr(self, 'current_step') else 0,
                    'rewards_valid': False
                }
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
        Return the wrapped Euler-angle delta between current and target orientations.
        """
        current_orientation = np.asarray(current_orientation, dtype=np.float64)
        target_orientation = np.asarray(target_orientation, dtype=np.float64)

        curr_norm = np.linalg.norm(current_orientation)
        target_norm = np.linalg.norm(target_orientation)
        if curr_norm < 1e-8 or target_norm < 1e-8:
            return np.zeros(3, dtype=np.float32)

        curr_normed = current_orientation / curr_norm
        target_normed = target_orientation / target_norm

        # PyBullet quaternions are [x, y, z, w]; conjugate flips the vector part only.
        curr_conjugate = np.array(
            [-curr_normed[0], -curr_normed[1], -curr_normed[2], curr_normed[3]],
            dtype=np.float64
        )

        _, relative_quat = p.multiplyTransforms(
            [0, 0, 0],
            curr_conjugate.tolist(),
            [0, 0, 0],
            target_normed.tolist()
        )

        euler_delta = np.array(p.getEulerFromQuaternion(relative_quat), dtype=np.float64)
        # Wrap to [-pi, pi] to avoid spurious large angles from numerical drift.
        euler_delta = (euler_delta + np.pi) % (2.0 * np.pi) - np.pi
        return euler_delta.astype(np.float32)

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

        # Evaluate against per-metric thresholds (position in meters, orientation in radians)
        position_error_norm = float(np.linalg.norm(self.position_error))
        orientation_error_norm = float(np.linalg.norm(self.orientation_error))

        success = (
            mean_joint_error < self.success_threshold
            and position_error_norm < self.position_threshold
            and orientation_error_norm < self.orientation_threshold
        )

        # Log the success criteria
        #print(f"Mean Joint Error: {mean_joint_error:.6f} (Threshold: {self.success_threshold:.4f}), Success: {success}")

        return success

    def is_agent_success(self, joint_index, joint_errors):
        """
        FIXED: Determines if an agent (joint) is successful based ONLY on its joint error.

        This matches the reward function fix - each joint is evaluated independently
        for clearer credit assignment and better learning.

        Args:
            joint_index (int): Index of the joint (agent).
            joint_errors (np.array): Array of joint errors for each joint.

        Returns:
            bool: True if the joint's error is below the error threshold, False otherwise.
        """
        # Get the specific joint error
        joint_error = joint_errors[joint_index]

        # FIXED: Use only joint-specific error (removed global position/orientation dependencies)
        # This gives each joint clear, independent feedback
        joint_success = joint_error < self.success_threshold

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
