# Import the environment class for running the simulation
from .environment import InverseKinematicsEnv

# Import the MAPPO agent class for training the position and orientation controllers
from .mappo import MAPPOAgent

# Import the reward functions for computing position and orientation rewards
from .reward_function import (
    compute_position_error,         # To compute position errors between current and target positions
    compute_quaternion_distance,    # To compute quaternion distance between current and target orientations
    compute_overall_distance,       # To compute the combined distance metric
    compute_reward,                 # To compute the overall reward based on the distance metrics
    compute_jacobian_linear,        # To compute the linear Jacobian matrix for the end-effector
    compute_jacobian_angular        # To compute the angular Jacobian matrix for the end-effector
)



# Import training metrics class
from .training_metrics import TrainingMetrics  # Ensure the path is correct

# Import utility functions
from .utils import (
    rotation_matrix_to_euler_angles,
    euler_angles_to_rotation_matrix,
    skew_symmetric,
    matrix_exp6,
    matrix_log6,
    vec_to_se3,
    extract_rotation_translation
)

# Define the __all__ variable to specify what is publicly available when this module is imported
__all__ = [
    "InverseKinematicsEnv", 
    "MAPPOAgent", 
    "compute_position_reward", 
    "compute_orientation_reward",
    "compute_joint_change_penalty",  # Ensure this is included
    "TrainingMetrics",  # Add this to make training metrics publicly available
    "rotation_matrix_to_euler_angles", 
    "euler_angles_to_rotation_matrix",
    "skew_symmetric",
    "matrix_exp6",
    "matrix_log6",
    "vec_to_se3",
    "extract_rotation_translation"
]
