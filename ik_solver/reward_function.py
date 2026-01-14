import pybullet as p
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


EPS = 1e-8            # single global epsilon

# Centralise reward-related constants so they can be re-used in reports.
REWARD_CONSTANTS = {
    "MAX_REWARD": 8.0,
    "MIN_REWARD": -2.0,
    "POSITION_SCALE": 1.2,
    "ORIENTATION_SCALE": 1.0,
    "PERFORMANCE_SCALE": 0.3,
    "IMPROVEMENT_SCALE": 0.5,
    "ERROR_PENALTY_SCALE": 0.5,
    "SUCCESS_BONUS_BASE": 2.5,  # REBALANCED: Increased from 1.0 (too weak) to 2.5 (strong signal without dominating)
    "STAY_REWARD_SCALE": 0.4,    # New: reward for holding a stable pose
    "STAY_THRESHOLD_RATIO": 0.5, # Movement ratio vs joint_threshold for full staying bonus
    "TEAM_BONUS_SCALE": 3.0,     # Cooperative bonus shared across joints
    "TEAM_ALIGNMENT_MIN": 0.5,   # Require reasonable pose alignment before granting team bonuses
    "POSITION_FAILURE_PENALTY": 2.0,
    "ORIENTATION_FAILURE_PENALTY": 3.0,
    "POSE_PENALTY_CAP": 10.0,
    "POSITION_RELAX_INIT": 10.0, # Start with looser shaping threshold (multiplier)
    "ORIENTATION_RELAX_INIT": 5.0,
    "RELAX_DECAY": 0.9,
    "RELAX_TARGET_SUCCESS": 0.17,
    "RELAX_MIN_FACTOR": 1.2,
    "RELAX_WINDOW": 60,
    "RELAX_EVAL_MIN_COUNT": 30,
}

REWARD_DEFAULT_ARGUMENTS = {
    "success_threshold": 0.01,
    "position_threshold": 0.02,
    "orientation_threshold": 0.01,  # TIGHTENED: 0.1 → 0.01 rad (5.73° → 0.57°)
    "joint_threshold": 0.01,         # TIGHTENED: 0.05 → 0.01 rad (2.86° → 0.57°)
    "ratio_threshold": 0.5,
    "time_penalty": -0.001,
    "smoothing_window": 10,
    "exploration_bonus": 0.02,
}

OVERALL_DISTANCE_WEIGHTS = {
    "position": 0.7,
    "orientation": 0.3,
}


def get_reward_parameters_snapshot():
    """
    Provide a serialisable snapshot of the reward shaping configuration.
    """
    return {
        "eps": EPS,
        "reward_constants": REWARD_CONSTANTS.copy(),
        "default_arguments": REWARD_DEFAULT_ARGUMENTS.copy(),
        "overall_distance_weights": OVERALL_DISTANCE_WEIGHTS.copy(),
    }


def wrap_angle_to_pi(angle):
    """
    Wrap angle to the range [-π, π].
    
    Args:
        angle (float or np.array): Angle(s) to wrap
        
    Returns:
        float or np.array: Wrapped angle(s) in [-π, π]
    """
    # Use modulo operation and adjust to [-π, π]
    wrapped = np.fmod(angle + np.pi, 2 * np.pi) - np.pi
    
    # Handle the edge case where fmod might return -π instead of π
    wrapped = np.where(wrapped == -np.pi, np.pi, wrapped)
    
    return wrapped


def compute_position_error(current_position, target_position):
    """
    Enhanced position error calculation with robust handling.
    
    Args:
        current_position (np.array): Current position [x, y, z]
        target_position (np.array): Target position [x, y, z]
        
    Returns:
        float: Euclidean distance between positions
    """
    try:
        current_position = np.array(current_position, dtype=np.float64)
        target_position = np.array(target_position, dtype=np.float64)
        error = np.linalg.norm(current_position - target_position)
        return max(float(error), 1e-8)
    except Exception as e:
        logging.warning(f"Error in position calculation: {e}")
        return 1e-3

def compute_quaternion_distance(q1, q2):
    """
    Return unsigned axis-angle distance in radians ∈ [0, π].
    API and units stay the same – you still call it exactly the same way.
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    n1 = np.linalg.norm(q1);  n2 = np.linalg.norm(q2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0

    q1 /= n1;  q2 /= n2
    dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)   # abs() handles double cover
    # small-angle safeguard
    if dot > 0.999999:
        return 0.0
    return 2.0 * np.arccos(dot)                      # ∈ (0, π]


def compute_orientation_error_euler(current_orientation, target_orientation):
    """
    Compute orientation error in Euler angles, properly wrapped to [-π, π].
    
    Args:
        current_orientation (np.array): Current quaternion [x, y, z, w]
        target_orientation (np.array): Target quaternion [x, y, z, w]
        
    Returns:
        np.array: Orientation error for each axis (roll, pitch, yaw) in [-π, π]
    """
    try:
        # Convert quaternions to Euler angles
        current_euler = np.array(p.getEulerFromQuaternion(current_orientation))
        target_euler = np.array(p.getEulerFromQuaternion(target_orientation))
        
        # Compute angular differences
        euler_diff = current_euler - target_euler
        
        # Wrap each angle component to [-π, π]
        wrapped_diff = wrap_angle_to_pi(euler_diff)
        
        return wrapped_diff
        
    except Exception as e:
        logging.warning(f"Error in Euler orientation calculation: {e}")
        return np.zeros(3)


def compute_joint_angle_error(current_angle, target_angle):
    """
    Compute joint angle error properly wrapped to [-π, π].
    
    Args:
        current_angle (float): Current joint angle
        target_angle (float): Target joint angle
        
    Returns:
        float: Joint angle error in [-π, π]
    """
    try:
        # Compute raw difference
        error = current_angle - target_angle
        
        # Wrap to [-π, π] to find equivalent angle
        wrapped_error = wrap_angle_to_pi(error)
        
        return float(wrapped_error)
        
    except Exception as e:
        logging.warning(f"Error in joint angle calculation: {e}")
        return 0.0


def compute_overall_distance(current_position, target_position, current_orientation, target_orientation):
    """
    Weighted combination of position and orientation errors with proper angle wrapping.
    
    Args:
        current_position (np.array): Current position [x, y, z]
        target_position (np.array): Target position [x, y, z]
        current_orientation (np.array): Current quaternion [x, y, z, w]
        target_orientation (np.array): Target quaternion [x, y, z, w]
        
    Returns:
        float: Combined distance metric
    """
    try:
        position_error = compute_position_error(current_position, target_position)
        
        # Use the properly wrapped quaternion distance
        orientation_error = compute_quaternion_distance(current_orientation, target_orientation)
        
        # For overall distance, we want the magnitude, so take absolute value
        orientation_error = abs(orientation_error)
        
        # Weight position more heavily than orientation for most IK tasks
        overall_distance = (
            OVERALL_DISTANCE_WEIGHTS["position"] * position_error +
            OVERALL_DISTANCE_WEIGHTS["orientation"] * orientation_error
        )
        return max(float(overall_distance), 1e-8)
        
    except Exception as e:
        logging.warning(f"Error in overall distance calculation: {e}")
        return 1.0

# --------------------------------------------------------------------------- #
#  IMPROVED compute_reward() with better success detection and late-stage boost #
# --------------------------------------------------------------------------- #
import time
def compute_reward(
    distance: float,
    begin_distance: float,
    prev_best: float,
    current_orientation: np.ndarray,
    target_orientation: np.ndarray,
    joint_errors: list,
    linear_weights: list,
    angular_weights: list,
    difficulties: list,
    episode_number: int,
    total_episodes: int,
    *,
    # NEW: Direct error inputs from environment
    position_error: Optional[float] = None,
    orientation_error: Optional[float] = None,
    success_threshold: float = 0.01,            # Dynamic success threshold from environment
    position_threshold: float = 0.02,           # Position error threshold from environment
    orientation_threshold: float = 0.01,        # TIGHTENED: Orientation error threshold (0.57°)
    joint_threshold: float = 0.01,              # TIGHTENED: Joint error threshold (0.57°)
    ratio_threshold: float = 0.5,               # Ratio of joints that need to succeed
    time_penalty: float = -0.001,
    smoothing_window: int = 10,
    exploration_bonus: float = 0.02,
    joint_movements: Optional[list] = None,
    done: bool = False                          # Episode completion flag
) -> tuple:
    """
    ENHANCED: Reward function with direct position/orientation errors and late-stage amplification.
    
    Args:
        distance: Current overall distance to target
        begin_distance: Initial distance at episode start
        prev_best: Best distance achieved so far
        current_orientation: Current end-effector orientation quaternion
        target_orientation: Target orientation quaternion
        joint_errors: List of individual joint errors
        linear_weights: Jacobian-based linear movement weights
        angular_weights: Jacobian-based angular movement weights
        difficulties: Per-agent difficulty levels
        episode_number: Current episode number
        total_episodes: Total episodes planned
        position_error: Direct position error from environment (NEW)
        orientation_error: Direct orientation error from environment (NEW)
        success_threshold: Dynamic success threshold from environment (legacy compatibility)
        position_threshold: Position error threshold from environment
        orientation_threshold: Orientation error threshold from environment  
        joint_threshold: Joint error threshold from environment
        ratio_threshold: Minimum ratio of successful joints for overall success
        time_penalty: Per-step penalty
        smoothing_window: Window size for error smoothing
        exploration_bonus: Bonus for exploration in early training
        done: Episode completion flag for state reset
        
    Returns:
        tuple: (final_rewards, joint_rewards, new_best_distance, success)
    """

    # ---- 0. Reset state at episode boundaries (Change 3) ---------------------
    if episode_number == 1 or done:
        # Flush state at the start of a new run or episode completion
        compute_reward._state = []
        compute_reward._relax_position = REWARD_CONSTANTS.get("POSITION_RELAX_INIT", 5.0)
        compute_reward._relax_orientation = REWARD_CONSTANTS.get("ORIENTATION_RELAX_INIT", 3.0)
        compute_reward._success_history = deque(maxlen=int(REWARD_CONSTANTS.get("RELAX_WINDOW", 60)))

    # ---- 1. Basic validation --------------------------------------------------
    distance        = max(float(distance), EPS)
    begin_distance  = max(float(begin_distance), EPS)
    prev_best       = max(float(prev_best), EPS)
    episode_number  = max(int(episode_number), 1)
    total_episodes  = max(int(total_episodes), 1)
    
    # Validate and use the three thresholds from environment
    position_threshold = max(float(position_threshold), 1e-6)
    orientation_threshold = max(float(orientation_threshold), 1e-6)
    joint_threshold = max(float(joint_threshold), 1e-6)
    success_threshold = max(float(success_threshold), 1e-6)  # Legacy compatibility

    num_joints = len(joint_errors)
    if num_joints == 0:
        logging.error("compute_reward: empty joint_errors")
        return np.zeros(0), [], prev_best, False

    # signed → magnitude
    joint_errors     = np.abs(wrap_angle_to_pi(np.asarray(joint_errors, dtype=np.float64)))
    linear_weights   = np.asarray(linear_weights,  dtype=np.float64)
    angular_weights  = np.asarray(angular_weights, dtype=np.float64)
    if joint_movements is None:
        joint_movements = np.zeros(num_joints, dtype=np.float64)
    else:
        joint_movements = np.asarray(joint_movements, dtype=np.float64)
        if joint_movements.size != num_joints:
            logging.warning("joint_movements length mismatch; resizing")
            joint_movements = np.resize(joint_movements, num_joints)

    # pad / warn on length mismatch
    if linear_weights.size  != num_joints:
        logging.warning("linear_weights length mismatch; padding / truncating")
        linear_weights = np.resize(linear_weights,  num_joints)
        linear_weights.fill(1.0 / num_joints)

    if angular_weights.size != num_joints:
        logging.warning("angular_weights length mismatch; padding / truncating")
        angular_weights = np.resize(angular_weights, num_joints)
        angular_weights.fill(1.0 / num_joints)

    linear_weights  = np.abs(linear_weights)
    angular_weights = np.abs(angular_weights)
    linear_weights  /= (linear_weights.sum()  + EPS)
    angular_weights /= (angular_weights.sum() + EPS)

    if len(difficulties) != num_joints:
        logging.warning("difficulties length mismatch; padding / truncating")
        difficulties = (difficulties + [1.0] * num_joints)[:num_joints]
    difficulties = np.clip(difficulties, 0.1, 5.0)

    # ---- 2. Use direct errors from environment (Change 1) --------------------
    # If caller provided direct errors – use them
    if position_error is None:
        # fallback to old estimate
        orientation_error_local = compute_quaternion_distance(
            current_orientation, target_orientation)
        position_error = max(0.0, (distance - 0.3 * orientation_error_local) / 0.7)
    if orientation_error is None:
        orientation_error = compute_quaternion_distance(
            current_orientation, target_orientation)
    
    norm_distance     = np.clip(distance  / begin_distance, 0.0, 5.0)
    norm_orientation  = np.clip(orientation_error / np.pi, 0.0, 1.0)
    norm_position     = np.clip(position_error / (begin_distance * 0.7), 0.0, 5.0)
    episode_progress  = np.clip(episode_number / total_episodes, 0.1, 1.0)

    # ---- 3. Per-agent tracking state (thread-safe per process) ----------------
    if not hasattr(compute_reward, "_state") or \
       len(compute_reward._state) != num_joints:
        compute_reward._state = [{
            "best_error": np.inf,
            "prev_error": None,
            "window": deque(maxlen=smoothing_window),
            "success_count": 0,
            "total_count": 0
        } for _ in range(num_joints)]

    if not hasattr(compute_reward, "_relax_position"):
        compute_reward._relax_position = REWARD_CONSTANTS.get("POSITION_RELAX_INIT", 5.0)
    if not hasattr(compute_reward, "_relax_orientation"):
        compute_reward._relax_orientation = REWARD_CONSTANTS.get("ORIENTATION_RELAX_INIT", 3.0)
    if not hasattr(compute_reward, "_success_history"):
        compute_reward._success_history = deque(maxlen=int(REWARD_CONSTANTS.get("RELAX_WINDOW", 60)))

    # ---- 4. Constants with late-phase amplifier (Change 2) -------------------
    MAX_REWARD         = REWARD_CONSTANTS["MAX_REWARD"]
    MIN_REWARD         = REWARD_CONSTANTS["MIN_REWARD"]
    POSITION_SCALE     = REWARD_CONSTANTS["POSITION_SCALE"]
    ORIENTATION_SCALE  = REWARD_CONSTANTS["ORIENTATION_SCALE"]
    PERFORMANCE_SCALE  = REWARD_CONSTANTS["PERFORMANCE_SCALE"]
    IMPROVEMENT_SCALE  = REWARD_CONSTANTS["IMPROVEMENT_SCALE"]
    ERROR_PENALTY_SCALE= REWARD_CONSTANTS["ERROR_PENALTY_SCALE"]
    SUCCESS_BONUS_BASE = REWARD_CONSTANTS["SUCCESS_BONUS_BASE"]   # FIXED: Reduced from 5.0 to 1.0 to match other components
    STAY_REWARD_SCALE  = REWARD_CONSTANTS["STAY_REWARD_SCALE"]
    STAY_THRESHOLD_RATIO = REWARD_CONSTANTS["STAY_THRESHOLD_RATIO"]

    # FIXED: Balanced late-phase boost that scales both rewards AND penalties
    # This prevents reward collapse by keeping the reward scale balanced
    if episode_progress > 0.80:
        late_gain = 1.0 + 0.3 * (episode_progress - 0.80)  # max 1.0 → 1.3 (reduced from 1.5)
        MAX_REWARD *= late_gain
        SUCCESS_BONUS_BASE *= late_gain
        # CRITICAL: Also scale up MIN_REWARD to keep penalties proportional
        MIN_REWARD = max(MIN_REWARD * late_gain, -3.0)  # Allow more negative but keep balanced

    # ---- 5. Compute per-joint reward with improved success detection ----------
    final_rewards         = np.zeros(num_joints, dtype=np.float32)
    joint_specific_rewards = []
    stay_rewards_list = []
    individual_successes = []

    linear_weight_total = float(np.sum(linear_weights))
    if not np.isfinite(linear_weight_total) or linear_weight_total <= EPS:
        linear_weight_total = float(num_joints)
    angular_weight_total = float(np.sum(angular_weights))
    if not np.isfinite(angular_weight_total) or angular_weight_total <= EPS:
        angular_weight_total = float(num_joints)
    safe_position_threshold = max(position_threshold, EPS)
    safe_orientation_threshold = max(orientation_threshold, EPS)
    base_position_scale = max(safe_position_threshold, joint_threshold)
    base_orientation_scale = max(safe_orientation_threshold, joint_threshold)

    relax_min_factor = REWARD_CONSTANTS.get("RELAX_MIN_FACTOR", 1.2)
    relax_position_factor = max(float(compute_reward._relax_position), relax_min_factor)
    relax_orientation_factor = max(float(compute_reward._relax_orientation), relax_min_factor)
    shaping_position_threshold = max(base_position_scale * relax_position_factor, safe_position_threshold)
    shaping_orientation_threshold = max(base_orientation_scale * relax_orientation_factor, safe_orientation_threshold)

    if np.isfinite(position_error) and position_error <= position_threshold:
        position_alignment = 1.0
    elif np.isfinite(position_error):
        position_alignment = float(np.clip(position_threshold / (position_error + EPS), 0.0, 1.0))
    else:
        position_alignment = 0.0

    if np.isfinite(orientation_error) and orientation_error <= orientation_threshold:
        orientation_alignment = 1.0
    elif np.isfinite(orientation_error):
        orientation_alignment = float(np.clip(orientation_threshold / (orientation_error + EPS), 0.0, 1.0))
    else:
        orientation_alignment = 0.0

    pose_alignment = max(0.0, min(position_alignment, orientation_alignment))

    for j in range(num_joints):
        err = joint_errors[j]
        movement = abs(joint_movements[j])
        diff = difficulties[j]

        # --- track best & improvement
        st = compute_reward._state[j]
        st["window"].append(err)
        st["total_count"] += 1
        
        if err < st["best_error"]:
            st["best_error"] = err
        prev_err = st.get("prev_error")
        if prev_err is None or not np.isfinite(prev_err):
            improvement = 0.0
        else:
            baseline = max(abs(prev_err), joint_threshold, EPS)
            improvement = (prev_err - err) / baseline
        improvement = np.clip(improvement, -1.0, 1.0)

        # --- SUCCESS DETECTION using all three thresholds ---
        joint_error_success = err <= joint_threshold
        position_success = position_error <= position_threshold
        orientation_success = orientation_error <= orientation_threshold

        # FIXED: Use only joint-specific error (removed global dependencies)
        # This gives clearer credit assignment - each joint learns independently
        is_joint_successful = joint_error_success
        
        if is_joint_successful:
            st["success_count"] += 1
        individual_successes.append(is_joint_successful)

        # --- reward components ---
        linear_share = linear_weights[j] / linear_weight_total
        angular_share = angular_weights[j] / angular_weight_total
        joint_position_error = position_error * linear_share
        joint_orientation_error = orientation_error * angular_share

        norm_joint_position = np.clip(joint_position_error / shaping_position_threshold, 0.0, 2.5)
        norm_joint_orientation = np.clip(joint_orientation_error / shaping_orientation_threshold, 0.0, 2.5)

        pos_r  = POSITION_SCALE    * (1.05 - norm_joint_position)
        ori_r  = ORIENTATION_SCALE * (1.05 - norm_joint_orientation)
        pos_r  = np.clip(pos_r, -0.6, 0.8)
        ori_r  = np.clip(ori_r, -0.4, 0.6)

        perf_r = PERFORMANCE_SCALE * (1.0 - np.clip(err / np.pi, 0.0, 1.0))
        impr_r = IMPROVEMENT_SCALE * improvement * episode_progress
        impr_r = np.clip(impr_r, -0.15, 0.15)

        err_pen = -ERROR_PENALTY_SCALE * np.clip(err / np.pi, 0.0, 1.0)

        # SUCCESS BONUS using joint_threshold
        succ_bonus = 0.0
        if is_joint_successful:
            # Base success bonus
            base_bonus = SUCCESS_BONUS_BASE * episode_progress
            
            # Precision bonus: the smaller the error relative to threshold, the bigger the bonus
            precision_factor = max(0.1, (joint_threshold - err) / joint_threshold)
            precision_bonus = base_bonus * precision_factor
            
            # Consistency bonus
            recent_success_rate = st["success_count"] / max(st["total_count"], 1)
            consistency_bonus = base_bonus * 0.5 * recent_success_rate

            succ_bonus = base_bonus + precision_bonus + consistency_bonus
            succ_bonus = np.clip(succ_bonus, 0.0, MAX_REWARD * 0.6)
            succ_bonus *= pose_alignment

        # FIXED: Extended exploration bonus with gradual decay
        # Instead of cutting off at 30%, gradually reduce until 60%
        if episode_progress < 0.6:
            decay_factor = 1.0 - (episode_progress / 0.6)  # 1.0 → 0.0 over first 60%
            expl_bonus = exploration_bonus * decay_factor
        else:
            expl_bonus = 0.0

        time_pen = time_penalty * (linear_weights[j] + angular_weights[j])

        # Staying reward encourages minimal motion after converging
        # FIXED: Only require joint success, not full pose convergence
        # This allows joints to get stay rewards independently
        stay_reward = 0.0
        if is_joint_successful:
            stay_threshold = max(joint_threshold * STAY_THRESHOLD_RATIO, 1e-6)
            stay_factor = np.clip(1.0 - (movement / stay_threshold), 0.0, 1.0)
            stay_reward = STAY_REWARD_SCALE * stay_factor * pose_alignment

        reward = (pos_r + ori_r + perf_r + impr_r +
                  err_pen + succ_bonus + expl_bonus + time_pen + stay_reward)

        # FIXED: Asymmetric difficulty scaling to avoid amplifying penalties
        # Only apply difficulty scaling to positive rewards to prevent negative spiral
        if reward > 0:
            reward *= np.clip(diff, 0.85, 1.15)  # Gentle boost for harder problems
        else:
            reward *= np.clip(diff, 0.95, 1.05)  # Minimal penalty scaling
        reward  = float(np.clip(reward, MIN_REWARD, MAX_REWARD))

        final_rewards[j] = reward
        stay_rewards_list.append(float(stay_reward))
        st["prev_error"] = err

    pose_penalty = 0.0
    if np.isfinite(position_error) and position_error > position_threshold:
        pos_overshoot = (position_error - position_threshold) / (position_threshold + EPS)
        pose_penalty -= REWARD_CONSTANTS.get("POSITION_FAILURE_PENALTY", 0.0) * pos_overshoot
    if np.isfinite(orientation_error) and orientation_error > orientation_threshold:
        ori_overshoot = (orientation_error - orientation_threshold) / (orientation_threshold + EPS)
        pose_penalty -= REWARD_CONSTANTS.get("ORIENTATION_FAILURE_PENALTY", 0.0) * ori_overshoot

    if pose_penalty < 0.0:
        penalty_cap = max(float(REWARD_CONSTANTS.get("POSE_PENALTY_CAP", 10.0)), 0.0)
        pose_penalty = float(np.clip(pose_penalty, -penalty_cap, 0.0))
        final_rewards = np.clip(
            final_rewards + (pose_penalty / num_joints),
            MIN_REWARD,
            MAX_REWARD
        )

    # ---- 6. REBALANCED success detection: Progressive strictness ---------------------------
    success_ratio = sum(individual_successes) / num_joints

    # Progressive criteria: easier early (OR logic), stricter late (AND logic)
    ramp = np.clip((episode_progress - 0.8) / 0.2, 0.0, 1.0)
    strict_ratio = 0.35 + 0.15 * ramp  # 35% → 50% over training (was 40% → 50%)

    # Early training (<80%): Need good joint ratio OR pose accuracy
    # Late training (≥80%): Need both joint ratio AND pose accuracy
    if episode_progress < 0.8:
        success = (success_ratio >= strict_ratio
                   or (position_error <= position_threshold
                       and orientation_error <= orientation_threshold))
    else:
        # Late training: stricter requirements
        success = (success_ratio >= strict_ratio
                   and position_error <= position_threshold
                   and orientation_error <= orientation_threshold)

    # Log detailed success info occasionally
    if episode_number % 50 == 0 and success:
        logging.info(f"SUCCESS at episode {episode_number}:")
        logging.info(f"  Joint success ratio: {success_ratio:.3f} (threshold: {strict_ratio:.3f})")
        logging.info(f"  Position error: {position_error:.4f} (threshold: {position_threshold:.4f})")
        logging.info(f"  Orientation error: {orientation_error:.4f} (threshold: {orientation_threshold:.4f})")
        logging.info(f"  Mean joint error: {np.mean(joint_errors):.4f} (threshold: {joint_threshold:.4f})")

    new_best = min(prev_best, distance)

    # ---- 6a. Update relaxation schedule based on achieved success -------------
    hist = compute_reward._success_history
    if hist is not None:
        hist.append(success_ratio)
    relax_target = REWARD_CONSTANTS.get("RELAX_TARGET_SUCCESS", 0.2)
    relax_min_factor = REWARD_CONSTANTS.get("RELAX_MIN_FACTOR", 1.2)
    relax_decay = REWARD_CONSTANTS.get("RELAX_DECAY", 0.9)
    relax_eval_min_count = int(REWARD_CONSTANTS.get("RELAX_EVAL_MIN_COUNT", 30))

    # Evaluate once enough data collected and only tighten when performance warrants it
    if hist is not None and len(hist) >= relax_eval_min_count:
        avg_success = float(sum(hist) / len(hist))
        if avg_success >= relax_target:
            compute_reward._relax_position = max(
                relax_min_factor, compute_reward._relax_position * relax_decay)
            compute_reward._relax_orientation = max(
                relax_min_factor, compute_reward._relax_orientation * relax_decay)
            hist.clear()

    # ---- 6b. Cooperative team bonus --------------------------------------------------------
    TEAM_BONUS_SCALE = REWARD_CONSTANTS.get("TEAM_BONUS_SCALE", 0.0)
    team_alignment_min = REWARD_CONSTANTS.get("TEAM_ALIGNMENT_MIN", 0.0)
    if TEAM_BONUS_SCALE and success_ratio > 0.0 and pose_alignment >= team_alignment_min:
        # FIXED: Keep team bonus constant instead of decreasing
        # This prevents reward collapse by maintaining cooperative incentives
        coop_scale = TEAM_BONUS_SCALE * 1.2  # Constant, was (1.2 - 0.2 * episode_progress)
        cooperative_total = coop_scale * success_ratio * pose_alignment
        successful_count = sum(individual_successes)

        shared_component = cooperative_total * 0.35
        targeted_component = cooperative_total - shared_component

        if shared_component > 0.0:
            shared_bonus = shared_component / num_joints
            final_rewards = np.clip(final_rewards + shared_bonus, MIN_REWARD, MAX_REWARD)

        if targeted_component > 0.0 and successful_count > 0:
            per_joint_bonus = targeted_component / successful_count
            for idx, succeeded in enumerate(individual_successes):
                if succeeded:
                    final_rewards[idx] = float(np.clip(final_rewards[idx] + per_joint_bonus, MIN_REWARD, MAX_REWARD))

        joint_specific_rewards = final_rewards.tolist()

    # ---- 7. Optional variance normalisation ------------------------------------
    if final_rewards.std() > 4.0:
        mu, sigma = final_rewards.mean(), final_rewards.std() + EPS
        final_rewards = np.clip((final_rewards - mu) / sigma,
                                MIN_REWARD, MAX_REWARD)
        joint_specific_rewards = final_rewards.tolist()
    else:
        joint_specific_rewards = final_rewards.tolist()

    return final_rewards, joint_specific_rewards, new_best, success, stay_rewards_list


def adaptive_clip(rewards, stats):
    """
    ENHANCED: Safer, faster adaptive clipping with Welford's online algorithm (Change 5).
    
    Args:
        rewards (np.array): Rewards to clip
        stats (dict): Statistics for adaptive clipping
        
    Returns:
        np.array: Clipped rewards
    """
    try:
        rewards = np.array(rewards)
        if len(rewards) == 0:
            return rewards
            
        # Handle non-finite values
        finite_mask = np.isfinite(rewards)
        if not np.any(finite_mask):
            return np.zeros_like(rewards)
            
        finite_rewards = rewards[finite_mask]
        curr_mean = np.mean(finite_rewards)
        
        # Welford's online update for numerical stability
        alpha = 0.99
        delta = curr_mean - stats.get('running_mean', curr_mean)
        stats['running_mean'] = stats.get('running_mean', curr_mean) + alpha * delta
        stats['running_M2'] = stats.get('running_M2', 0.0) + delta * (curr_mean - stats['running_mean'])
        n = stats.get('count', 0) + 1
        stats['count'] = n
        stats['running_std'] = math.sqrt(stats['running_M2'] / max(n-1, 1))
        
        # Conservative clipping bounds
        clip_factor = 1.0
        lower_bound = stats['running_mean'] - clip_factor * stats['running_std']
        upper_bound = stats['running_mean'] + clip_factor * stats['running_std']
        
        # Apply conservative bounds
        lower_bound = max(lower_bound, -2.0)  # Never clip below -2
        upper_bound = min(upper_bound, 8.0)   # Never clip above 8 (increased for success bonuses)
        
        clipped_rewards = np.copy(rewards)
        clipped_rewards[finite_mask] = np.clip(finite_rewards, lower_bound, upper_bound)
        
        return clipped_rewards
        
    except Exception as e:
        logging.warning(f"Adaptive clipping error: {e}")
        return np.clip(rewards, -2.0, 8.0)


# ====== ENHANCED JACOBIAN COMPUTATION (Change 6) ======

def compute_jacobian_linear(robot_id, joint_indices, joint_angles):
    """
    ENHANCED: Use PyBullet's built-in Jacobian computation for better performance.
    
    Args:
        robot_id (int): PyBullet robot ID
        joint_indices (list): List of joint indices
        joint_angles (list): List of joint angles
        
    Returns:
        np.array: Linear Jacobian matrix (3 x n)
    """
    try:
        if not joint_indices:
            return np.zeros((3, 1))
            
        zero_vec = [0.0] * len(joint_indices)
        j_lin, _ = p.calculateJacobian(
            robot_id, joint_indices[-1],
            [0, 0, 0],  # local position of end-effector
            list(joint_angles), zero_vec, zero_vec
        )
        return np.asarray(j_lin, dtype=np.float64)
        
    except Exception as e:
        logging.warning(f"Built-in linear Jacobian computation failed: {e}")
        # Fallback to manual computation
        return compute_jacobian_linear_manual(robot_id, joint_indices, joint_angles)


def compute_jacobian_angular(robot_id, joint_indices, joint_angles):
    """
    ENHANCED: Use PyBullet's built-in Jacobian computation for better performance.
    
    Args:
        robot_id (int): PyBullet robot ID
        joint_indices (list): List of joint indices
        joint_angles (list): List of joint angles
        
    Returns:
        np.array: Angular Jacobian matrix (3 x n)
    """
    try:
        if not joint_indices:
            return np.zeros((3, 1))
            
        zero_vec = [0.0] * len(joint_indices)
        _, j_ang = p.calculateJacobian(
            robot_id, joint_indices[-1],
            [0, 0, 0],  # local position of end-effector
            list(joint_angles), zero_vec, zero_vec
        )
        return np.asarray(j_ang, dtype=np.float64)
        
    except Exception as e:
        logging.warning(f"Built-in angular Jacobian computation failed: {e}")
        # Fallback to manual computation
        return compute_jacobian_angular_manual(robot_id, joint_indices, joint_angles)


def compute_jacobian_linear_manual(robot_id, joint_indices, joint_angles):
    """
    Manual linear Jacobian computation as fallback.
    """
    try:
        num_joints = len(joint_indices)
        if num_joints == 0:
            return np.zeros((3, 1))
            
        J_linear = np.zeros((3, num_joints))
        
        ee_state = p.getLinkState(robot_id, joint_indices[-1])
        ee_pos = np.array(ee_state[4], dtype=np.float64)
        
        for i, joint_idx in enumerate(joint_indices):
            try:
                joint_info = p.getJointInfo(robot_id, joint_idx)
                joint_state = p.getLinkState(robot_id, joint_idx)
                
                joint_pos = np.array(joint_state[4], dtype=np.float64)
                joint_axis = np.array(joint_info[13], dtype=np.float64)
                
                # Robust axis normalization
                axis_norm = np.linalg.norm(joint_axis)
                if axis_norm > 1e-8:
                    joint_axis = joint_axis / axis_norm
                else:
                    joint_axis = np.array([0, 0, 1], dtype=np.float64)
                
                r = ee_pos - joint_pos
                cross_product = np.cross(joint_axis, r)
                
                # Ensure finite values
                if np.all(np.isfinite(cross_product)):
                    J_linear[:, i] = cross_product
                else:
                    J_linear[:, i] = [0, 0, 0]
                    
            except Exception as e:
                logging.warning(f"Error computing manual Jacobian for joint {i}: {e}")
                J_linear[:, i] = [0, 0, 0]
        
        return J_linear
        
    except Exception as e:
        logging.error(f"Manual Jacobian linear computation failed: {e}")
        return np.eye(3, len(joint_indices) if joint_indices else 1)


def compute_jacobian_angular_manual(robot_id, joint_indices, joint_angles):
    """
    Manual angular Jacobian computation as fallback.
    """
    try:
        num_joints = len(joint_indices)
        if num_joints == 0:
            return np.zeros((3, 1))
            
        J_angular = np.zeros((3, num_joints))
        
        for i, joint_idx in enumerate(joint_indices):
            try:
                joint_info = p.getJointInfo(robot_id, joint_idx)
                joint_state = p.getLinkState(robot_id, joint_idx)
                
                joint_axis = np.array(joint_info[13], dtype=np.float64)
                joint_orientation = np.array(joint_state[5], dtype=np.float64)
                
                # Robust normalization
                axis_norm = np.linalg.norm(joint_axis)
                if axis_norm > 1e-8:
                    joint_axis = joint_axis / axis_norm
                else:
                    joint_axis = np.array([0, 0, 1], dtype=np.float64)
                
                R = quaternion_to_rotation_matrix(joint_orientation)
                result = R @ joint_axis
                
                if np.all(np.isfinite(result)):
                    J_angular[:, i] = result
                else:
                    J_angular[:, i] = [0, 0, 1]
                    
            except Exception as e:
                logging.warning(f"Error computing manual angular Jacobian for joint {i}: {e}")
                J_angular[:, i] = [0, 0, 1]
        
        return J_angular
        
    except Exception as e:
        logging.error(f"Manual Jacobian angular computation failed: {e}")
        return np.eye(3, len(joint_indices) if joint_indices else 1)


def quaternion_to_rotation_matrix(q):
    """
    Robust quaternion to rotation matrix conversion.
    
    Args:
        q (np.array): Quaternion [x, y, z, w]
        
    Returns:
        np.array: 3x3 rotation matrix
    """
    try:
        q = np.array(q, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        
        if q_norm < 1e-8:
            return np.eye(3, dtype=np.float64)
            
        q = q / q_norm
        x, y, z, w = q
        
        R = np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
            [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
            [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ], dtype=np.float64)
        
        # Ensure orthogonality (numerical stability)
        if not np.allclose(np.linalg.det(R), 1.0, atol=1e-6):
            logging.warning("Non-orthogonal rotation matrix detected")
            return np.eye(3, dtype=np.float64)
            
        return R
        
    except Exception as e:
        logging.warning(f"Quaternion conversion failed: {e}")
        return np.eye(3, dtype=np.float64)


def assign_joint_weights(jacobian_linear, jacobian_angular):
    """
    Robust weight assignment with proper handling of edge cases.
    
    Args:
        jacobian_linear (np.array): Linear Jacobian matrix
        jacobian_angular (np.array): Angular Jacobian matrix
        
    Returns:
        tuple: (linear_weights, angular_weights)
    """
    try:
        if jacobian_linear.size == 0 or jacobian_angular.size == 0:
            n_joints = max(jacobian_linear.shape[1] if jacobian_linear.size > 0 else 1,
                          jacobian_angular.shape[1] if jacobian_angular.size > 0 else 1)
            uniform = np.ones(n_joints) / n_joints
            return uniform, uniform
            
        linear_weights = np.linalg.norm(jacobian_linear, axis=0)
        angular_weights = np.linalg.norm(jacobian_angular, axis=0)
        
        # Handle zero or near-zero weights
        linear_sum = np.sum(linear_weights)
        angular_sum = np.sum(angular_weights)
        
        if linear_sum < 1e-8:
            linear_weights = np.ones(len(linear_weights)) / len(linear_weights)
        else:
            linear_weights = linear_weights / linear_sum
            
        if angular_sum < 1e-8:
            angular_weights = np.ones(len(angular_weights)) / len(angular_weights)
        else:
            angular_weights = angular_weights / angular_sum
        
        # Ensure weights are finite
        linear_weights = np.where(np.isfinite(linear_weights), linear_weights, 1.0/len(linear_weights))
        angular_weights = np.where(np.isfinite(angular_weights), angular_weights, 1.0/len(angular_weights))
        
        return linear_weights, angular_weights
        
    except Exception as e:
        logging.error(f"Joint weight assignment failed: {e}")
        n = jacobian_linear.shape[1] if len(jacobian_linear.shape) > 1 else 1
        uniform = np.ones(n) / n
        return uniform, uniform


# ====== HELPER FUNCTIONS ======

def compute_weighted_joint_rewards(joint_errors, linear_weights, angular_weights, progress_reward):
    """
    Simplified weighted reward computation with proper bounds.
    
    Args:
        joint_errors (list): List of joint errors
        linear_weights (np.array): Linear movement weights
        angular_weights (np.array): Angular movement weights
        progress_reward (float): Progress-based reward
        
    Returns:
        np.array: Weighted joint rewards
    """
    try:
        combined_weights = (linear_weights + angular_weights) / 2.0
        combined_weights = combined_weights / (np.sum(combined_weights) + 1e-8)
        
        # Simple inverse error weighting
        safe_errors = np.maximum(np.array(joint_errors), 1e-6)
        inverse_errors = 1.0 / safe_errors
        inverse_errors = inverse_errors / (np.sum(inverse_errors) + 1e-8)
        
        rewards = np.clip(progress_reward * combined_weights * inverse_errors, -1.0, 2.0)
        return rewards
        
    except Exception:
        return np.zeros(len(joint_errors))


def compute_shaped_distance(distance, begin_distance):
    """
    Safe distance shaping with bounds checking.
    
    Args:
        distance (float): Current distance
        begin_distance (float): Initial distance
        
    Returns:
        tuple: (shaped_distance, shaped_begin_distance)
    """
    try:
        safe_dist = max(float(distance), 1e-8)
        safe_begin = max(float(begin_distance), 1e-8)
        return np.log(safe_dist + 1e-6), np.log(safe_begin + 1e-6)
    except Exception:
        return 0.0, 1.0


def compute_progress_reward(shaped_dist, shaped_begin, scale=1.0):
    """
    Bounded progress reward calculation with proper error handling.
    
    Args:
        shaped_dist (float): Current shaped distance
        shaped_begin (float): Initial shaped distance
        scale (float): Reward scaling factor
        
    Returns:
        float: Progress reward
    """
    try:
        progress = shaped_begin - shaped_dist
        if progress > 0:
            reward = scale * (np.exp(np.clip(progress, -5, 5)) - 1)
        else:
            reward = scale * progress * 0.3
        return np.clip(reward, -5.0, 5.0)
    except Exception:
        return 0.0


def compute_orientation_bonus(quaternion_distance):
    """
    Safe orientation bonus with proper angle handling.
    
    Args:
        quaternion_distance (float): Angular distance in [0, π]
        
    Returns:
        float: Orientation bonus between 0 and 1
    """
    try:
        # Take absolute value since we want magnitude for bonus calculation
        safe_dist = abs(quaternion_distance)
        safe_dist = np.clip(safe_dist, 0, np.pi)
        
        # Exponential decay: closer to target = higher bonus
        bonus = np.exp(-safe_dist)
        return float(bonus)
    except Exception:
        return 0.0


def inverse_scaled_rewards(scaled_rewards, epsilon=1e-6):
    """
    Safe inverse computation with bounds checking.
    
    Args:
        scaled_rewards (list): Scaled rewards to invert
        epsilon (float): Small value to avoid division by zero
        
    Returns:
        list: Inverse of the scaled rewards
    """
    try:
        result = []
        for reward in scaled_rewards:
            if abs(reward) < epsilon:
                result.append(1.0 / epsilon)
            else:
                result.append(1.0 / reward)
        return result
    except Exception:
        return [1.0] * len(scaled_rewards)


# ====== UTILITY FUNCTIONS FOR DEBUGGING ======

def validate_reward_function_inputs(
    distance, begin_distance, prev_best, current_orientation, target_orientation,
    joint_errors, linear_weights, angular_weights, difficulties
):
    """
    Validate all inputs to the reward function for debugging purposes.
    
    Args:
        All reward function parameters
        
    Returns:
        dict: Validation results with warnings and fixes applied
    """
    validation_results = {
        'warnings': [],
        'fixes_applied': [],
        'is_valid': True
    }
    
    try:
        # Check distance values
        if not np.isfinite(distance) or distance < 0:
            validation_results['warnings'].append(f"Invalid distance: {distance}")
            validation_results['is_valid'] = False
            
        if not np.isfinite(begin_distance) or begin_distance <= 0:
            validation_results['warnings'].append(f"Invalid begin_distance: {begin_distance}")
            validation_results['is_valid'] = False
            
        # Check quaternions
        current_quat = np.array(current_orientation)
        target_quat = np.array(target_orientation)
        
        if len(current_quat) != 4 or not np.all(np.isfinite(current_quat)):
            validation_results['warnings'].append("Invalid current_orientation quaternion")
            validation_results['is_valid'] = False
            
        if len(target_quat) != 4 or not np.all(np.isfinite(target_quat)):
            validation_results['warnings'].append("Invalid target_orientation quaternion")
            validation_results['is_valid'] = False
            
        # Check joint errors
        if not joint_errors or len(joint_errors) == 0:
            validation_results['warnings'].append("Empty joint_errors list")
            validation_results['is_valid'] = False
        else:
            for i, error in enumerate(joint_errors):
                if not np.isfinite(error):
                    validation_results['warnings'].append(f"Non-finite joint error at index {i}: {error}")
                    validation_results['is_valid'] = False
                    
        # Check weights
        if len(linear_weights) != len(joint_errors):
            validation_results['warnings'].append(
                f"Linear weights length ({len(linear_weights)}) != joint errors length ({len(joint_errors)})"
            )
            
        if len(angular_weights) != len(joint_errors):
            validation_results['warnings'].append(
                f"Angular weights length ({len(angular_weights)}) != joint errors length ({len(joint_errors)})"
            )
            
        # Check difficulties
        if len(difficulties) != len(joint_errors):
            validation_results['warnings'].append(
                f"Difficulties length ({len(difficulties)}) != joint errors length ({len(joint_errors)})"
            )
            
    except Exception as e:
        validation_results['warnings'].append(f"Validation error: {str(e)}")
        validation_results['is_valid'] = False
        
    return validation_results


def log_reward_statistics(rewards, episode_number, log_frequency=100):
    """
    Log reward statistics for monitoring training stability.
    
    Args:
        rewards (list or np.array): Reward values
        episode_number (int): Current episode number
        log_frequency (int): How often to log statistics
    """
    if episode_number % log_frequency == 0:
        try:
            rewards_array = np.array(rewards)
            
            stats = {
                'episode': episode_number,
                'mean': np.mean(rewards_array),
                'std': np.std(rewards_array),
                'min': np.min(rewards_array),
                'max': np.max(rewards_array),
                'median': np.median(rewards_array),
                'finite_count': np.sum(np.isfinite(rewards_array)),
                'total_count': len(rewards_array)
            }
            
            logging.info(f"Episode {episode_number} Reward Stats:")
            logging.info(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            logging.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            logging.info(f"  Median: {stats['median']:.4f}")
            logging.info(f"  Finite values: {stats['finite_count']}/{stats['total_count']}")
            
            # Warning checks
            if stats['std'] > 3.0:
                logging.warning("⚠️  High reward variance detected!")
                
            if stats['finite_count'] < stats['total_count']:
                logging.error("🚨 Non-finite rewards detected!")
                
            if abs(stats['mean']) > 5.0:
                logging.warning("⚠️  Extreme reward mean detected!")
                
        except Exception as e:
            logging.error(f"Error in reward statistics logging: {e}")


def test_reward_function():
    """
    Test the reward function with various edge cases to ensure robustness.
    """
    print("Testing enhanced reward function robustness...")
    
    # Test case 1: Normal inputs
    try:
        rewards, _, _, success, _ = compute_reward(
            distance=0.5,
            begin_distance=1.0,
            prev_best=0.6,
            current_orientation=[0, 0, 0, 1],
            target_orientation=[0, 0, 0.1, 0.995],
            joint_errors=[0.1, 0.2, 0.15],
            linear_weights=[0.3, 0.4, 0.3],
            angular_weights=[0.2, 0.5, 0.3],
            difficulties=[1.0, 1.5, 1.2],
            episode_number=100,
            total_episodes=1000,
            position_error=0.05,
            orientation_error=0.1
        )
        print("✅ Normal inputs test passed")
        print(f"   Rewards: {rewards}")
        print(f"   Success: {success}")
        
    except Exception as e:
        print(f"❌ Normal inputs test failed: {e}")
    
    # Test case 2: Success condition test with direct errors
    try:
        rewards, _, _, success, _ = compute_reward(
            distance=0.005,  # Very small distance
            begin_distance=1.0,
            prev_best=0.6,
            current_orientation=[0, 0, 0, 1],
            target_orientation=[0, 0, 0, 1],  # Same orientation
            joint_errors=[0.005, 0.008, 0.003],  # Very small joint errors
            linear_weights=[0.3, 0.4, 0.3],
            angular_weights=[0.2, 0.5, 0.3],
            difficulties=[1.0, 1.5, 1.2],
            episode_number=500,
            total_episodes=1000,
            position_error=0.005,  # Direct position error
            orientation_error=0.01,  # Direct orientation error
            position_threshold=0.02,
            orientation_threshold=0.05,
            joint_threshold=0.01
        )
        print("✅ Success condition test with direct errors passed")
        print(f"   Rewards: {rewards}")
        print(f"   Success detected: {success}")
        
    except Exception as e:
        print(f"❌ Success condition test failed: {e}")
    
    # Test case 3: Late-stage training boost
    try:
        rewards_early, _, _, _, _ = compute_reward(
            distance=0.01,
            begin_distance=1.0,
            prev_best=0.02,
            current_orientation=[0, 0, 0, 1],
            target_orientation=[0, 0, 0, 1],
            joint_errors=[0.008, 0.009, 0.007],
            linear_weights=[0.3, 0.4, 0.3],
            angular_weights=[0.2, 0.5, 0.3],
            difficulties=[1.0, 1.5, 1.2],
            episode_number=200,  # Early training (20%)
            total_episodes=1000,
            position_error=0.008,
            orientation_error=0.005
        )
        
        rewards_late, _, _, _, _ = compute_reward(
            distance=0.01,
            begin_distance=1.0,
            prev_best=0.02,
            current_orientation=[0, 0, 0, 1],
            target_orientation=[0, 0, 0, 1],
            joint_errors=[0.008, 0.009, 0.007],
            linear_weights=[0.3, 0.4, 0.3],
            angular_weights=[0.2, 0.5, 0.3],
            difficulties=[1.0, 1.5, 1.2],
            episode_number=900,  # Late training (90%)
            total_episodes=1000,
            position_error=0.008,
            orientation_error=0.005
        )
        
        print("✅ Late-stage amplification test passed")
        print(f"   Early rewards: {rewards_early}")
        print(f"   Late rewards: {rewards_late}")
        print(f"   Amplification factor: {np.mean(rewards_late) / max(np.mean(rewards_early), 1e-6):.2f}")
        
    except Exception as e:
        print(f"❌ Late-stage amplification test failed: {e}")
    
    # Test case 4: Built-in Jacobian computation
    try:
        # This would require a real PyBullet environment, so we'll just test the fallback
        linear_jac = compute_jacobian_linear_manual(None, [0, 1, 2], [0.1, 0.2, 0.3])
        angular_jac = compute_jacobian_angular_manual(None, [0, 1, 2], [0.1, 0.2, 0.3])
        print("✅ Manual Jacobian fallback test passed")
        print(f"   Linear Jacobian shape: {linear_jac.shape}")
        print(f"   Angular Jacobian shape: {angular_jac.shape}")
        
    except Exception as e:
        print(f"❌ Jacobian computation test failed: {e}")
    
    # Test case 5: State reset functionality
    try:
        # Test episode boundary reset
        rewards1, _, _, _, _ = compute_reward(
            distance=0.1, begin_distance=1.0, prev_best=0.2,
            current_orientation=[0, 0, 0, 1], target_orientation=[0, 0, 0, 1],
            joint_errors=[0.05, 0.06], linear_weights=[0.5, 0.5], angular_weights=[0.5, 0.5],
            difficulties=[1.0, 1.0], episode_number=1, total_episodes=100,
            done=True  # Episode completion
        )
        
        rewards2, _, _, _, _ = compute_reward(
            distance=0.1, begin_distance=1.0, prev_best=0.2,
            current_orientation=[0, 0, 0, 1], target_orientation=[0, 0, 0, 1],
            joint_errors=[0.05, 0.06], linear_weights=[0.5, 0.5], angular_weights=[0.5, 0.5],
            difficulties=[1.0, 1.0], episode_number=2, total_episodes=100
        )
        
        print("✅ State reset test passed")
        print(f"   Episode 1 rewards: {rewards1}")
        print(f"   Episode 2 rewards: {rewards2}")
        
    except Exception as e:
        print(f"❌ State reset test failed: {e}")
    
    print("Enhanced reward function testing completed!")


# ====== ENVIRONMENT INTEGRATION EXAMPLE ======

def example_environment_integration():
    """
    Example of how to integrate the enhanced reward function with the environment.
    This shows the call site changes needed in InverseKinematicsEnv.step()
    """
    print("Example environment integration:")
    print("""
    # In InverseKinematicsEnv.step() - call site change only
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
        # NEW: Direct error inputs ↓↓↓
        position_error=float(np.linalg.norm(self.position_error)),
        orientation_error=float(np.linalg.norm(self.orientation_error)),
        position_threshold=float(self.position_threshold),
        orientation_threshold=float(self.orientation_threshold),
        joint_threshold=float(self.success_threshold),
        done=done  # Pass episode completion flag
    )
    """)


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_reward_function()
    example_environment_integration()
