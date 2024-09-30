import numpy as np

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

def compute_cosine_distance(current_orientation, target_orientation):
    """
    Computes the cosine distance between two orientations represented as quaternions.

    Args:
        current_orientation (np.array): Current orientation quaternion [w, x, y, z].
        target_orientation (np.array): Target orientation quaternion [w, x, y, z].

    Returns:
        float: Cosine distance, adjusted to range [1, 3].
    """
    # Compute the dot product between the quaternions
    dot_product = np.clip(np.dot(current_orientation, target_orientation), -1.0, 1.0)
    cosine_similarity = dot_product

    # Cosine distance
    cosine_distance = 1 - cosine_similarity  # Range [0, 2]
    cosine_distance += 1  # Adjust to range [1, 3]

    return cosine_distance

def compute_overall_distance(current_position, target_position, current_orientation, target_orientation):
    """
    Computes the overall distance combining Euclidean and cosine distances.

    Returns:
        float: The overall distance.
    """
    # Euclidean (Gaussian) distance between positions
    euclidean_distance = compute_position_error(current_position, target_position)

    # Cosine distance between orientations
    cosine_distance = compute_cosine_distance(current_orientation, target_orientation)

    # Compute offset b
    b = cosine_distance - 1  # Range [0, 2]

    # Compute overall distance
    overall_distance = euclidean_distance * cosine_distance + b

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

