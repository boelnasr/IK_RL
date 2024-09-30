import numpy as np
import matplotlib.pyplot as plt
import json
import os

class TrainingMetrics:
    def __init__(self):
        self.logs = []

    def log_episode(self, joint_errors, rewards, success, entropy, actor_loss, critic_loss, env):
        """
        Log metrics for a single episode, ensuring that joint errors and success list match the environment's joint count.

        Args:
            joint_errors (list): Joint errors for each joint.
            rewards (list): Rewards for each step.
            success (list): Success status of the episode per joint (list of booleans).
            entropy (float): Entropy value for the episode.
            actor_loss (float): Actor loss for the episode.
            critic_loss (float): Critic loss for the episode.
            env (InverseKinematicsEnv): Instance of the environment.
        """
        # Ensure success is a list with a value for each joint
        if not isinstance(success, (list, np.ndarray)):
            raise ValueError(f"Expected success to be a list or array, but got {type(success)}. Value: {success}")

        if len(success) != env.num_joints:
            raise ValueError(f"Expected success list length to match number of joints ({env.num_joints}), "
                            f"but got {len(success)}. Success values: {success}")

        # Ensure joint_errors length matches the environment's joint count
        if len(joint_errors) != env.num_joints:
            raise ValueError(f"Expected joint_errors length to match number of joints ({env.num_joints}), "
                            f"but got {len(joint_errors)}. Joint errors: {joint_errors}")

        # Log the metrics for the episode
        self.logs.append({
            "joint_errors": joint_errors,
            "rewards": rewards,
            "success": success,  # Now a list of successes per joint
            "entropy": entropy,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss
        })



    def save_logs(self, log_file):
        """
        Save logs to a JSON file, converting numpy arrays and other non-serializable types to lists or strings.
        """
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert ndarray to list
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)  # Convert numpy floats to Python floats
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)  # Convert numpy ints to Python ints
            elif isinstance(obj, np.bool_):
                return bool(obj)  # Convert numpy bool to Python bool
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]  # Recursively process lists/tuples
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}  # Recursively process dicts
            else:
                return obj  # Return the object as-is if it's already serializable

        # Recursively convert all logs to serializable format
        serializable_logs = [convert_to_serializable(log) for log in self.logs]

        # Write the serializable logs to the JSON file
        with open(log_file, 'w') as f:
            json.dump(serializable_logs, f, indent=4)

    def calculate_metrics(self, env):
        """
        Calculate metrics such as mean joint errors, max joint errors, per-agent success rates, etc.

        Args:
            env (InverseKinematicsEnv): Instance of the environment to reference the number of joints.
        """
        metrics = {
            "mean_joint_errors": [],
            "max_joint_errors": [],
            "cumulative_rewards": [],
            "mean_episode_rewards": [],
            "success_rate_per_agent": [[] for _ in range(env.num_joints)],  # Success rate per agent
            "entropy": [],
            "actor_loss": [],
            "critic_loss": []
        }

        for episode_idx, episode_log in enumerate(self.logs):
            joint_errors = np.array(episode_log['joint_errors'])

            # Check if joint_errors length matches environment's joint count
            if joint_errors.shape[0] != env.num_joints:
                raise ValueError(f"Mismatch in joint error dimensions for episode {episode_idx}. "
                                 f"Expected {env.num_joints}, got {joint_errors.shape[0]}")

            rewards = episode_log['rewards']
            success = episode_log['success']  # List of success statuses per joint
            entropy = episode_log['entropy']
            actor_loss = episode_log['actor_loss']
            critic_loss = episode_log['critic_loss']

            # Calculate Mean and Max joint errors for each joint
            if len(joint_errors.shape) == 1:
                mean_joint_error = joint_errors
                max_joint_error = joint_errors
            else:
                mean_joint_error = np.mean(joint_errors, axis=0)
                max_joint_error = np.max(joint_errors, axis=0)

            # Calculate mean episode reward
            mean_reward = np.mean(rewards)

            # Append per-agent success rates
            for joint_idx in range(env.num_joints):
                metrics["success_rate_per_agent"][joint_idx].append(success[joint_idx])

            # Append to the metrics dictionary
            metrics["mean_joint_errors"].append(mean_joint_error)
            metrics["max_joint_errors"].append(max_joint_error)
            metrics["cumulative_rewards"].append(np.sum(rewards))
            metrics["mean_episode_rewards"].append(mean_reward)
            metrics["entropy"].append(entropy)
            metrics["actor_loss"].append(actor_loss)
            metrics["critic_loss"].append(critic_loss)

        # Convert metrics lists to arrays for easier plotting
        metrics["mean_joint_errors"] = np.array(metrics["mean_joint_errors"])
        metrics["max_joint_errors"] = np.array(metrics["max_joint_errors"])

        return metrics

    def plot_metrics(self, metrics, num_episodes, env):
        """
        Plot the calculated metrics, adjusting for the number of joints.

        Args:
            metrics (dict): Dictionary containing metric arrays.
            num_episodes (int): Number of episodes.
            env (InverseKinematicsEnv): Instance of the environment to reference the number of joints.
        """
        episodes = np.arange(1, num_episodes + 1)

        # Check the number of joints
        mean_errors = np.array(metrics['mean_joint_errors'])
        max_errors = np.array(metrics['max_joint_errors'])

        # Check if mean_errors and max_errors are empty or have unexpected dimensions
        if len(mean_errors) == 0 or len(max_errors) == 0:
            raise ValueError("mean_errors or max_errors are empty or malformed. Ensure the logging mechanism is working correctly.")
        
        if len(mean_errors.shape) < 2 or len(max_errors.shape) < 2:
            raise ValueError(f"Expected 2D arrays for mean_errors and max_errors, but got shapes: "
                             f"mean_errors: {mean_errors.shape}, max_errors: {max_errors.shape}")

        # Check if mean_errors and max_errors match the number of joints in the environment
        if mean_errors.shape[1] != env.num_joints or max_errors.shape[1] != env.num_joints:
            raise ValueError(f"Mismatch between environment joints ({env.num_joints}) and error logs. "
                             f"Mean errors shape: {mean_errors.shape}, Max errors shape: {max_errors.shape}")

        # Plot Mean Joint Errors
        plt.figure(figsize=(10, 6))
        for joint_idx in range(env.num_joints):
            plt.plot(episodes, mean_errors[:, joint_idx], label=f'Joint {joint_idx+1} Mean Error')

        plt.xlabel('Episodes')
        plt.ylabel('Mean Joint Errors')
        plt.title('Mean Joint Errors Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot Max Joint Errors
        plt.figure(figsize=(10, 6))
        for joint_idx in range(env.num_joints):
            plt.plot(episodes, max_errors[:, joint_idx], label=f'Joint {joint_idx+1} Max Error')
        plt.xlabel('Episodes')
        plt.ylabel('Max Joint Errors')
        plt.title('Max Joint Errors Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot Cumulative Rewards
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, metrics['cumulative_rewards'], label='Cumulative Rewards', color='b')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Rewards')
        plt.title('Cumulative Rewards Over Time')
        plt.grid(True)
        plt.show()

        # Plot Mean Episode Rewards
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, metrics['mean_episode_rewards'], label='Mean Episode Rewards', color='g')
        plt.xlabel('Episodes')
        plt.ylabel('Mean Episode Rewards')
        plt.title('Mean Episode Rewards Over Time')
        plt.grid(True)
        plt.show()

        # Plot Success Rate per Agent
        plt.figure(figsize=(10, 6))
        for joint_idx in range(env.num_joints):
            agent_success_rate = np.cumsum(metrics['success_rate_per_agent'][joint_idx]) / np.arange(1, num_episodes + 1)
            plt.plot(episodes, agent_success_rate, label=f'Joint {joint_idx+1} Success Rate')
        
        plt.xlabel('Episodes')
        plt.ylabel('Success Rate')
        plt.title('Success Rate Per Agent Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot Entropy
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, metrics['entropy'], label='Entropy', color='c')
        plt.xlabel('Episodes')
        plt.ylabel('Entropy')
        plt.title('Entropy Over Time')
        plt.grid(True)
        plt.show()

        # Plot Actor and Critic Loss
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, metrics['actor_loss'], label='Actor Loss', color='m')
        plt.plot(episodes, metrics['critic_loss'], label='Critic Loss', color='y')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.title('Actor and Critic Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
