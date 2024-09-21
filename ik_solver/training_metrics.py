import numpy as np
import matplotlib.pyplot as plt
import json
import os


class TrainingMetrics:
    def __init__(self):
        self.logs = []
        

    def log_episode(self, joint_errors, rewards, success, entropy, actor_loss, critic_loss):
        """
        Log metrics for a single episode.
        """
        self.logs.append({
            "joint_errors": joint_errors,
            "rewards": rewards,
            "success": success,
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

    def calculate_metrics(self):
        """
        Calculate metrics such as mean joint errors, max joint errors, etc.
        """
        metrics = {
            "mean_joint_errors": [],
            "max_joint_errors": [],
            "cumulative_rewards": [],
            "mean_episode_rewards": [],
            "success_rate": [],
            "entropy": [],
            "actor_loss": [],
            "critic_loss": []
        }

        for episode_log in self.logs:
            joint_errors = np.array(episode_log['joint_errors'])  # This will stay the same if `joint_errors` is already per-joint
            rewards = episode_log['rewards']
            success = episode_log['success']
            entropy = episode_log['entropy']
            actor_loss = episode_log['actor_loss']
            critic_loss = episode_log['critic_loss']

            # Calculate Mean and Max joint errors
            mean_joint_error = np.mean(joint_errors, axis=0)
            max_joint_error = np.max(joint_errors, axis=0)

            # Calculate mean episode reward
            mean_reward = np.mean(rewards)

            # Calculate success rate
            success_rate = np.mean(success)

            # Append to the metrics dictionary
            metrics["mean_joint_errors"].append(mean_joint_error)
            metrics["max_joint_errors"].append(max_joint_error)
            metrics["cumulative_rewards"].append(np.sum(rewards))
            metrics["mean_episode_rewards"].append(mean_reward)
            metrics["success_rate"].append(success_rate)
            metrics["entropy"].append(entropy)
            metrics["actor_loss"].append(actor_loss)
            metrics["critic_loss"].append(critic_loss)

        return metrics

    def plot_metrics(self, metrics, num_episodes,env):
        """
        Plot the calculated metrics.
        """
        episodes = np.arange(1, num_episodes + 1)

        # Check the number of joints
        mean_errors = np.array(metrics['mean_joint_errors'])
        max_errors = np.array(metrics['max_joint_errors'])
        
        # Verify the number of joints being plotted
        num_joints = env.num_joints # Assuming the second dimension represents joints
        print(f"Number of joints being plotted: {num_joints}")
        
        # Plot Mean Joint Errors
        plt.figure(figsize=(10, 6))
        for joint_idx in range(num_joints):
            plt.plot(episodes, mean_errors[:, joint_idx], label=f'Joint {joint_idx+1} Mean Error')
            
        plt.xlabel('Episodes')
        plt.ylabel('Mean Joint Errors')
        plt.title('Mean Joint Errors Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot Max Joint Errors
        plt.figure(figsize=(10, 6))
        for joint_idx in range(num_joints):
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

        # Plot Success Rate
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, metrics['success_rate'], label='Success Rate', color='r')
        plt.xlabel('Episodes')
        plt.ylabel('Success Rate')
        plt.title('Success Rate Over Time')
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
