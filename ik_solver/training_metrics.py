import numpy as np
import matplotlib.pyplot as plt
import json
import os
import logging
import time
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if available

import matplotlib.pyplot as plt
class TrainingMetrics:
    def __init__(self):
        timestamp = time.strftime("%H_%M_%S_%d_%m_%Y")
        self.log_filename = f"training_{timestamp}.log"

        # Set up a single logger instance to avoid duplicate logs
        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_filename, mode='w')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        
        self.logger.info(f"Training metrics logger initialized. Log file: {self.log_filename}")
        
        self.logs = []

    def log_episode(self, joint_errors, rewards, success, entropy, actor_loss, critic_loss, policy_loss, env):
        """
        Log metrics for a single episode, ensuring that joint errors and success list match the environment's joint count.

        Args:
            joint_errors (list): Joint errors for each time step (list of lists or arrays).
            rewards (list): Rewards for each step.
            success (list): Success status of the episode per joint (list of booleans).
            entropy (float): Entropy value for the episode.
            actor_loss (float): Actor loss for the episode.
            critic_loss (float): Critic loss for the episode.
            policy_loss (float): Policy loss for the episode.
            env (InverseKinematicsEnv): Instance of the environment.
        """
        # Ensure joint_errors is a list of lists (steps x joints)
        if not isinstance(joint_errors, list) or not all(isinstance(je, list) or isinstance(je, np.ndarray) for je in joint_errors):
            raise ValueError("joint_errors should be a list of lists or arrays, with each inner list representing joint errors at a step.")

        # Ensure success is a list with a value for each joint
        if not isinstance(success, (list, np.ndarray)):
            raise ValueError(f"Expected success to be a list or array, but got {type(success)}. Value: {success}")

        if len(success) != env.num_joints:
            raise ValueError(f"Expected success list length to match number of joints ({env.num_joints}), "
                            f"but got {len(success)}. Success values: {success}")

        # Log the metrics for the episode
        self.logs.append({
            "joint_errors": joint_errors,
            "rewards": rewards,
            "success": success,
            "entropy": entropy,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "policy_loss": policy_loss
        })

        logging.info(f"Episode data logged successfully. Joint Errors collected over {len(joint_errors)} steps.")

    def save_logs(self, log_file=None):
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

        # If no log filename is provided, generate one with a timestamp
        if log_file is None:
            timestamp = time.strftime("%H_%M_%S_%d_%m_%Y")  # Ensure the 'time' module is imported
            log_file = f"training_logs_{timestamp}.json"

        # Check if self.logs is initialized and not empty before proceeding
        if not self.logs:
            logging.warning("No logs available to save.")
            return

        # Recursively convert all logs to serializable format
        serializable_logs = [convert_to_serializable(log) for log in self.logs]

        # Write the serializable logs to the JSON file
        try:
            with open(log_file, 'w') as f:
                json.dump(serializable_logs, f, indent=4)
            logging.info(f"Logs successfully saved to {log_file}")
        except Exception as e:
            logging.error(f"Failed to save logs to {log_file}. Error: {e}")

    def calculate_metrics(self, env):
        logging.info("Starting metric calculation")
        if not self.logs:
            logging.warning("No logs available for metric calculation")
            return None

        metrics = {
            "mean_joint_errors": [],
            "max_joint_errors": [],
            "cumulative_rewards_per_agent": [[] for _ in range(env.num_joints)],  # Changed
            "mean_episode_rewards_per_agent": [[] for _ in range(env.num_joints)],  # Changed
            "success_rate_per_agent": [[] for _ in range(env.num_joints)],
            "entropy": [],
            "actor_loss": [],
            "critic_loss": [],
            "policy_loss": []
        }

        for episode_idx, episode_log in enumerate(self.logs):
            joint_errors = np.array(episode_log['joint_errors'])

            # Validate joint error dimensions
            if joint_errors.size == 0:
                logging.warning(f"Empty joint errors for episode {episode_idx}. Skipping.")
                continue

            if joint_errors.ndim != 2 or joint_errors.shape[1] != env.num_joints:
                logging.warning(f"Invalid joint error dimensions for episode {episode_idx}. "
                            f"Expected 2D array with shape (num_steps, {env.num_joints}), "
                            f"but got shape {joint_errors.shape}. Skipping this episode.")
                continue

            # Compute mean and max joint errors across steps for the episode
            metrics["mean_joint_errors"].append(np.mean(joint_errors, axis=0))
            metrics["max_joint_errors"].append(np.max(joint_errors, axis=0))

            # Process rewards per agent
            rewards = np.array(episode_log['rewards']).reshape(-1, env.num_joints)  # Reshape rewards
            for joint_idx in range(env.num_joints):
                joint_rewards = rewards[:, joint_idx]
                metrics["cumulative_rewards_per_agent"][joint_idx].append(np.sum(joint_rewards))
                metrics["mean_episode_rewards_per_agent"][joint_idx].append(np.mean(joint_rewards))

            # Additional metrics
            success = episode_log['success']
            entropy = episode_log['entropy']
            actor_loss = episode_log['actor_loss']
            critic_loss = episode_log['critic_loss']
            policy_loss = episode_log['policy_loss']

            # Append per-agent success rates
            for joint_idx in range(env.num_joints):
                metrics["success_rate_per_agent"][joint_idx].append(success[joint_idx])

            metrics["entropy"].append(entropy)
            metrics["actor_loss"].append(actor_loss)
            metrics["critic_loss"].append(critic_loss)
            metrics["policy_loss"].append(policy_loss)

        # Convert lists to numpy arrays
        metrics["mean_joint_errors"] = np.array(metrics["mean_joint_errors"])
        metrics["max_joint_errors"] = np.array(metrics["max_joint_errors"])
        metrics["cumulative_rewards_per_agent"] = [np.array(rewards) for rewards in metrics["cumulative_rewards_per_agent"]]
        metrics["mean_episode_rewards_per_agent"] = [np.array(rewards) for rewards in metrics["mean_episode_rewards_per_agent"]]

        return metrics
    def plot_metrics(self, metrics, env, show_plots=False):
        logging.info("Starting to plot metrics")
        # Use MATLAB-like style
        plt.style.use('classic')

        # Set global parameters to resemble MATLAB-like look
        plt.rcParams.update({
            'lines.linewidth': 2,  # Set default line width
            'axes.labelsize': 14,  # Set label font size
            'axes.titlesize': 16,  # Set title font size
            'axes.grid': True,     # Enable grid by default
            'grid.alpha': 0.7,     # Set grid transparency
            'grid.linestyle': '--',  # Set dashed grid line style
            'legend.fontsize': 12,  # Set legend font size
            'legend.frameon': True,  # Add frame around legend
            'font.family': 'sans-serif',  # Set font to sans-serif
            'font.sans-serif': 'Arial',   # Set default font family
            'xtick.labelsize': 12,  # X-axis tick label size
            'ytick.labelsize': 12,  # Y-axis tick label size
            'figure.figsize': (12, 8)  # Default figure size
        })

        # Check that the metrics dictionary has required keys
        if not metrics or 'mean_joint_errors' not in metrics or 'max_joint_errors' not in metrics:
            logging.error("Metrics dictionary is missing required keys")
            return

        mean_errors = metrics['mean_joint_errors']
        max_errors = metrics['max_joint_errors']
        
        # Log the shapes of mean_errors and max_errors
        logging.info(f"Shape of mean_errors: {mean_errors.shape if hasattr(mean_errors, 'shape') else 'no shape'}")
        logging.info(f"Shape of max_errors: {max_errors.shape if hasattr(max_errors, 'shape') else 'no shape'}")
        
        # Check if mean_errors and max_errors are not empty
        if not isinstance(mean_errors, np.ndarray) or not isinstance(max_errors, np.ndarray):
            logging.error("mean_errors or max_errors are not numpy arrays")
            return
        
        if mean_errors.size == 0 or max_errors.size == 0:
            logging.error("mean_errors or max_errors are empty")
            return

        # Ensure mean_errors and max_errors are 2D arrays
        if mean_errors.ndim == 1:
            mean_errors = mean_errors.reshape(-1, 1)
        if max_errors.ndim == 1:
            max_errors = max_errors.reshape(-1, 1)

        # Derive number of episodes from data
        num_episodes = mean_errors.shape[0]
        episodes = np.arange(1, num_episodes + 1)

        # Check if mean_errors and max_errors match the number of joints in the environment
        if mean_errors.shape[1] != env.num_joints or max_errors.shape[1] != env.num_joints:
            logging.error(f"Mismatch in number of joints. Expected {env.num_joints}, got {mean_errors.shape[1]}")
            return

        # Plot Mean and Max Joint Errors for each joint in subplots
        num_joints = env.num_joints
        fig, axes = plt.subplots(nrows=num_joints, ncols=1, figsize=(12, 4 * num_joints))
        if num_joints == 1:
            axes = [axes]  # Ensure axes is iterable

        for joint_idx in range(num_joints):
            ax = axes[joint_idx]
            ax.plot(episodes, mean_errors[:, joint_idx], label=f'Joint {joint_idx+1} Mean Error', color='b')
            ax.plot(episodes, max_errors[:, joint_idx], label=f'Joint {joint_idx+1} Max Error', color='r')
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Joint Error')
            ax.set_title(f'Joint {joint_idx+1} Mean and Max Errors Over Time')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig('joint_errors_per_joint.png')
        logging.info("Joint errors per joint plot saved successfully.")
        if show_plots:
            plt.show()
        plt.close()

        # Plot Cumulative Rewards per Agent
        fig, axes = plt.subplots(nrows=env.num_joints, ncols=1, figsize=(12, 4 * env.num_joints))
        if env.num_joints == 1:
            axes = [axes]

        for joint_idx in range(env.num_joints):
            ax = axes[joint_idx]
            ax.plot(episodes, metrics['cumulative_rewards_per_agent'][joint_idx], 
                    label=f'Joint {joint_idx+1} Cumulative Reward', color='b')
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Cumulative Reward')
            ax.set_title(f'Joint {joint_idx+1} Cumulative Reward Over Time')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig('cumulative_rewards_per_joint.png')
        logging.info("Cumulative rewards per joint plot saved successfully.")
        if show_plots:
            plt.show()
        plt.close()

        # Plot Mean Episode Rewards per Agent
        fig, axes = plt.subplots(nrows=env.num_joints, ncols=1, figsize=(12, 4 * env.num_joints))
        if env.num_joints == 1:
            axes = [axes]

        for joint_idx in range(env.num_joints):
            ax = axes[joint_idx]
            ax.plot(episodes, metrics['mean_episode_rewards_per_agent'][joint_idx], 
                    label=f'Joint {joint_idx+1} Mean Reward', color='g')
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Mean Episode Reward')
            ax.set_title(f'Joint {joint_idx+1} Mean Episode Reward Over Time')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig('mean_episode_rewards_per_joint.png')
        logging.info("Mean episode rewards per joint plot saved successfully.")
        if show_plots:
            plt.show()
        plt.close()

        # Plot Mean Episode Rewards per Agent
        fig, axes = plt.subplots(nrows=env.num_joints, ncols=1, figsize=(12, 4 * env.num_joints))
        if env.num_joints == 1:
            axes = [axes]  # Ensure axes is iterable

        mean_rewards_per_agent = np.array(metrics['mean_episode_rewards_per_agent']).reshape(-1, env.num_joints)
        for joint_idx in range(env.num_joints):
            ax = axes[joint_idx]
            ax.plot(episodes, mean_rewards_per_agent[:, joint_idx], 
                    label=f'Joint {joint_idx+1} Mean Reward', color='g')
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Mean Episode Reward')
            ax.set_title(f'Joint {joint_idx+1} Mean Episode Reward Over Time')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig('mean_episode_rewards_per_joint.png')
        logging.info("Mean episode rewards per joint plot saved successfully.")
        if show_plots:
            plt.show()
        plt.close()

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
        plt.savefig('success_rate_per_agent.png')
        logging.info("Success rate per agent plot saved successfully.")
        if show_plots:
            plt.show()
        plt.close()

        # Plot Entropy
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, metrics['entropy'], label='Entropy', color='c')
        plt.xlabel('Episodes')
        plt.ylabel('Entropy')
        plt.title('Entropy Over Time')
        plt.grid(True)
        plt.savefig('entropy.png')
        logging.info("Entropy plot saved successfully.")
        if show_plots:
            plt.show()
        plt.close()

        # Plot Actor and Critic Loss
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, metrics['actor_loss'], label='Actor Loss', color='m')
        plt.plot(episodes, metrics['critic_loss'], label='Critic Loss', color='y')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.title('Actor and Critic Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig('actor_critic_loss.png')
        logging.info("Actor and critic loss plot saved successfully.")
        if show_plots:
            plt.show()
        plt.close()

        # Plot Policy Loss
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, metrics['policy_loss'], label='Policy Loss', color='orange')
        plt.xlabel('Episodes')
        plt.ylabel('Policy Loss')
        plt.title('Policy Loss Over Time')
        plt.grid(True)
        plt.savefig('policy_loss.png')
        logging.info("Policy loss plot saved successfully.")
        if show_plots:
            plt.show()
        plt.close()
