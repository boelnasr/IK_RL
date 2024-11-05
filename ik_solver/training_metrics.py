import numpy as np
import matplotlib.pyplot as plt
import json
import os
import logging
import time
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive environments
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import logging
import time
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive environments

class TrainingMetrics:
    def __init__(self, num_joints: int, convergence_threshold: float = 0.95):
        """Initialize training metrics with improved logging and data structures."""
        # Set up timestamped logging
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.base_path = f"training_logs_{timestamp}"
        os.makedirs(self.base_path, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler(os.path.join(self.base_path, 'training.log'))
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)
        
        # Initialize metrics storage
        self.logs = []
        self.episode_metrics = {
            'joint_errors': [],
            'rewards': [],
            'success_rates': [],
            'entropy': [],
            'actor_loss': [],
            'critic_loss': [],
            'policy_loss': []
        }

        # Additional attributes for tracking
        self.episode_rewards = []
        self.successes = []
        self.policy_losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        self.joint_errors = []
        self.cumulative_rewards = []
        self.training_episodes = 0
        self.converged = False
        self.episodes_to_converge = None
        self.convergence_threshold = convergence_threshold

        # Per-joint success tracking
        self.num_joints = num_joints
        self.joint_successes = {joint_idx: [] for joint_idx in range(num_joints)}
        
        self.logger.info(f"Training metrics initialized. Logs will be saved to: {self.base_path}")

    def log_episode(self, joint_errors: List[List[float]], rewards: List[float], success: List[bool], 
                    entropy: float, actor_loss: float, critic_loss: float, policy_loss: List[float], 
                    env: Any) -> None:
        """
        Log comprehensive metrics for a single episode.
        
        Args:
            joint_errors: List of joint errors over time.
            rewards: List of rewards per agent for the episode.
            success: Success status per agent.
            entropy: Policy entropy value.
            actor_loss: Actor network loss.
            critic_loss: Critic network loss.
            policy_loss: List of policy losses per agent.
            env: Environment instance for additional metrics.
        """
        try:
            # Convert inputs to numpy arrays for consistency
            joint_errors = np.array(joint_errors)
            rewards = np.array(rewards)
            success = np.array(success)
            
            # Calculate episode statistics
            episode_data = {
                'joint_errors': {
                    'mean': np.mean(joint_errors, axis=0),
                    'max': np.max(joint_errors, axis=0),
                    'min': np.min(joint_errors, axis=0),
                    'std': np.std(joint_errors, axis=0)
                },
                'rewards': {
                    'total': rewards,
                    'mean': np.mean(rewards),
                    'std': np.std(rewards)
                },
                'success': success.tolist(),
                'success_rate': np.mean(success),
                'entropy': float(entropy),
                'actor_loss': float(actor_loss),
                'critic_loss': float(critic_loss),
                'policy_loss': policy_loss,
                'episode_length': joint_errors.shape[0]
            }
            
            # Add to logs
            self.logs.append(episode_data)
            
            # Update metrics and track success per joint
            self._update_episode_metrics(episode_data)
            self.episode_rewards.append(np.sum(rewards))
            self.successes.append(np.any(success))
            self.policy_losses.append(np.mean(policy_loss))
            self.actor_losses.append(actor_loss)
            self.critic_losses.append(critic_loss)
            self.entropies.append(entropy)
            self.joint_errors.append(np.mean(joint_errors))

            self.cumulative_rewards.append(np.sum(rewards) if not self.cumulative_rewards else 
                                            self.cumulative_rewards[-1] + np.sum(rewards))
            self.training_episodes += 1

            for joint_idx, joint_success in enumerate(success):
                self.joint_successes[joint_idx].append(joint_success)

            # Check for convergence
            if not self.converged and len(self.successes) >= 100:
                recent_success_rate = np.mean(self.successes[-100:])
                if recent_success_rate >= self.convergence_threshold:
                    self.converged = True
                    self.episodes_to_converge = self.training_episodes
                    self.logger.info(f"Convergence achieved at episode {self.training_episodes} with success rate {recent_success_rate:.4f}")

            self.logger.info(f"Episode {self.training_episodes} logged successfully. "
                             f"Mean reward: {episode_data['rewards']['mean']:.4f}, "
                             f"Success rate: {episode_data['success_rate']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error logging episode: {str(e)}")
            raise

    def _update_episode_metrics(self, episode_data: Dict) -> None:
        """Update running metrics with new episode data."""
        self.episode_metrics['joint_errors'].append(episode_data['joint_errors']['mean'])
        self.episode_metrics['rewards'].append(episode_data['rewards']['total'])
        self.episode_metrics['success_rates'].append(episode_data['success_rate'])
        self.episode_metrics['entropy'].append(episode_data['entropy'])
        self.episode_metrics['actor_loss'].append(episode_data['actor_loss'])
        self.episode_metrics['critic_loss'].append(episode_data['critic_loss'])
        self.episode_metrics['policy_loss'].append(episode_data['policy_loss'])

    def calculate_metrics(self, env: Any) -> Dict:
        """
        Calculate comprehensive training metrics.
        
        Args:
            env: Environment instance for context.
            
        Returns:
            Dictionary containing calculated metrics.
        """
        if not self.logs:
            self.logger.warning("No logs available for metric calculation")
            return {}

        try:
            num_episodes = len(self.logs)
            num_agents = self.num_joints
            total_rewards_per_agent = np.array([log['rewards']['total'] for log in self.logs])
            cumulative_rewards_per_agent = np.cumsum(total_rewards_per_agent, axis=0).T

            mean_episode_rewards_per_agent = total_rewards_per_agent.T
            agent_success = np.array([log['success'] for log in self.logs])
            cumulative_success_per_agent = np.cumsum(agent_success, axis=0).T
            success_rate_per_agent = cumulative_success_per_agent / np.arange(1, num_episodes + 1)
            overall_success_rate = np.mean(self.successes)

            policy_loss_per_agent = np.array(self.episode_metrics['policy_loss'])
            average_cumulative_reward = np.mean(self.cumulative_rewards)
            average_policy_loss = np.mean(self.policy_losses)
            average_joint_error = np.mean(self.joint_errors)
            episodes_to_converge = self.episodes_to_converge

            joint_success_rates = {f'Joint_{joint_idx + 1}_success_rate': np.mean(self.joint_successes[joint_idx])
                                   for joint_idx in range(num_agents)}

            metrics = {
                'joint_errors': {
                    'mean': np.array([log['joint_errors']['mean'] for log in self.logs]),
                    'max': np.array([log['joint_errors']['max'] for log in self.logs]),
                    'min': np.array([log['joint_errors']['min'] for log in self.logs])
                },
                'rewards': {
                    'mean': np.array([log['rewards']['mean'] for log in self.logs]),
                    'total': total_rewards_per_agent,
                    'cumulative': np.array(self.cumulative_rewards)
                },
                'success_rate': {
                    'per_episode': np.array([log['success_rate'] for log in self.logs]),
                    'cumulative': np.cumsum([log['success_rate'] for log in self.logs]) / np.arange(1, num_episodes + 1),
                    'overall': overall_success_rate,
                    'per_joint': joint_success_rates
                },
                'training': {
                    'entropy': np.array(self.episode_metrics['entropy']),
                    'actor_loss': np.array(self.episode_metrics['actor_loss']),
                    'critic_loss': np.array(self.episode_metrics['critic_loss']),
                    'policy_loss': np.mean(policy_loss_per_agent, axis=1)
                },
                'agent_success': agent_success,
                'cumulative_rewards_per_agent': cumulative_rewards_per_agent,
                'mean_episode_rewards_per_agent': mean_episode_rewards_per_agent,
                'success_rate_per_agent': success_rate_per_agent,
                'policy_loss_per_agent': policy_loss_per_agent.T,
                'average_cumulative_reward': average_cumulative_reward,
                'average_policy_loss': average_policy_loss,
                'average_joint_error': average_joint_error,
                'episodes_to_converge': episodes_to_converge,
                'total_training_episodes': self.training_episodes
            }
            
            window = min(100, num_episodes // 10)
            if window > 0:
                metrics['moving_averages'] = {
                    'reward': self._compute_moving_average(metrics['rewards']['mean'], window),
                    'success_rate': self._compute_moving_average(metrics['success_rate']['per_episode'], window)
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def _compute_moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute moving average with the specified window size."""
        return np.convolve(data, np.ones(window) / window, mode='valid')
    def save_logs(self, filename: str = None) -> None:
        """
        Save training logs to a JSON file.
        
        Args:
            filename: Optional custom filename for the logs
        """
        if not filename:
            filename = os.path.join(self.base_path, 'training_logs.json')
            
        try:
            # Convert complex data types to serializable types
            def make_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().tolist()
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    return str(obj)  # Convert other types to string representation

            serializable_logs = []
            for log in self.logs:
                serializable_log = {key: make_serializable(value) for key, value in log.items()}
                serializable_logs.append(serializable_log)

            with open(filename, 'w') as f:
                json.dump(serializable_logs, f, indent=2)
            
            self.logger.info(f"Logs saved successfully to: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving logs: {str(e)}")
            raise

    def save_figure(self, fig: plt.Figure, name: str) -> None:
        """Save figure with proper handling."""
        path = os.path.join(self.base_path, f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_metrics(self, metrics: Dict, env: Any, show_plots: bool = False) -> None:
        """Create and save all plots."""
        try:
            self._plot_joint_metrics(metrics, env, show_plots)
            self._plot_per_joint_cumulative_rewards(metrics, env, show_plots)
            self._plot_training_metrics(metrics, show_plots)
            self._plot_success_metrics(metrics, env, show_plots)
            self._plot_success_rate_per_agent_over_time(metrics, env, show_plots)
            self._plot_cumulative_rewards_per_agent(metrics, env, show_plots)
            self._plot_mean_episode_rewards_per_agent(metrics, env, show_plots)
            self._plot_combined_metrics(metrics, show_plots)
            self._plot_policy_loss(metrics, show_plots)
            self._plot_policy_loss_per_agent(metrics, show_plots)

            self.logger.info("All plots generated and saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {str(e)}")
            raise

    def _plot_joint_metrics(self, metrics: Dict, env: Any, show_plots: bool) -> None:
        """Plot joint-specific metrics."""
        num_joints = env.num_joints
        episodes = np.arange(1, len(metrics['joint_errors']['mean']) + 1)
        
        fig, axes = plt.subplots(num_joints, 1, figsize=(30, 6 * num_joints))
        if num_joints == 1:
            axes = [axes]
            
        for joint_idx in range(num_joints):
            ax = axes[joint_idx]
            mean_errors = metrics['joint_errors']['mean'][:, joint_idx]
            max_errors = metrics['joint_errors']['max'][:, joint_idx]
            min_errors = metrics['joint_errors']['min'][:, joint_idx]
            
            ax.plot(episodes, mean_errors, label='Mean Error', color='blue')
            ax.fill_between(episodes, min_errors, max_errors, alpha=0.2, color='blue')
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Joint Error')
            ax.set_title(f'Joint {joint_idx + 1} Error Metrics')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        self.save_figure(fig, 'joint_metrics')
        if show_plots:
            plt.show()

    def _plot_per_joint_cumulative_rewards(self, metrics: Dict, env: Any, show_plots: bool) -> None:
        """Plot cumulative rewards for each joint."""
        episodes = np.arange(1, metrics['cumulative_rewards_per_agent'].shape[1] + 1)
        num_joints = env.num_joints
        cumulative_rewards_per_agent = metrics['cumulative_rewards_per_agent']
        
        fig, axes = plt.subplots(nrows=num_joints, ncols=1, figsize=(30, 6 * num_joints))
        if num_joints == 1:
            axes = [axes]
        
        for joint_idx in range(num_joints):
            ax = axes[joint_idx]
            ax.plot(episodes, cumulative_rewards_per_agent[joint_idx], 
                    label=f'Joint {joint_idx+1} Cumulative Reward', color='b')
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Cumulative Reward')
            ax.set_title(f'Joint {joint_idx+1} Cumulative Reward Over Time')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        self.save_figure(fig, 'cumulative_rewards_per_joint')
        self.logger.info("Cumulative rewards per joint plot saved successfully.")
        if show_plots:
            plt.show()
        plt.close(fig)

    def _plot_training_metrics(self, metrics: Dict, show_plots: bool) -> None:
        """Plot training-specific metrics including policy loss."""
        episodes = np.arange(1, len(metrics['training']['entropy']) + 1)
        window = min(100, len(episodes) // 10)
        
        # 1. Plot entropy
        fig_entropy, ax_entropy = plt.subplots(figsize=(12, 6))
        ax_entropy.plot(episodes, metrics['training']['entropy'], label='Entropy', color='cyan')
        # Add moving average
        if window > 1:
            entropy_ma = self._compute_moving_average(metrics['training']['entropy'], window)
            ma_episodes = episodes[window-1:]
            ax_entropy.plot(ma_episodes, entropy_ma, label='Moving Average', color='red', linewidth=2)
        
        ax_entropy.set_title('Policy Entropy Over Time')
        ax_entropy.set_xlabel('Episodes')
        ax_entropy.set_ylabel('Entropy')
        ax_entropy.grid(True, alpha=0.3)
        ax_entropy.legend()
        plt.tight_layout()
        self.save_figure(fig_entropy, 'entropy_plot')
        plt.close(fig_entropy)

        # 2. Plot actor, critic, and policy losses
        fig_losses, ax_losses = plt.subplots(figsize=(12, 6))
        # Actor loss
        ax_losses.plot(episodes, metrics['training']['actor_loss'], label='Actor Loss', color='blue', alpha=0.7)
        # Critic loss
        ax_losses.plot(episodes, metrics['training']['critic_loss'], label='Critic Loss', color='red', alpha=0.7)
        # Policy loss
        ax_losses.plot(episodes, metrics['training']['policy_loss'], label='Policy Loss', color='green', alpha=0.7)
        
        # Add moving averages
        if window > 1:
            actor_ma = self._compute_moving_average(metrics['training']['actor_loss'], window)
            critic_ma = self._compute_moving_average(metrics['training']['critic_loss'], window)
            policy_ma = self._compute_moving_average(metrics['training']['policy_loss'], window)
            ma_episodes = episodes[window-1:]
            ax_losses.plot(ma_episodes, actor_ma, label='Actor MA', color='blue', linestyle='--', linewidth=2)
            ax_losses.plot(ma_episodes, critic_ma, label='Critic MA', color='red', linestyle='--', linewidth=2)
            ax_losses.plot(ma_episodes, policy_ma, label='Policy MA', color='green', linestyle='--', linewidth=2)
        
        ax_losses.set_title('Actor, Critic, and Policy Losses Over Time')
        ax_losses.set_xlabel('Episodes')
        ax_losses.set_ylabel('Loss')
        ax_losses.grid(True, alpha=0.3)
        ax_losses.legend()
        plt.tight_layout()
        self.save_figure(fig_losses, 'losses_plot')
        plt.close(fig_losses)

    def _plot_success_metrics(self, metrics: Dict, env: Any, show_plots: bool) -> None:
        """Plot success-related metrics."""
        episodes = np.arange(1, len(metrics['success_rate']['per_episode']) + 1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(episodes, metrics['success_rate']['cumulative'], label='Success Rate', color='green')
        
        window = min(100, len(episodes) // 10)
        if window > 1 and 'moving_averages' in metrics:
            ma_episodes = episodes[window-1:]
            ax.plot(ma_episodes, metrics['moving_averages']['success_rate'], 
                    label='Moving Average', color='red', linewidth=2)
        
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        self.save_figure(fig, 'success_rate')
        if show_plots:
            plt.show()
        plt.close(fig)

    def _plot_success_rate_per_agent_over_time(self, metrics: Dict, env: Any, show_plots: bool) -> None:
        """Plot success rate per agent over episodes."""
        num_agents = env.num_joints
        num_episodes = len(self.logs)
        episodes = np.arange(1, num_episodes + 1)
        success_rate_per_agent = metrics['success_rate_per_agent']  # Shape: (num_agents, num_episodes)

        fig, ax = plt.subplots(figsize=(10, 6))
        for joint_idx in range(num_agents):
            agent_success_rate = success_rate_per_agent[joint_idx]
            ax.plot(episodes, agent_success_rate, label=f'Joint {joint_idx+1} Success Rate')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate Per Agent Over Time')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        self.save_figure(fig, 'success_rate_per_agent')
        self.logger.info("Success rate per agent plot saved successfully.")
        if show_plots:
            plt.show()
        plt.close(fig)

    def _plot_cumulative_rewards_per_agent(self, metrics: Dict, env: Any, show_plots: bool) -> None:
        """Plot cumulative rewards per agent (joint) over episodes."""
        episodes = np.arange(1, metrics['cumulative_rewards_per_agent'].shape[1] + 1)
        num_joints = env.num_joints
        cumulative_rewards_per_agent = metrics['cumulative_rewards_per_agent']
        
        fig, axes = plt.subplots(nrows=num_joints, ncols=1, figsize=(30, 6 * num_joints))
        if num_joints == 1:
            axes = [axes]
        
        for joint_idx in range(num_joints):
            ax = axes[joint_idx]
            ax.plot(episodes, cumulative_rewards_per_agent[joint_idx], 
                    label=f'Joint {joint_idx+1} Cumulative Reward', color='b')
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Cumulative Reward')
            ax.set_title(f'Joint {joint_idx+1} Cumulative Reward Over Time')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        self.save_figure(fig, 'cumulative_rewards_per_joint')
        self.logger.info("Cumulative rewards per joint plot saved successfully.")
        if show_plots:
            plt.show()
        plt.close(fig)

    def _plot_mean_episode_rewards_per_agent(self, metrics: Dict, env: Any, show_plots: bool) -> None:
        """Plot mean episode rewards per agent (joint) over episodes."""
        episodes = np.arange(1, metrics['mean_episode_rewards_per_agent'].shape[1] + 1)
        num_joints = env.num_joints
        mean_episode_rewards_per_agent = metrics['mean_episode_rewards_per_agent']
        
        fig, axes = plt.subplots(nrows=num_joints, ncols=1, figsize=(30, 6 * num_joints))
        if num_joints == 1:
            axes = [axes]
        
        for joint_idx in range(num_joints):
            ax = axes[joint_idx]
            ax.plot(episodes, mean_episode_rewards_per_agent[joint_idx], 
                    label=f'Joint {joint_idx+1} Mean Reward', color='g')
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Mean Episode Reward')
            ax.set_title(f'Joint {joint_idx+1} Mean Episode Reward Over Time')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        self.save_figure(fig, 'mean_episode_rewards_per_joint')
        self.logger.info("Mean episode rewards per joint plot saved successfully.")
        if show_plots:
            plt.show()
        plt.close(fig)

    def _plot_combined_metrics(self, metrics: Dict, show_plots: bool) -> None:
        """Plot combined performance metrics."""
        episodes = np.arange(1, len(metrics['rewards']['total']) + 1)
        
        # Normalize metrics
        norm_reward = (np.array([np.sum(r) for r in metrics['rewards']['total']]) - np.min(metrics['rewards']['cumulative'])) / \
                      (np.max(metrics['rewards']['cumulative']) - np.min(metrics['rewards']['cumulative']) + 1e-8)
        norm_success = metrics['success_rate']['cumulative']
        norm_error = 1.0 - (np.mean(metrics['joint_errors']['mean'], axis=1) - 
                            np.min(np.mean(metrics['joint_errors']['mean'], axis=1))) / \
                        (np.max(np.mean(metrics['joint_errors']['mean'], axis=1)) - 
                            np.min(np.mean(metrics['joint_errors']['mean'], axis=1)) + 1e-8)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(episodes, norm_reward, label='Normalized Reward', color='blue')
        ax.plot(episodes, norm_success, label='Success Rate', color='green')
        ax.plot(episodes, norm_error, label='Normalized Accuracy', color='red')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Combined Performance Metrics')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        self.save_figure(fig, 'combined_metrics')
        plt.close(fig)

    def save_final_report(self, metrics: Dict) -> None:
        """
        Generate and save a comprehensive training report.
        
        Args:
            metrics: Dictionary containing all training metrics
        """
        report_path = os.path.join(self.base_path, 'training_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=== Training Performance Report ===\n\n")
            
            # Overall Statistics
            f.write("Overall Performance:\n")
            f.write(f"Total Episodes: {metrics['total_training_episodes']}\n")
            f.write(f"Final Success Rate: {metrics['success_rate']['overall']:.4f}\n")
            f.write(f"Average Cumulative Reward: {metrics['average_cumulative_reward']:.4f}\n")
            f.write(f"Average Policy Loss: {metrics['average_policy_loss']:.4f}\n")
            f.write(f"Average Joint Error: {metrics['average_joint_error']:.4f}\n")
            if metrics['episodes_to_converge'] is not None:
                f.write(f"Training Episodes to Converge: {metrics['episodes_to_converge']}\n")
            else:
                f.write("Training Episodes to Converge: Not converged within training episodes\n")
            f.write(f"Total Cumulative Reward: {metrics['rewards']['cumulative'][-1]:.4f}\n\n")
            
            # Training Stability
            f.write("Training Stability:\n")
            f.write(f"Final Actor Loss: {metrics['training']['actor_loss'][-1]:.4f}\n")
            f.write(f"Final Critic Loss: {metrics['training']['critic_loss'][-1]:.4f}\n")
            f.write(f"Final Policy Loss: {metrics['training']['policy_loss'][-1]:.4f}\n")
            f.write(f"Final Entropy: {metrics['training']['entropy'][-1]:.4f}\n\n")
            
            # Joint Performance
            f.write("Joint Performance:\n")
            final_errors = metrics['joint_errors']['mean'][-1]
            for i, error in enumerate(final_errors):
                f.write(f"Joint {i+1} Final Mean Error: {error:.4f}\n")
            # Include overall and per-joint success rates in the report
            f.write(f"Overall Success Rate: {metrics['success_rate']['overall']:.4f}\n")
            f.write("Joint Success Rates:\n")
            for joint_idx in range(self.num_joints):
                f.write(f"  Joint {joint_idx + 1} Success Rate: {metrics['success_rate']['per_joint'][f'Joint_{joint_idx + 1}_success_rate']:.4f}\n")
            f.write("\n=== End of Report ===\n")

        self.logger.info(f"Training report saved to: {report_path}")


    def _plot_policy_loss(self, metrics: Dict, show_plots: bool) -> None:
        """Plot Policy Loss over episodes."""
        episodes = np.arange(1, len(metrics['training']['policy_loss']) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, metrics['training']['policy_loss'], label='Policy Loss', color='orange')
        plt.xlabel('Episodes')
        plt.ylabel('Policy Loss')
        plt.title('Policy Loss Over Time')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        self.save_figure(plt.gcf(), 'policy_loss')
        self.logger.info("Policy loss plot saved successfully.")
        if show_plots:
            plt.show()
        plt.close()
    
    def _plot_policy_loss_per_agent(self, metrics: Dict, show_plots: bool) -> None:
        """Plot policy loss per agent over episodes."""
        num_agents = metrics['policy_loss_per_agent'].shape[0]
        num_episodes = metrics['policy_loss_per_agent'].shape[1]
        episodes = np.arange(1, num_episodes + 1)
        policy_loss_per_agent = metrics['policy_loss_per_agent']  # Shape: (num_agents, num_episodes)
        plt.figure(figsize=(10, 6))
        fig, axes = plt.subplots(nrows=num_agents, ncols=1, figsize=(30, 6 * num_agents))
        if num_agents == 1:
            axes = [axes]

        for agent_idx in range(num_agents):
            ax = axes[agent_idx]
            ax.plot(episodes, policy_loss_per_agent[agent_idx], label=f'Agent {agent_idx+1} Policy Loss', color='orange')
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Policy Loss')
            ax.set_title(f'Agent {agent_idx+1} Policy Loss Over Time')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        self.save_figure(fig, 'policy_loss_per_agent')
        self.logger.info("Policy loss per agent plot saved successfully.")
        if show_plots:
            plt.show()
        plt.close(fig)
