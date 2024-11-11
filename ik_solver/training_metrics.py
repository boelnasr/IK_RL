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
            
            # Calculate per-agent rewards properly
            total_rewards_per_agent = np.array([log['rewards']['total'] for log in self.logs])  # Shape: (episodes, num_agents)
            
            # Initialize arrays for cumulative rewards
            cumulative_rewards_per_agent = np.zeros((num_agents, num_episodes))
            mean_episode_rewards_per_agent = total_rewards_per_agent.T  # Shape: (num_agents, episodes)
            
            # Calculate cumulative rewards per agent properly
            for agent_idx in range(num_agents):
                running_sum = 0
                for episode in range(num_episodes):
                    running_sum += total_rewards_per_agent[episode][agent_idx]
                    cumulative_rewards_per_agent[agent_idx][episode] = running_sum

            # Calculate success metrics
            agent_success = np.array([log['success'] for log in self.logs])  # Shape: (episodes, num_agents)
            cumulative_success_per_agent = np.zeros((num_agents, num_episodes))
            success_rate_per_agent = np.zeros((num_agents, num_episodes))
            
            # Calculate cumulative success and success rate per agent
            for agent_idx in range(num_agents):
                cumulative_success_per_agent[agent_idx] = np.cumsum(agent_success[:, agent_idx])
                success_rate_per_agent[agent_idx] = cumulative_success_per_agent[agent_idx] / np.arange(1, num_episodes + 1)

            # Calculate overall metrics
            overall_success_rate = np.mean(self.successes)
            policy_loss_per_agent = np.array(self.episode_metrics['policy_loss'])
            average_cumulative_reward = np.mean([cum_rew[-1] for cum_rew in cumulative_rewards_per_agent])
            average_policy_loss = np.mean(self.policy_losses)
            average_joint_error = np.mean(self.joint_errors)

            # Calculate joint-specific success rates
            joint_success_rates = {
                f'Joint_{joint_idx + 1}_success_rate': np.mean(self.joint_successes[joint_idx])
                for joint_idx in range(num_agents)
            }

            # Compile all metrics into a structured dictionary
            metrics = {
                'joint_errors': {
                    'mean': np.array([log['joint_errors']['mean'] for log in self.logs]),
                    'max': np.array([log['joint_errors']['max'] for log in self.logs]),
                    'min': np.array([log['joint_errors']['min'] for log in self.logs]),
                    'std': np.array([log['joint_errors']['std'] for log in self.logs])
                },
                'rewards': {
                    'mean': np.array([log['rewards']['mean'] for log in self.logs]),
                    'total': total_rewards_per_agent,
                    'cumulative': np.array(self.cumulative_rewards),
                    'per_agent_cumulative': cumulative_rewards_per_agent,
                    'per_agent_mean': mean_episode_rewards_per_agent,
                    'final_cumulative': cumulative_rewards_per_agent[:, -1],
                    'average_per_episode': np.mean(total_rewards_per_agent, axis=1)
                },
                'success_rate': {
                    'per_episode': np.array([log['success_rate'] for log in self.logs]),
                    'cumulative': np.cumsum([log['success_rate'] for log in self.logs]) / np.arange(1, num_episodes + 1),
                    'overall': overall_success_rate,
                    'per_joint': joint_success_rates,
                    'per_agent_over_time': success_rate_per_agent
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
                'episodes_to_converge': self.episodes_to_converge,
                'total_training_episodes': self.training_episodes,
                'convergence': {
                    'achieved': self.converged,
                    'episode': self.episodes_to_converge,
                    'threshold': self.convergence_threshold
                }
            }

            # Calculate moving averages if enough episodes
            window = min(10, num_episodes // 10)
            if window > 0:
                metrics['moving_averages'] = {
                    'reward': self._compute_moving_average(metrics['rewards']['average_per_episode'], window),
                    'success_rate': self._compute_moving_average(metrics['success_rate']['per_episode'], window)
                }

                # Add per-agent moving averages
                metrics['moving_averages']['per_agent_rewards'] = np.array([
                    self._compute_moving_average(mean_episode_rewards_per_agent[i], window)
                    for i in range(num_agents)
                ])

            # Add training stability metrics
            if len(self.actor_losses) > 1:
                metrics['training_stability'] = {
                    'actor_loss_std': np.std(self.actor_losses),
                    'critic_loss_std': np.std(self.critic_losses),
                    'policy_loss_std': np.std(self.policy_losses),
                    'reward_std': np.std([log['rewards']['mean'] for log in self.logs]),
                    'entropy_std': np.std(self.entropies)
                }

            self.logger.info("Metrics calculated successfully")
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

    def plot_metrics(self, metrics: Dict, env: Any, show_plots: bool = True) -> None:
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
            self._plot_minimum_joint_errors(metrics, show_plots)  # Add this line


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
        """Plot training-specific metrics, separating entropy and losses."""
        episodes = np.arange(1, len(metrics['training']['entropy']) + 1)
        window = min(10, len(episodes) // 10)
        
        # 1. Plot entropy
        fig_entropy, ax_entropy = plt.subplots(figsize=(12, 6))
        ax_entropy.plot(episodes, metrics['training']['entropy'], 
                    label='Entropy', 
                    color='cyan', 
                    alpha=0.7)
        
        # Add moving average for entropy
        if window > 1:
            entropy_ma = self._compute_moving_average(metrics['training']['entropy'], window)
            ma_episodes = episodes[window-1:]
            ax_entropy.plot(ma_episodes, entropy_ma, 
                        label=f'Moving Average (window={window})', 
                        color='red', 
                        linewidth=2)
        
        ax_entropy.set_title('Policy Entropy Over Time')
        ax_entropy.set_xlabel('Episodes')
        ax_entropy.set_ylabel('Entropy')
        ax_entropy.grid(True, alpha=0.3)
        ax_entropy.legend()
        plt.tight_layout()
        self.save_figure(fig_entropy, 'entropy_plot')
        plt.close(fig_entropy)

        # 2. Plot only actor and critic losses
        fig_losses, ax_losses = plt.subplots(figsize=(12, 6))
        
        # Plot raw data with low alpha
        ax_losses.plot(episodes, metrics['training']['actor_loss'], 
                    label='Actor Loss', 
                    color='blue', 
                    alpha=0.4)
        ax_losses.plot(episodes, metrics['training']['critic_loss'], 
                    label='Critic Loss', 
                    color='red', 
                    alpha=0.4)
        
        # Add moving averages
        if window > 1:
            actor_ma = self._compute_moving_average(metrics['training']['actor_loss'], window)
            critic_ma = self._compute_moving_average(metrics['training']['critic_loss'], window)
            ma_episodes = episodes[window-1:]
            
            ax_losses.plot(ma_episodes, actor_ma, 
                        label='Actor MA', 
                        color='blue', 
                        linestyle='--', 
                        linewidth=2)
            ax_losses.plot(ma_episodes, critic_ma, 
                        label='Critic MA', 
                        color='red', 
                        linestyle='--', 
                        linewidth=2)
        
        # Add trend lines
        for data, color, name in zip(
            [metrics['training']['actor_loss'], metrics['training']['critic_loss']],
            ['blue', 'red'],
            ['Actor', 'Critic']
        ):
            z = np.polyfit(episodes, data, 1)
            p = np.poly1d(z)
            ax_losses.plot(episodes, p(episodes), 
                        color=color, 
                        linestyle=':', 
                        alpha=0.8,
                        label=f'{name} Trend')
        
        # Calculate and display statistics
        stats_text = (
            f'Actor Loss:\n'
            f'  Mean: {np.mean(metrics["training"]["actor_loss"]):.4f}\n'
            f'  Std: {np.std(metrics["training"]["actor_loss"]):.4f}\n'
            f'Critic Loss:\n'
            f'  Mean: {np.mean(metrics["training"]["critic_loss"]):.4f}\n'
            f'  Std: {np.std(metrics["training"]["critic_loss"]):.4f}'
        )
        ax_losses.text(0.02, 0.98, stats_text,
                    transform=ax_losses.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_losses.set_title('Actor and Critic Losses Over Time')
        ax_losses.set_xlabel('Episodes')
        ax_losses.set_ylabel('Loss')
        ax_losses.grid(True, alpha=0.3)
        ax_losses.legend(loc='upper right')
        plt.tight_layout()
        
        self.save_figure(fig_losses, 'actor_critic_losses')
        plt.close(fig_losses)


    def _plot_success_metrics(self, metrics: Dict, env: Any, show_plots: bool) -> None:
        """Plot success-related metrics."""
        episodes = np.arange(1, len(metrics['success_rate']['per_episode']) + 1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(episodes, metrics['success_rate']['cumulative'], label='Success Rate', color='green')
        
        window = min(10, len(episodes) // 10)
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
        """Plot cumulative rewards per agent with improved visualization."""
        episodes = np.arange(1, metrics['rewards']['per_agent_cumulative'].shape[1] + 1)
        num_joints = env.num_joints
        cumulative_rewards = metrics['rewards']['per_agent_cumulative']
        
        fig, axes = plt.subplots(nrows=num_joints, ncols=1, figsize=(30, 6 * num_joints))
        if num_joints == 1:
            axes = [axes]
        
        for joint_idx in range(num_joints):
            ax = axes[joint_idx]
            
            # Plot raw cumulative rewards
            ax.plot(episodes, cumulative_rewards[joint_idx], 
                    label=f'Joint {joint_idx+1} Cumulative Reward', color='b')
            
            # Add moving average trend line
            window = min(10, len(episodes) // 10) if len(episodes) > 50 else 1
            if window > 1:
                rolling_mean = np.convolve(cumulative_rewards[joint_idx], 
                                        np.ones(window)/window, mode='valid')
                ax.plot(episodes[window-1:], rolling_mean, 
                    label=f'Moving Average (window={window})', 
                    color='r', linestyle='--')
            
            # Calculate and display reward rate
            total_reward = cumulative_rewards[joint_idx][-1]
            reward_rate = total_reward / len(episodes)
            ax.text(0.02, 0.98, f'Average Reward Rate: {reward_rate:.2f}/episode\nTotal Reward: {total_reward:.2f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Cumulative Reward')
            ax.set_title(f'Joint {joint_idx+1} Cumulative Reward Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'cumulative_rewards_per_agent')
        if show_plots:
            plt.show()
        plt.close(fig)

    def _plot_mean_episode_rewards_per_agent(self, metrics: Dict, env: Any, show_plots: bool) -> None:
        """
        Plot mean episode rewards per agent (joint) over episodes with moving average and trend line.

        Args:
            metrics (Dict): Dictionary containing calculated training metrics.
            env (Any): Environment instance containing relevant attributes like `num_joints`.
            show_plots (bool): Flag to display plots interactively.
        """
        try:
            # Extract necessary data
            episodes = np.arange(1, metrics['mean_episode_rewards_per_agent'].shape[1] + 1)
            num_joints = env.num_joints if env else self.num_joints
            mean_episode_rewards_per_agent = metrics['mean_episode_rewards_per_agent']
            
            # Create subplots for each joint
            fig, axes = plt.subplots(nrows=num_joints, ncols=1, figsize=(30, 6 * num_joints))
            if num_joints == 1:
                axes = [axes]
            
            # Define moving average window size
            window = min(10, len(episodes) // 10)  # Adjust window based on the number of episodes
            window = max(window, 1)  # Ensure window is at least 1
            
            for joint_idx in range(num_joints):
                ax = axes[joint_idx]
                joint_rewards = mean_episode_rewards_per_agent[joint_idx]
                
                # Plot raw mean episode rewards
                ax.plot(
                    episodes, 
                    joint_rewards, 
                    label=f'Joint {joint_idx + 1} Mean Reward', 
                    color='green', 
                    linewidth=2
                )
                
                # Compute and plot moving average if window > 1
                if window > 1 and len(joint_rewards) >= window:
                    moving_avg = self._compute_moving_average(joint_rewards, window)
                    ma_episodes = episodes[window - 1:]
                    ax.plot(
                        ma_episodes, 
                        moving_avg, 
                        label=f'Moving Average (window={window})', 
                        color='red', 
                        linewidth=2, 
                        linestyle='--'
                    )
                
                # Compute and plot trend line using linear regression
                if len(joint_rewards) >= 2:
                    z = np.polyfit(episodes, joint_rewards, 1)  # Linear fit
                    p = np.poly1d(z)
                    ax.plot(
                        episodes, 
                        p(episodes), 
                        label='Trend', 
                        color='blue', 
                        linewidth=2, 
                        linestyle=':'
                    )
                
                # Customize plot
                ax.set_xlabel('Episodes', fontsize=12)
                ax.set_ylabel('Mean Episode Reward', fontsize=12)
                ax.set_title(f'Joint {joint_idx + 1} Mean Episode Reward Over Time', fontsize=14)
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.save_figure(fig, 'mean_episode_rewards_per_joint')
            self.logger.info("Mean episode rewards per joint plot saved successfully.")
            if show_plots:
                plt.show()
            plt.close(fig)
        
        except Exception as e:
            self.logger.error(f"Error plotting mean episode rewards per agent: {str(e)}")
            raise


    def _plot_minimum_joint_errors(self, metrics: Dict, show_plots: bool) -> None:
        """
        Plot minimum joint errors per episode for each joint in subplots.
        
        Args:
            metrics: Dictionary containing training metrics
            show_plots: Boolean to control whether to display plots
        """
        try:
            # Extract joint errors from metrics
            joint_errors = np.array(metrics['joint_errors']['min'])  # Shape: (num_episodes, num_joints)
            num_episodes, num_joints = joint_errors.shape
            episodes = np.arange(1, num_episodes + 1)

            # Create figure with subplots
            fig, axes = plt.subplots(num_joints, 1, figsize=(15, 4 * num_joints))
            if num_joints == 1:
                axes = [axes]

            # Color scheme for better visualization
            colors = plt.cm.viridis(np.linspace(0, 1, 3))  # 3 colors for different lines

            # Plot for each joint
            for joint_idx in range(num_joints):
                ax = axes[joint_idx]
                errors = joint_errors[:, joint_idx]
                
                # Calculate rolling minimum
                window = min(10, len(episodes) // 10) if len(episodes) > 50 else 1
                min_errors = np.array([np.min(errors[max(0, i-window):i+1]) 
                                    for i in range(len(errors))])
                
                # Plot raw minimum errors
                ax.plot(episodes, min_errors, 
                    label='Minimum Error', 
                    color=colors[0], 
                    alpha=0.5)
                
                # Add moving average of minimum errors
                if window > 1:
                    moving_avg = np.convolve(min_errors, 
                                        np.ones(window)/window, 
                                        mode='valid')
                    ax.plot(episodes[window-1:], 
                        moving_avg,
                        label=f'Moving Average (window={window})',
                        color=colors[1],
                        linewidth=2)

                # Add trend line
                z = np.polyfit(episodes, min_errors, 1)
                p = np.poly1d(z)
                ax.plot(episodes, p(episodes), 
                    linestyle='--', 
                    color=colors[2],
                    label='Trend',
                    alpha=0.8)

                # Calculate and display statistics
                stats = {
                    'Min': np.min(min_errors),
                    'Mean': np.mean(min_errors),
                    'Std': np.std(min_errors),
                    'Final': min_errors[-1],
                    'Improvement': ((min_errors[0] - min_errors[-1]) / min_errors[0] * 100 
                                if min_errors[0] != 0 else 0)
                }
                
                stats_text = (f"Minimum Error: {stats['Min']:.6f}\n"
                            f"Mean Error: {stats['Mean']:.6f}\n"
                            f"Std Dev: {stats['Std']:.6f}\n"
                            f"Final Error: {stats['Final']:.6f}\n"
                            f"Improvement: {stats['Improvement']:.1f}%")
                
                ax.text(0.02, 0.98, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                # Highlight key points
                lowest_error = np.min(min_errors)
                lowest_error_idx = np.argmin(min_errors)
                ax.scatter(episodes[lowest_error_idx], lowest_error, 
                        color='red', s=100, zorder=5,
                        label='Best Performance')

                # Customize subplot
                ax.set_xlabel('Episodes')
                ax.set_ylabel('Joint Error (radians)')
                ax.set_title(f'Joint {joint_idx + 1}  Error Over Time')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Set y-axis to log scale if range is large
                if np.max(min_errors) / (np.min(min_errors) + 1e-10) > 100:
                    ax.set_yscale('log')

            plt.tight_layout()
            self.save_figure(fig, 'minimum_joint_errors')
            if show_plots:
                plt.show()
            plt.close(fig)

            # Create summary comparison plot
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot smoothed minimum errors for all joints
            for joint_idx in range(num_joints):
                errors = joint_errors[:, joint_idx]
                window = min(10, len(episodes) // 10) if len(episodes) > 50 else 1
                
                # Calculate smoothed minimum errors
                min_errors = np.array([np.min(errors[max(0, i-window):i+1]) 
                                    for i in range(len(errors))])
                moving_avg = np.convolve(min_errors, 
                                    np.ones(window)/window, 
                                    mode='valid')
                
                ax.plot(episodes[window-1:], 
                    moving_avg,
                    label=f'Joint {joint_idx + 1}',
                    linewidth=2)

            # Add overall error trend
            mean_errors = np.mean(joint_errors, axis=1)
            z = np.polyfit(episodes, mean_errors, 1)
            p = np.poly1d(z)
            ax.plot(episodes, p(episodes), 
                linestyle='--', 
                color='black',
                label='Overall Trend',
                alpha=0.5)

            ax.set_xlabel('Episodes')
            ax.set_ylabel('Minimum Joint Error (radians)')
            ax.set_title('Comparison of Minimum Joint Errors Across All Joints')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Set y-axis to log scale if range is large
            if np.max(joint_errors) / (np.min(joint_errors) + 1e-10) > 100:
                ax.set_yscale('log')

            plt.tight_layout()
            self.save_figure(fig, 'minimum_joint_errors_comparison')
            if show_plots:
                plt.show()
            plt.close(fig)

            # Additional convergence analysis plot
            fig, ax = plt.subplots(figsize=(15, 6))
            
            # Calculate convergence metrics for each joint
            for joint_idx in range(num_joints):
                errors = joint_errors[:, joint_idx]
                window = min(10, len(episodes) // 5)
                convergence = np.array([np.std(errors[max(0, i-window):i+1]) 
                                    for i in range(len(errors))])
                
                ax.plot(episodes, convergence, 
                    label=f'Joint {joint_idx + 1}',
                    alpha=0.7)

            ax.set_xlabel('Episodes')
            ax.set_ylabel('Error Stability (Std Dev)')
            ax.set_title('Joint Error Convergence Analysis')
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            self.save_figure(fig, 'joint_errors_convergence')
            if show_plots:
                plt.show()
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error plotting minimum joint errors: {str(e)}")
            raise
    
    
    
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
        """Plot Policy Loss over episodes with smoothing."""
        episodes = np.arange(1, len(metrics['training']['policy_loss']) + 1)
        raw_policy_loss = np.array(metrics['training']['policy_loss'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot raw data with low alpha
        ax.plot(episodes, raw_policy_loss, 
                label='Raw Policy Loss', 
                color='orange', 
                alpha=0.3, 
                linewidth=1)
        
        # Add different smoothing levels
        windows = [5, 20, 50]  # Different window sizes for smoothing
        colors = ['red', 'blue', 'green']  # Different colors for each smoothing level
        
        for window, color in zip(windows, colors):
            if len(episodes) > window:
                # Compute exponential moving average
                alpha = 2 / (window + 1)
                smoothed = np.zeros_like(raw_policy_loss)
                smoothed[0] = raw_policy_loss[0]
                for i in range(1, len(raw_policy_loss)):
                    smoothed[i] = alpha * raw_policy_loss[i] + (1 - alpha) * smoothed[i-1]
                
                ax.plot(episodes, smoothed, 
                    label=f'EMA (window={window})', 
                    color=color, 
                    linewidth=2)
        
        # Add trend line
        z = np.polyfit(episodes, raw_policy_loss, 1)
        p = np.poly1d(z)
        ax.plot(episodes, p(episodes), 
                '--', 
                color='purple', 
                label='Trend', 
                linewidth=2)
        
        # Calculate and display statistics
        stats_text = (
            f'Mean: {np.mean(raw_policy_loss):.4f}\n'
            f'Std: {np.std(raw_policy_loss):.4f}\n'
            f'Min: {np.min(raw_policy_loss):.4f}\n'
            f'Max: {np.max(raw_policy_loss):.4f}'
        )
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss Over Time (with Smoothing)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        self.save_figure(fig, 'policy_loss_smoothed')
        self.logger.info("Smoothed policy loss plot saved successfully.")
        if show_plots:
            plt.show()
        plt.close(fig)

    def _plot_policy_loss_per_agent(self, metrics: Dict, show_plots: bool) -> None:
            """Plot policy loss per agent over episodes with smoothing."""
            try:
                num_agents = metrics['policy_loss_per_agent'].shape[0]
                num_episodes = metrics['policy_loss_per_agent'].shape[1]
                episodes = np.arange(1, num_episodes + 1)
                policy_loss_per_agent = metrics['policy_loss_per_agent']
                
                fig, axes = plt.subplots(nrows=num_agents, ncols=1, figsize=(30, 6 * num_agents))
                if num_agents == 1:
                    axes = [axes]

                windows = [5, 20, 50]  # Different window sizes for smoothing
                colors = ['red', 'blue', 'green']  # Different colors for each smoothing level
                
                for agent_idx in range(num_agents):
                    ax = axes[agent_idx]
                    raw_data = policy_loss_per_agent[agent_idx]
                    
                    # Plot raw data with low alpha
                    ax.plot(episodes, raw_data, 
                            label='Raw Data', 
                            color='orange', 
                            alpha=0.7, 
                            linewidth=1)
                    
                    # Add different smoothing levels
                    for window, color in zip(windows, colors):
                        if len(episodes) > window:
                            # Compute exponential moving average
                            alpha_val = 2 / (window + 1)
                            smoothed = np.zeros_like(raw_data)
                            smoothed[0] = raw_data[0]
                            for i in range(1, len(raw_data)):
                                smoothed[i] = alpha_val * raw_data[i] + (1 - alpha_val) * smoothed[i-1]
                            
                            ax.plot(episodes, smoothed, 
                                label=f'EMA (window={window})', 
                                color=color, 
                                linewidth=2)
                    
                    # Add trend line
                    z = np.polyfit(episodes, raw_data, 1)
                    p = np.poly1d(z)
                    ax.plot(episodes, p(episodes), 
                            '--', 
                            color='purple', 
                            label='Trend', 
                            linewidth=2)
                    
                    # Calculate and display statistics
                    stats_text = (
                        f'Mean: {np.mean(raw_data):.4f}\n'
                        f'Std: {np.std(raw_data):.4f}\n'
                        f'Min: {np.min(raw_data):.4f}\n'
                        f'Max: {np.max(raw_data):.4f}'
                    )
                    ax.text(0.02, 0.98, stats_text,
                            transform=ax.transAxes,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax.set_xlabel('Episodes')
                    ax.set_ylabel('Policy Loss')
                    ax.set_title(f'Agent {agent_idx+1} Policy Loss Over Time')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper right')
                
                plt.tight_layout()
                self.save_figure(fig, 'policy_loss_per_agent_smoothed')
                self.logger.info("Smoothed policy loss per agent plot saved successfully.")
                if show_plots:
                    plt.show()
                plt.close(fig)
            
            except Exception as e:
                self.logger.error(f"Error plotting policy loss per agent: {str(e)}")
                raise

    def plot_distance_metrics(self):
        """
        Create comprehensive visualizations for distance metrics.
        """
        try:
            # Create directory for plots if it doesn't exist
            plot_dir = "distance_metrics_plots"
            os.makedirs(plot_dir, exist_ok=True)

            # Setup subplots
            fig = plt.figure(figsize=(20, 12))
            gs = plt.GridSpec(3, 2, figure=fig)

            # 1. Position Distance Over Time
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_position_distance(ax1)

            # 2. Orientation Distance Over Time
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_orientation_distance(ax2)

            # 3. Component-wise Position Distances
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_component_distances(ax3)

            # 4. Euler Angle Distances
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_euler_distances(ax4)

            # 5. Task Completion Progress
            ax5 = fig.add_subplot(gs[2, 0])
            self._plot_completion_progress(ax5)

            # 6. Combined Metrics
            ax6 = fig.add_subplot(gs[2, 1])
            self._plot_combined_metrics(ax6)

            plt.tight_layout()
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            plt.savefig(os.path.join(plot_dir, f'distance_metrics_{timestamp}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logging.error(f"Error in plot_distance_metrics: {e}")

    def _plot_position_distance(self, ax):
        """Plot position distance metrics."""
        steps = range(len(self.distance_history['position']['euclidean']))
        
        # Plot raw distance
        ax.plot(steps, self.distance_history['position']['euclidean'], 
                label='Euclidean Distance', color='blue', alpha=0.6)
        
        # Add moving average
        window = min(20, len(steps))
        if window > 1:
            moving_avg = np.convolve(self.distance_history['position']['euclidean'], 
                                np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], moving_avg, 
                    label=f'Moving Average (n={window})', 
                    color='red', linewidth=2)
        
        ax.set_title('Position Distance Over Time')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Distance (m)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_orientation_distance(self, ax):
        """Plot orientation distance metrics."""
        steps = range(len(self.distance_history['orientation']['quaternion']))
        
        # Plot quaternion distance
        ax.plot(steps, self.distance_history['orientation']['quaternion'], 
                label='Quaternion Distance', color='green', alpha=0.6)
        
        # Add moving average
        window = min(20, len(steps))
        if window > 1:
            moving_avg = np.convolve(self.distance_history['orientation']['quaternion'], 
                                np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], moving_avg, 
                    label=f'Moving Average (n={window})', 
                    color='red', linewidth=2)
        
        ax.set_title('Orientation Distance Over Time')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Distance (rad)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_component_distances(self, ax):
        """Plot component-wise position distances."""
        steps = range(len(self.distance_history['position']['components']['x']))
        
        # Plot each component
        components = ['x', 'y', 'z']
        colors = ['red', 'green', 'blue']
        
        for component, color in zip(components, colors):
            ax.plot(steps, self.distance_history['position']['components'][component], 
                    label=f'{component.upper()} Distance', 
                    color=color, alpha=0.7)
        
        ax.set_title('Component-wise Position Distances')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Distance (m)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_euler_distances(self, ax):
        """Plot Euler angle distances."""
        steps = range(len(self.distance_history['orientation']['euler']['roll']))
        
        # Plot each Euler angle
        angles = ['roll', 'pitch', 'yaw']
        colors = ['red', 'green', 'blue']
        
        for angle, color in zip(angles, colors):
            ax.plot(steps, self.distance_history['orientation']['euler'][angle], 
                    label=f'{angle.capitalize()}', 
                    color=color, alpha=0.7)
        
        ax.set_title('Euler Angle Distances')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Distance (rad)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_completion_progress(self, ax):
        """Plot task completion progress."""
        steps = range(len(self.completion_history['overall']))
        
        metrics = ['overall', 'position', 'orientation']
        colors = ['purple', 'blue', 'green']
        
        for metric, color in zip(metrics, colors):
            ax.plot(steps, self.completion_history[metric], 
                    label=f'{metric.capitalize()} Completion', 
                    color=color, alpha=0.7)
        
        ax.set_title('Task Completion Progress')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Completion (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 100])

    def _plot_combined_metrics(self, metrics: Dict, show_plots: bool) -> None:
        """Plot combined performance metrics."""
        try:
            episodes = np.arange(1, len(metrics['rewards']['average_per_episode']) + 1)
            # Normalize the metrics
            # Example normalization: scale each metric between 0 and 1
            rewards = metrics['rewards']['average_per_episode']
            success_rate = metrics['success_rate']['per_episode']
            joint_error_mean = np.mean(metrics['joint_errors']['mean'], axis=1)
            
            # Normalize rewards
            norm_rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards) + 1e-8)
            # Normalize success rate (already between 0 and 1)
            norm_success_rate = success_rate
            # Normalize joint error mean
            norm_joint_error = 1.0 - (joint_error_mean - np.min(joint_error_mean)) / (np.max(joint_error_mean) - np.min(joint_error_mean) + 1e-8)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(episodes, norm_rewards, label='Normalized Reward', color='blue')
            ax.plot(episodes, norm_success_rate, label='Success Rate', color='green')
            ax.plot(episodes, norm_joint_error, label='Normalized Accuracy', color='red')
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Normalized Value')
            ax.set_title('Combined Performance Metrics')
            ax.grid(True, alpha=0.3)
            ax.legend()

            self.save_figure(fig, 'combined_metrics')
            if show_plots:
                plt.show()
            plt.close(fig)
        
        except Exception as e:
            self.logger.error(f"Error plotting combined metrics: {str(e)}")
            raise