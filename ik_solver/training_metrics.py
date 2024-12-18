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
            'advantages': [],
            'policy_loss': [],
            'overall_actor_loss': []
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

        # Global plot configurations
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'lines.linewidth': 1.5,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'text.usetex': False,  # Disable LaTeX rendering
            # 'text.latex.preamble': r'\usepackage{amsmath}'  # Commented out
        })

        # Define colorblind-friendly palettes
        self.color_palette = plt.cm.get_cmap('tab10')  # For discrete data
        self.sequential_palette = plt.cm.get_cmap('viridis')  # For gradients


    def log_episode(self, 
                    joint_errors: List[List[float]], 
                    rewards: List[List[float]], 
                    success: List[bool],
                    entropy: float, 
                    actor_loss: float, 
                    critic_loss: float, 
                    policy_loss: List[float],
                    advantages: np.ndarray, 
                    actor_loss_per_actor: List[float],
                    env: Any, 
                    success_threshold: float) -> None:
        """
        Log comprehensive metrics for a single episode.

        Args:
            joint_errors: List of joint errors over time (time_steps x num_agents).
            rewards: List of rewards per time step per agent for the episode (time_steps x num_agents).
            success: Success status per agent (num_agents, boolean values).
            entropy: Policy entropy value (float).
            actor_loss: Overall actor network loss (scalar float).
            critic_loss: Critic network loss (scalar float).
            policy_loss: List of policy losses per agent (length = num_agents).
            advantages: Numpy array of advantages calculated during the episode (time_steps,).
            actor_loss_per_actor: List of actor losses per agent (length = num_agents, each a float).
            env: Environment instance for additional metrics.
            success_threshold: Threshold for determining success in the episode (float).
        """
        try:
            # Convert inputs to numpy arrays for consistency
            joint_errors = np.array(joint_errors)  # Shape: (time_steps, num_agents)
            rewards = np.array(rewards)            # Shape: (time_steps, num_agents)
            success = np.array(success)            # Shape: (num_agents,)
            advantages = np.array(advantages)      # Shape: (time_steps,)

            # Ensure joint_errors is 2D: (time_steps, num_agents)
            if joint_errors.ndim == 1:
                joint_errors = joint_errors.reshape(-1, self.num_joints)

            # Ensure rewards is 2D: (time_steps, num_agents)
            if rewards.ndim == 1:
                rewards = rewards.reshape(-1, self.num_joints)

            # Ensure success is 1D: (num_agents,)
            success = np.atleast_1d(success)

            # Calculate per-agent total rewards
            total_rewards_per_agent = np.sum(rewards, axis=0)  # Shape: (num_agents,)
            total_rewards_per_agent = np.atleast_1d(total_rewards_per_agent)

            # Construct the episode_data dictionary
            episode_data = {
                'joint_errors': {
                    'mean': np.mean(joint_errors, axis=0).tolist(),
                    'max': np.max(joint_errors, axis=0).tolist(),
                    'min': np.min(joint_errors, axis=0).tolist(),
                    'std': np.std(joint_errors, axis=0).tolist()
                },
                'rewards': {
                    'total': total_rewards_per_agent.tolist(),
                    'mean': float(np.mean(rewards)),
                    'std': float(np.std(rewards)),
                    'std_per_agent': np.std(rewards, axis=0).tolist()
                },
                'success': success.tolist(),
                'success_rate': float(np.mean(success)),
                'entropy': float(entropy),
                'overall_actor_loss': float(actor_loss),  # Make sure this is present
                'critic_loss': float(critic_loss),
                'policy_loss': policy_loss,               # List of floats, one per agent
                'advantages': advantages.tolist(),
                'episode_length': int(joint_errors.shape[0]),
                'success_threshold': float(success_threshold),
                'actor_loss_per_actor': actor_loss_per_actor  # List of per-actor losses
            }

            # Add the episode data to logs
            self.logs.append(episode_data)

            # Update the running episode metrics
            self._update_episode_metrics(episode_data)

            # Update other tracked variables
            self.episode_rewards.append(np.sum(total_rewards_per_agent))
            self.successes.append(np.any(success))
            self.policy_losses.append(np.mean(policy_loss))
            self.actor_losses.append(actor_loss)
            self.critic_losses.append(critic_loss)
            self.entropies.append(entropy)
            self.joint_errors.append(np.mean(joint_errors))

            # Update cumulative rewards
            cumulative_reward = (np.sum(total_rewards_per_agent)
                                if not self.cumulative_rewards 
                                else self.cumulative_rewards[-1] + np.sum(total_rewards_per_agent))
            self.cumulative_rewards.append(cumulative_reward)
            self.training_episodes += 1

            # Track success per joint
            for joint_idx, joint_success in enumerate(success):
                self.joint_successes[joint_idx].append(joint_success)

            # Check for convergence if enough episodes have passed
            if not self.converged and len(self.successes) >= 100:
                recent_success_rate = np.mean(self.successes[-100:])
                if recent_success_rate >= self.convergence_threshold:
                    self.converged = True
                    self.episodes_to_converge = self.training_episodes
                    self.logger.info(
                        f"Convergence achieved at episode {self.training_episodes} "
                        f"with success rate {recent_success_rate:.4f}"
                    )

            # Log episode details including the success threshold
            self.logger.info(
                f"Episode {self.training_episodes} logged successfully. "
                f"Mean reward: {episode_data['rewards']['mean']:.4f}, "
                f"Success rate: {episode_data['success_rate']:.4f}, "
                f"Success threshold: {success_threshold:.4f}"
            )

        except Exception as e:
            self.logger.error(f"Error logging episode: {str(e)}")
            raise


    def _update_episode_metrics(self, episode_data: Dict) -> None:
        """Update running metrics with new episode data."""
        self.episode_metrics['joint_errors'].append(episode_data['joint_errors']['mean'])
        self.episode_metrics['rewards'].append(episode_data['rewards']['total'])
        self.episode_metrics['success_rates'].append(episode_data['success_rate'])
        self.episode_metrics['entropy'].append(episode_data['entropy'])
        # Append per-actor losses (list of floats, one per agent)
        self.episode_metrics['actor_loss'].append(episode_data['actor_loss_per_actor'])
        # Append overall actor loss (single float)
        self.episode_metrics['overall_actor_loss'].append(episode_data['overall_actor_loss'])
        self.episode_metrics['critic_loss'].append(episode_data['critic_loss'])
        self.episode_metrics['policy_loss'].append(episode_data['policy_loss'])
        self.episode_metrics['advantages'].append(episode_data['advantages'])


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
            
            # Extract total rewards per agent for each episode
            total_rewards_per_agent = np.array([log['rewards']['total'] for log in self.logs])  # Shape: (episodes, num_agents)
            if total_rewards_per_agent.ndim == 1:
                total_rewards_per_agent = total_rewards_per_agent.reshape(-1, 1)
                
            # Per-agent cumulative rewards
            cumulative_rewards_per_agent = np.zeros((num_agents, num_episodes))
            mean_episode_rewards_per_agent = total_rewards_per_agent.T  # Shape: (num_agents, episodes)
            for agent_idx in range(num_agents):
                cumulative_rewards_per_agent[agent_idx] = np.cumsum(total_rewards_per_agent[:, agent_idx])

            # Extract success data
            agent_success = np.array([log['success'] for log in self.logs])  # Shape: (episodes, num_agents)
            if agent_success.ndim == 1:
                agent_success = agent_success.reshape(-1, 1)
            
            cumulative_success_per_agent = np.zeros((num_agents, num_episodes))
            success_rate_per_agent = np.zeros((num_agents, num_episodes))
            for agent_idx in range(num_agents):
                cumulative_success_per_agent[agent_idx] = np.cumsum(agent_success[:, agent_idx])
                success_rate_per_agent[agent_idx] = cumulative_success_per_agent[agent_idx] / np.arange(1, num_episodes + 1)

            overall_success_rate = np.mean(self.successes)
            average_cumulative_reward = np.mean([cum_rew[-1] for cum_rew in cumulative_rewards_per_agent])
            average_policy_loss = np.mean(self.policy_losses)
            average_joint_error = np.mean(self.joint_errors)

            # Calculate joint-specific success rates
            joint_success_rates = {
                f'Joint_{joint_idx + 1}_success_rate': np.mean(self.joint_successes[joint_idx])
                for joint_idx in range(num_agents)
            }

            # Convert stored metrics into arrays
            # policy_loss is a list of lists, each inner list corresponds to a single episode
            policy_loss_per_agent = np.array(self.episode_metrics['policy_loss'])   # Shape: (episodes, num_agents)
            actor_loss_per_agent = np.array(self.episode_metrics['actor_loss'])     # Shape: (episodes, num_agents)
            overall_actor_loss_array = np.array(self.episode_metrics['overall_actor_loss'])  # Shape: (episodes,)
            critic_loss_array = np.array(self.episode_metrics['critic_loss'])       # Shape: (episodes,)

            # Compute mean actor loss per episode (averaged over agents)
            actor_loss_array = np.array([np.mean(al) for al in actor_loss_per_agent])

            # Extract joint error statistics
            joint_errors_mean = np.array([log['joint_errors']['mean'] for log in self.logs])
            joint_errors_max = np.array([log['joint_errors']['max'] for log in self.logs])
            joint_errors_min = np.array([log['joint_errors']['min'] for log in self.logs])
            joint_errors_std = np.array([log['joint_errors']['std'] for log in self.logs])

            # Compile all metrics into a structured dictionary
            metrics = {
                'joint_errors': {
                    'mean': joint_errors_mean,
                    'max': joint_errors_max,
                    'min': joint_errors_min,
                    'std': joint_errors_std
                },
                'rewards': {
                    'mean': np.array([log['rewards']['mean'] for log in self.logs]),
                    'total': total_rewards_per_agent,
                    'cumulative': np.array(self.cumulative_rewards),
                    'per_agent_cumulative': cumulative_rewards_per_agent,
                    'per_agent_mean': mean_episode_rewards_per_agent,
                    'final_cumulative': cumulative_rewards_per_agent[:, -1],
                    'average_per_episode': np.mean(total_rewards_per_agent, axis=1),
                    'std_per_episode': np.array([log['rewards']['std'] for log in self.logs]),
                    'std_per_agent_per_episode': np.array([log['rewards']['std_per_agent'] for log in self.logs])
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
                    'actor_loss': actor_loss_array,          # Mean per-actor loss
                    'overall_actor_loss': overall_actor_loss_array,  # Overall actor loss
                    'critic_loss': critic_loss_array,
                    'policy_loss': np.mean(policy_loss_per_agent, axis=1),
                    # 'advantages': np.array(self.episode_metrics['advantages'])
                },
                'agent_success': agent_success,
                'cumulative_rewards_per_agent': cumulative_rewards_per_agent,
                'mean_episode_rewards_per_agent': mean_episode_rewards_per_agent,
                'success_rate_per_agent': success_rate_per_agent,
                'policy_loss_per_agent': policy_loss_per_agent.T,
                'actor_loss_per_agent': actor_loss_per_agent.T,
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

            # Compute moving averages if enough episodes
            window = min(10, num_episodes // 10)
            if window > 0:
                metrics['moving_averages'] = {
                    'reward': self._compute_moving_average(metrics['rewards']['average_per_episode'], window),
                    'success_rate': self._compute_moving_average(metrics['success_rate']['per_episode'], window),
                    'per_agent_rewards': np.array([
                        self._compute_moving_average(mean_episode_rewards_per_agent[i], window)
                        for i in range(num_agents)
                    ])
                }

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
        """
        Save figure with high-quality settings in multiple formats.

        Args:
            fig: Matplotlib figure object to save.
            name: Name of the figure file (without extension).
        """
        # Define file paths
        png_path = os.path.join(self.base_path, f"{name}.png")
        pdf_path = os.path.join(self.base_path, f"{name}.pdf")
        
        # Save in both PNG and PDF formats
        fig.savefig(png_path, dpi=300, bbox_inches='tight')  # High-quality PNG
       #fig.savefig(pdf_path, format='pdf', bbox_inches='tight')  # Vector PDF
        plt.close(fig)

        self.logger.info(f"Saved figure '{name}' as PNG and PDF.")

    def plot_metrics(self, metrics: Dict, env: Any, show_plots: bool = True) -> None:
        """Create and save all plots."""
        try:
            print("Rewards mean:", metrics['rewards']['mean'])
            print("Per-agent mean rewards:", metrics['rewards']['per_agent_mean'])

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
            self._plot_minimum_joint_errors(metrics, show_plots)
            self._plot_normalized_reward(metrics, show_plots)
            self._plot_normalized_reward_per_agent(metrics, show_plots)
            self._plot_normalized_cumulative_reward_per_agent(metrics, show_plots)
            # self._plot_advantages(metrics, show_plots)
            self._plot_value_loss(metrics, show_plots)
            self._plot_convergence(metrics, show_plots)  # Added convergence plot
            self._plot_reward_std_over_time(metrics, show_plots)
            self._plot_reward_std_per_agent(metrics, show_plots)
            self.plot_joint_error_convergence(metrics, show_plots,num_joints=self.num_joints)
            self._plot_entropy(metrics, show_plots)
            self._plot_actor_loss_per_actor(metrics, show_plots)


            self.logger.info("All plots generated and saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {str(e)}")
            raise

 # Add the missing save_final_report method
    def save_final_report(self, metrics: Dict) -> None:
        """
        Generate and save a comprehensive training report.

        Args:
            metrics: Dictionary containing all training metrics
        """
        report_path = os.path.join(self.base_path, 'training_report.txt')

        try:
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
                    f.write(f"Joint {i + 1} Final Mean Error: {error:.4f}\n")
                f.write("\nJoint Success Rates:\n")
                for joint_idx in range(self.num_joints):
                    success_rate = metrics['success_rate']['per_joint'][f'Joint_{joint_idx + 1}_success_rate']
                    f.write(f"  Joint {joint_idx + 1} Success Rate: {success_rate:.4f}\n")
                f.write("\n=== End of Report ===\n")

            self.logger.info(f"Training report saved to: {report_path}")

        except Exception as e:
            self.logger.error(f"Error saving final report: {str(e)}")
            raise


    def _plot_joint_metrics(self, metrics: Dict, env: Any, show_plots: bool) -> None:
        """Plot joint-specific metrics with enhanced styling for research papers."""
        num_joints = env.num_joints
        episodes = np.arange(1, len(metrics['joint_errors']['mean']) + 1)

        # Determine subplot grid dimensions
        if num_joints % 2 == 0:
            ncols = 2
            nrows = num_joints // 2
        else:
            ncols = 1
            nrows = num_joints

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 3 * nrows), dpi=300)
        axes = np.atleast_1d(axes).flatten()

        colors = self.color_palette

        for joint_idx in range(num_joints):
            ax = axes[joint_idx]
            mean_errors = metrics['joint_errors']['mean'][:, joint_idx]
            std_errors = metrics['joint_errors']['std'][:, joint_idx]

            # Plot mean error with shaded standard deviation
            ax.plot(episodes, mean_errors, label='Mean Error', color=colors(joint_idx % 10), linewidth=1.5)
            ax.fill_between(episodes,
                            mean_errors - std_errors,
                            mean_errors + std_errors,
                            alpha=0.2,
                            color=colors(joint_idx % 10),
                            label='Standard Deviation')

            # Enhance labels and titles
            ax.set_xlabel('Episodes', fontsize=12)
            ax.set_ylabel('Joint Error (radians)', fontsize=12)
            ax.set_title(f'Joint {joint_idx + 1} Error Metrics', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

        # Remove any unused subplots
        for idx in range(num_joints, nrows * ncols):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        self.save_figure(fig, 'joint_metrics')
        if show_plots:
            plt.show()
        plt.close(fig)


    def _plot_per_joint_cumulative_rewards(self, metrics: Dict, env: Any, show_plots: bool) -> None:
        """Plot cumulative rewards for each joint with enhanced styling."""
        episodes = np.arange(1, metrics['cumulative_rewards_per_agent'].shape[1] + 1)
        num_joints = env.num_joints
        cumulative_rewards_per_agent = metrics['cumulative_rewards_per_agent']

        # Determine subplot grid dimensions
        if num_joints % 2 == 0:
            ncols = 2
            nrows = num_joints // 2
        else:
            ncols = 1
            nrows = num_joints

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 3 * nrows), dpi=300)
        axes = np.atleast_1d(axes).flatten()

        colors = self.color_palette

        for joint_idx in range(num_joints):
            ax = axes[joint_idx]
            rewards = cumulative_rewards_per_agent[joint_idx]

            ax.plot(episodes, rewards, label=f'Joint {joint_idx + 1}', color=colors(joint_idx % 10), linewidth=1.5)

            # Apply moving average
            window = max(1, len(episodes) // 50)
            if window > 1:
                rewards_smooth = np.convolve(rewards, np.ones(window) / window, mode='valid')
                ax.plot(episodes[window - 1:], rewards_smooth, label='Moving Average', color='tab:red', linewidth=2, linestyle='--')

            ax.set_xlabel('Episodes', fontsize=12)
            ax.set_ylabel('Cumulative Reward', fontsize=12)
            ax.set_title(f'Joint {joint_idx + 1} Cumulative Reward Over Time', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

        # Remove any unused subplots
        for idx in range(num_joints, nrows * ncols):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        self.save_figure(fig, 'cumulative_rewards_per_joint')
        if show_plots:
            plt.show()
        plt.close(fig)


    def _plot_training_metrics(self, metrics: Dict, show_plots: bool) -> None:
        """Plot training losses over episodes with enhanced styling."""
        episodes = np.arange(1, len(metrics['training']['actor_loss']) + 1)
        window = min(10, len(episodes) // 10)
        actor_loss = metrics['training']['actor_loss']
        critic_loss = metrics['training']['critic_loss']
    
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
    def _plot_entropy(self, metrics: Dict, show_plots: bool) -> None:
        """Plot entropy over episodes with enhanced styling."""
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
            print(f"Entropy MA: {entropy_ma}")
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

    def _plot_success_metrics(self, metrics: Dict, env: Any, show_plots: bool) -> None:
        """Plot success-related metrics with enhanced styling."""
        episodes = np.arange(1, len(metrics['success_rate']['per_episode']) + 1)
        success_rate = metrics['success_rate']['cumulative']

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.plot(episodes, success_rate, label='Success Rate', color='tab:green', linewidth=1.5)

        # Apply moving average for smoothing
        window = max(1, len(episodes) // 50)
        if window > 1:
            success_rate_smooth = np.convolve(success_rate, np.ones(window) / window, mode='valid')
            ax.plot(episodes[window - 1:], success_rate_smooth, label=f'Moving Average (window={window})', color='tab:blue', linewidth=2, linestyle='--')

        # Axis labels and title
        ax.set_xlabel('Episodes', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Success Rate Over Time', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        plt.tight_layout()
        self.save_figure(fig, 'success_rate')
        if show_plots:
            plt.show()
        plt.close(fig)

    def _plot_success_rate_per_agent_over_time(self, metrics: Dict, env: Any, show_plots: bool) -> None:
        """Plot success rate per agent over episodes with enhanced styling."""
        num_agents = env.num_joints
        num_episodes = len(self.logs)
        episodes = np.arange(1, num_episodes + 1)
        success_rate_per_agent = metrics['success_rate_per_agent']  # Shape: (num_agents, num_episodes)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        colors = self.color_palette

        for joint_idx in range(num_agents):
            agent_success_rate = success_rate_per_agent[joint_idx]
            ax.plot(episodes, agent_success_rate, label=f'Joint {joint_idx + 1} Success Rate', color=colors(joint_idx), linewidth=1.5)

        ax.set_xlabel('Episodes', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Success Rate Per Agent Over Time', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        plt.tight_layout()
        self.save_figure(fig, 'success_rate_per_agent')
        if show_plots:
            plt.show()
        plt.close(fig)

    def _plot_cumulative_rewards_per_agent(self, metrics: Dict, env: Any, show_plots: bool) -> None:
        """Plot cumulative rewards per agent with enhanced styling."""
        episodes = np.arange(1, metrics['rewards']['per_agent_cumulative'].shape[1] + 1)
        num_joints = env.num_joints
        cumulative_rewards = metrics['rewards']['per_agent_cumulative']

        # Determine subplot grid dimensions
        if num_joints % 2 == 0:
            ncols = 2
            nrows = num_joints // 2
        else:
            ncols = 1
            nrows = num_joints

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 3 * nrows), dpi=300)
        axes = np.atleast_1d(axes).flatten()

        colors = self.color_palette

        for joint_idx in range(num_joints):
            ax = axes[joint_idx]
            rewards = cumulative_rewards[joint_idx]

            ax.plot(episodes, rewards, label=f'Joint {joint_idx + 1}', color=colors(joint_idx), linewidth=1.5)

            # Apply moving average
            window = max(1, len(episodes) // 50)
            if window > 1:
                rewards_smooth = np.convolve(rewards, np.ones(window) / window, mode='valid')
                ax.plot(episodes[window - 1:], rewards_smooth, label='Moving Average', color='tab:red', linewidth=2, linestyle='--')

            ax.set_xlabel('Episodes', fontsize=12)
            ax.set_ylabel('Cumulative Reward', fontsize=12)
            ax.set_title(f'Joint {joint_idx + 1} Cumulative Reward Over Time', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

        plt.tight_layout()
        self.save_figure(fig, 'cumulative_rewards_per_agent')
        if show_plots:
            plt.show()
        plt.close(fig)


    def _plot_mean_episode_rewards_per_agent(self, metrics: Dict, env: Any, show_plots: bool) -> None:
        """Plot mean episode rewards per agent with enhanced styling and additional smoothing."""
        episodes = np.arange(1, metrics['mean_episode_rewards_per_agent'].shape[1] + 1)
        num_joints = env.num_joints if env else self.num_joints
        mean_episode_rewards_per_agent = metrics['mean_episode_rewards_per_agent']

        # Determine subplot grid dimensions
        if num_joints % 2 == 0:
            ncols = 2
            nrows = num_joints // 2
        else:
            ncols = 1
            nrows = num_joints

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 3 * nrows), dpi=300)
        axes = np.atleast_1d(axes).flatten()

        colors = self.color_palette

        for joint_idx in range(num_joints):
            ax = axes[joint_idx]
            joint_rewards = mean_episode_rewards_per_agent[joint_idx]
            # Clip rewards to a reasonable range
            joint_rewards = np.clip(joint_rewards, -10, 1200)

            # Plot raw data
            ax.plot(episodes, joint_rewards, label=f'Joint {joint_idx + 1} Mean Reward', 
                    color=colors(joint_idx), linewidth=1.5, alpha=0.5)

            # Compute simple moving average (SMA)
            window = max(1, len(episodes) // 50)
            if window > 1:
                moving_avg = np.convolve(joint_rewards, np.ones(window) / window, mode='valid')
                ax.plot(episodes[window - 1:], moving_avg, 
                        label=f'SMA (window={window})', color='tab:red', linewidth=2, linestyle='--')

            # Compute exponential moving average (EMA)
            if len(joint_rewards) > 1:
                alpha_val = 2 / (window + 1)  # EMA factor based on window size
                ema = np.zeros_like(joint_rewards)
                ema[0] = joint_rewards[0]
                for i in range(1, len(joint_rewards)):
                    ema[i] = alpha_val * joint_rewards[i] + (1 - alpha_val) * ema[i - 1]
                
                ax.plot(episodes, ema, label=f'EMA (window={window})', 
                        color='tab:green', linewidth=2, linestyle=':')

            # Customize plot
            ax.set_xlabel('Episodes', fontsize=12)
            ax.set_ylabel('Mean Episode Reward', fontsize=12)
            ax.set_title(f'Joint {joint_idx + 1} Mean Episode Reward Over Time', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

        plt.tight_layout()
        self.save_figure(fig, 'mean_episode_rewards_per_joint')
        if show_plots:
            plt.show()
        plt.close(fig)

    def _plot_combined_metrics(self, metrics: Dict, show_plots: bool) -> None:
        """Plot combined performance metrics with enhanced styling."""
        try:
            episodes = np.arange(1, len(metrics['rewards']['average_per_episode']) + 1)
            rewards = metrics['rewards']['average_per_episode']
            success_rate = metrics['success_rate']['per_episode']
            joint_error_mean = np.mean(metrics['joint_errors']['mean'], axis=1)

            # Normalize rewards
            norm_rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards) + 1e-8)
            # Success rate is already between 0 and 1
            norm_success_rate = success_rate
            # Normalize joint error mean (invert to represent accuracy)
            norm_joint_error = 1.0 - (joint_error_mean - np.min(joint_error_mean)) / (np.max(joint_error_mean) - np.min(joint_error_mean) + 1e-8)

            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            ax.plot(episodes, norm_rewards, label='Normalized Reward', color='tab:blue', linewidth=1.5)
            ax.plot(episodes, norm_success_rate, label='Success Rate', color='tab:green', linewidth=1.5)
            ax.plot(episodes, norm_joint_error, label='Normalized Accuracy', color='tab:red', linewidth=1.5)

            # Smoothing
            window = max(1, len(episodes) // 50)
            if window > 1:
                for data, label, color in zip(
                    [norm_rewards, norm_success_rate, norm_joint_error],
                    ['Normalized Reward', 'Success Rate', 'Normalized Accuracy'],
                    ['tab:blue', 'tab:green', 'tab:red']
                ):
                    data_smooth = np.convolve(data, np.ones(window) / window, mode='valid')
                    ax.plot(episodes[window - 1:], data_smooth, linestyle='--', label=f'{label} (Smoothed)', color=color, linewidth=2)

            ax.set_xlabel('Episodes', fontsize=12)
            ax.set_ylabel('Normalized Value', fontsize=12)
            ax.set_title('Combined Performance Metrics', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

            plt.tight_layout()
            self.save_figure(fig, 'combined_metrics')
            if show_plots:
                plt.show()
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error plotting combined metrics: {str(e)}")
            raise


    def _plot_policy_loss(self, metrics: Dict, show_plots: bool) -> None:
        """Plot Policy Loss over episodes with enhanced styling and robust handling."""
        try:
            raw_policy_loss = np.array(metrics['training']['policy_loss'])
            episodes = np.arange(1, len(raw_policy_loss) + 1)

            if len(raw_policy_loss) == 0:
                self.logger.error("No policy loss data available to plot.")
                return

            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

            # Plot raw data
            ax.plot(episodes, raw_policy_loss, label='Raw Policy Loss', color='tab:orange', alpha=0.3, linewidth=1)

            # Add different smoothing levels
            windows = [5, 20, 50]  # Smoothing window sizes
            colors = ['tab:red', 'tab:blue', 'tab:green']  # Colors for each smoothing level

            for window, color in zip(windows, colors):
                if len(episodes) > window:
                    alpha_val = 2 / (window + 1)
                    smoothed = np.zeros_like(raw_policy_loss)
                    smoothed[0] = raw_policy_loss[0]
                    for i in range(1, len(raw_policy_loss)):
                        smoothed[i] = alpha_val * raw_policy_loss[i] + (1 - alpha_val) * smoothed[i-1]

                    ax.plot(episodes, smoothed, label=f'EMA (window={window})', color=color, linewidth=2)

            # Add trend line
            if len(raw_policy_loss) > 1:
                z = np.polyfit(episodes, raw_policy_loss, 1)
                p = np.poly1d(z)
                ax.plot(episodes, p(episodes), '--', color='tab:purple', label='Trend', linewidth=2)

            # Calculate and display statistics
            stats_text = (
                f'Mean: {np.mean(raw_policy_loss):.4f}\n'
                f'Std: {np.std(raw_policy_loss):.4f}\n'
                f'Min: {np.min(raw_policy_loss):.4f}\n'
                f'Max: {np.max(raw_policy_loss):.4f}'
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

            ax.set_xlabel('Episodes', fontsize=12)
            ax.set_ylabel('Policy Loss', fontsize=12)
            ax.set_title('Policy Loss Over Time (with Smoothing)', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=10)

            plt.tight_layout()
            self.save_figure(fig, 'policy_loss_smoothed')
            self.logger.info("Smoothed policy loss plot saved successfully.")
            if show_plots:
                plt.show()
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error plotting policy loss: {str(e)}")
            raise

    def _plot_policy_loss_per_agent(self, metrics: Dict, show_plots: bool) -> None:
        """Plot policy loss per agent over episodes with enhanced styling and robust handling."""
        try:
            policy_loss_per_agent = np.array(metrics['policy_loss_per_agent'])
            num_agents = policy_loss_per_agent.shape[0]
            num_episodes = policy_loss_per_agent.shape[1]
            episodes = np.arange(1, num_episodes + 1)

            if num_agents == 0 or num_episodes == 0:
                self.logger.error("No policy loss data available for agents.")
                return

            # Determine subplot grid dimensions
            ncols = 2 if num_agents > 1 else 1
            nrows = (num_agents + 1) // 2

            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 3 * nrows), dpi=300)
            axes = np.atleast_1d(axes).flatten()

            # Plot for each agent
            for agent_idx in range(num_agents):
                ax = axes[agent_idx]
                raw_data = policy_loss_per_agent[agent_idx]
                raw_data = np.clip(raw_data, -5, 10)  # Clip the data to a reasonable range
                ax.plot(episodes, raw_data, label='Raw Data', color='tab:orange', alpha=0.3, linewidth=1)

                # Add different smoothing levels
                windows = [5, 20, 50]
                colors = ['tab:red', 'tab:blue', 'tab:green']
                for window, color in zip(windows, colors):
                    if len(episodes) > window:
                        alpha_val = 2 / (window + 1)
                        smoothed = np.zeros_like(raw_data)
                        smoothed[0] = raw_data[0]
                        for i in range(1, len(raw_data)):
                            smoothed[i] = alpha_val * raw_data[i] + (1 - alpha_val) * smoothed[i-1]

                        ax.plot(episodes, smoothed, label=f'EMA (window={window})', color=color, linewidth=2)

                # Add trend line
                if len(raw_data) > 1:
                    z = np.polyfit(episodes, raw_data, 1)
                    p = np.poly1d(z)
                    ax.plot(episodes, p(episodes), '--', color='tab:purple', label='Trend', linewidth=2)

                # Calculate and display statistics
                stats_text = (
                    f'Mean: {np.mean(raw_data):.4f}\n'
                    f'Std: {np.std(raw_data):.4f}\n'
                    f'Min: {np.min(raw_data):.4f}\n'
                    f'Max: {np.max(raw_data):.4f}'
                )
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

                ax.set_xlabel('Episodes', fontsize=12)
                ax.set_ylabel('Policy Loss', fontsize=12)
                ax.set_title(f'Agent {agent_idx + 1} Policy Loss Over Time', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=10)

            # Hide unused subplots
            for ax in axes[num_agents:]:
                ax.axis('off')

            plt.tight_layout()
            self.save_figure(fig, 'policy_loss_per_agent_smoothed')
            self.logger.info("Smoothed policy loss per agent plot saved successfully.")
            if show_plots:
                plt.show()
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error plotting policy loss per agent: {str(e)}")
            raise


    def _plot_minimum_joint_errors(self, metrics: Dict, show_plots: bool) -> None:
        """Plot minimum joint errors per episode for each joint and joint error convergence analysis."""
        try:
            # Extract joint errors from metrics
            joint_errors = np.array(metrics['joint_errors']['min'])  # Shape: (num_episodes, num_joints)
            num_episodes, num_joints = joint_errors.shape
            episodes = np.arange(1, num_episodes + 1)

            # Determine subplot grid dimensions for minimum joint errors
            if num_joints % 2 == 0:
                ncols = 2
                nrows = num_joints // 2
            else:
                ncols = 1
                nrows = num_joints

            fig_min_errors, axes_min = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(10 * ncols, 3 * nrows), dpi=300
            )
            axes_min = np.atleast_1d(axes_min).flatten()

            colors = self.color_palette

            for joint_idx in range(num_joints):
                ax = axes_min[joint_idx]
                errors = joint_errors[:, joint_idx]
                window = min(10, len(episodes) // 10) if len(episodes) > 50 else 1
                min_errors = np.array([np.min(errors[max(0, i - window):i + 1]) for i in range(len(errors))])

                # Raw minimum errors
                ax.plot(episodes, min_errors, label=' Error', color=colors(0), alpha=0.5, linewidth=1.5)

                # Moving average
                if window > 1:
                    moving_avg = np.convolve(min_errors, np.ones(window) / window, mode='valid')
                    ax.plot(episodes[window - 1:], moving_avg, label=f'Moving Avg (window={window})',
                            color=colors(1), linewidth=2, linestyle='--')

                # Trend line
                z = np.polyfit(episodes, min_errors, 1)
                p = np.poly1d(z)
                ax.plot(episodes, p(episodes), linestyle=':', color=colors(2), label='Trend', linewidth=2)

                # Statistics
                stats = {
                    'Min': np.min(min_errors),
                    'Mean': np.mean(min_errors),
                    'Std': np.std(min_errors),
                    'Final': min_errors[-1],
                    'Improvement': ((min_errors[0] - min_errors[-1]) / min_errors[0] * 100 if min_errors[0] != 0 else 0)
                }
                stats_text = (f"Minimum: {stats['Min']:.6f}\nMean: {stats['Mean']:.6f}\n"
                            f"Std Dev: {stats['Std']:.6f}\nFinal: {stats['Final']:.6f}\n"
                            f"Improvement: {stats['Improvement']:.1f}%")
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

                # Highlight best performance
                lowest_error = np.min(min_errors)
                lowest_error_idx = np.argmin(min_errors)
                ax.scatter(episodes[lowest_error_idx], lowest_error, color='red', s=100, zorder=5, label='Best Performance')

                # Customize
                ax.set_xlabel('Episodes', fontsize=12)
                ax.set_ylabel('Joint Error (radians)', fontsize=12)
                ax.set_title(f'Joint {joint_idx + 1}  Error Over Time', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10)

            plt.tight_layout()
            self.save_figure(fig_min_errors, 'minimum_joint_errors')
            if show_plots:
                plt.show()
            plt.close(fig_min_errors)

            # Create subplots for joint error convergence
            fig_convergence, axes_conv = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(10 * ncols, 3 * nrows), dpi=300
            )
            axes_conv = np.atleast_1d(axes_conv).flatten()

            for joint_idx in range(num_joints):
                ax = axes_conv[joint_idx]
                errors = joint_errors[:, joint_idx]
                window = min(10, len(episodes) // 5)
                convergence = np.array([
                    np.std(errors[max(0, i - window):i + 1]) for i in range(len(errors))
                ])

                # Convergence metrics
                ax.plot(episodes, convergence, label='Convergence', color=colors(0), alpha=0.7)

                # Trend line
                z = np.polyfit(episodes, convergence, 1)
                p = np.poly1d(z)
                ax.plot(episodes, p(episodes), linestyle='--', color=colors(1), label='Trend', linewidth=2)

                # Customize
                ax.set_xlabel('Episodes', fontsize=12)
                ax.set_ylabel('Error Stability (Std Dev)', fontsize=12)
                ax.set_title(f'Joint {joint_idx + 1} Error Stability Over Time', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10)

            plt.tight_layout()
            self.save_figure(fig_convergence, 'joint_errors_convergence')
            if show_plots:
                plt.show()
            plt.close(fig_convergence)

        except Exception as e:
            self.logger.error(f"Error plotting minimum joint errors: {str(e)}")
            raise





    def plot_joint_error_convergence(self, metrics, show_plots, num_joints):
        # Ensure 'joint_errors' exists in metrics
        if 'joint_errors' not in metrics:
            print("Error: 'joint_errors' is missing in metrics.")
            return

        joint_errors = metrics['joint_errors']

        # Ensure joint_errors is not empty and has the correct shape
        if not joint_errors or len(joint_errors) == 0:
            print("Error: 'joint_errors' is empty.")
            return

        joint_errors = np.array(joint_errors)
        if joint_errors.ndim != 2 or joint_errors.shape[1] != num_joints:
            print(f"Error: 'joint_errors' has an invalid shape. Expected (?, {num_joints}), got {joint_errors.shape}.")
            return

        # Proceed with plotting
        episodes = np.arange(1, len(joint_errors) + 1)
        fig, ax = plt.subplots(figsize=(15, 6))
        for joint_idx in range(num_joints):
            errors = joint_errors[:, joint_idx]
            window = min(10, len(episodes) // 5)
            convergence = np.array([np.std(errors[max(0, i - window):i + 1]) for i in range(len(errors))])
            ax.plot(episodes, convergence, label=f'Joint {joint_idx + 1}', alpha=0.7)

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


    def _plot_normalized_reward(self, metrics: Dict, show_plots: bool, smoothing_window: int = 10) -> None:
        """Plot smoothed and normalized reward over episodes with overshot control."""
        try:
            rewards = metrics['rewards']['mean']
            # Apply clipping to limit overshoots
            clipped_rewards = self.clip_rewards(rewards)

            episodes = np.arange(1, len(clipped_rewards) + 1)
            
            # Apply moving average smoothing
            if len(clipped_rewards) >= smoothing_window:
                smoothed_rewards = np.convolve(clipped_rewards, np.ones(smoothing_window) / smoothing_window, mode='valid')
                adjusted_episodes = episodes[:len(smoothed_rewards)]
            else:
                smoothed_rewards = clipped_rewards
                adjusted_episodes = episodes

            # Normalize the smoothed rewards
            min_reward = np.min(smoothed_rewards)
            max_reward = np.max(smoothed_rewards)
            if max_reward - min_reward > 1e-8:
                normalized_rewards = (smoothed_rewards - min_reward) / (max_reward - min_reward)
            else:
                normalized_rewards = smoothed_rewards - min_reward  # All zeros in this case

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            ax.plot(adjusted_episodes, normalized_rewards, label='Smoothed Normalized Reward', color='tab:blue', linewidth=1.5)
            ax.set_title('Smoothed and Normalized Reward Over Episodes', fontsize=14)
            ax.set_xlabel('Episodes', fontsize=12)
            ax.set_ylabel('Normalized Reward', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

            plt.tight_layout()
            self.save_figure(fig, 'normalized_reward')
            self.logger.info("Smoothed normalized reward plot saved successfully.")
            if show_plots:
                plt.show()
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error plotting normalized reward: {str(e)}")
            raise

    def _plot_normalized_reward_per_agent(self, metrics: Dict, show_plots: bool, smoothing_window: int = 10) -> None:
        """Plot smoothed and normalized reward per agent with overshot control."""
        try:
            mean_rewards_per_agent = metrics['rewards']['per_agent_mean']
            num_agents = mean_rewards_per_agent.shape[0]
            episodes = np.arange(1, mean_rewards_per_agent.shape[1] + 1)

            # Determine subplot grid dimensions
            ncols, nrows = (2, num_agents // 2) if num_agents % 2 == 0 else (1, num_agents)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 3 * nrows), dpi=300)
            axes = np.atleast_1d(axes).flatten()

            colors = self.color_palette

            for agent_idx in range(num_agents):
                rewards = mean_rewards_per_agent[agent_idx]
                # Clip and smooth the rewards
                clipped_rewards = self.clip_rewards(rewards)
                smoothed_rewards = np.convolve(
                    clipped_rewards, 
                    np.ones(smoothing_window) / smoothing_window, 
                    mode='valid'
                )
                # Normalize the smoothed rewards
                min_reward, max_reward = np.min(smoothed_rewards), np.max(smoothed_rewards)
                normalized_rewards = (smoothed_rewards - min_reward) / (max_reward - min_reward) if max_reward - min_reward > 1e-8 else smoothed_rewards - min_reward
                
                # Adjust the episodes to match the length of smoothed_rewards
                adjusted_episodes = episodes[:len(smoothed_rewards)]
                axes[agent_idx].plot(adjusted_episodes, normalized_rewards, label=f'Agent {agent_idx + 1}', color=colors(agent_idx), linewidth=1.5)
                axes[agent_idx].set_title(f'Agent {agent_idx + 1}', fontsize=14)
                axes[agent_idx].set_ylabel('Normalized Reward', fontsize=12)
                axes[agent_idx].grid(True, alpha=0.3)
                axes[agent_idx].legend(fontsize=10)

            axes[-1].set_xlabel('Episodes', fontsize=12)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self.save_figure(fig, 'normalized_reward_per_agent')
            self.logger.info("Smoothed normalized reward per agent plot saved successfully.")
            if show_plots:
                plt.show()
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error plotting normalized reward per agent: {str(e)}")
            raise


    def _plot_normalized_cumulative_reward_per_agent(self, metrics: Dict, show_plots: bool) -> None:
        """Plot normalized cumulative reward per agent over episodes with enhanced styling."""
        try:
            cumulative_rewards_per_agent = metrics['rewards']['per_agent_cumulative']
            num_agents = cumulative_rewards_per_agent.shape[0]
            episodes = np.arange(1, cumulative_rewards_per_agent.shape[1] + 1)
            
            # Determine subplot grid dimensions
            if num_agents % 2 == 0:
                ncols = 2
                nrows = num_agents // 2
            else:
                ncols = 1
                nrows = num_agents

            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 3 * nrows), dpi=300)
            axes = np.atleast_1d(axes).flatten()

            fig.suptitle('Normalized Cumulative Reward Per Agent Over Episodes', fontsize=16)

            colors = self.color_palette

            for agent_idx in range(num_agents):
                rewards = cumulative_rewards_per_agent[agent_idx]
                # Normalize the rewards
                min_reward = np.min(rewards)
                max_reward = np.max(rewards)
                if max_reward - min_reward > 1e-8:
                    normalized_rewards = (rewards - min_reward) / (max_reward - min_reward)
                else:
                    normalized_rewards = rewards - min_reward  # All zeros in this case
                axes[agent_idx].plot(episodes, normalized_rewards, label=f'Agent {agent_idx + 1}', color=colors(agent_idx), linewidth=1.5)
                axes[agent_idx].set_title(f'Agent {agent_idx + 1}', fontsize=14)
                axes[agent_idx].set_ylabel('Normalized Cumulative Reward', fontsize=12)
                axes[agent_idx].grid(True, alpha=0.3)
                axes[agent_idx].legend(fontsize=10)

            axes[-1].set_xlabel('Episodes', fontsize=12)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self.save_figure(fig, 'normalized_cumulative_reward_per_agent')
            self.logger.info("Normalized cumulative reward per agent plot saved successfully.")
            if show_plots:
                plt.show()
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error plotting normalized cumulative reward per agent: {str(e)}")
            raise

    def _plot_advantages(self, metrics: Dict, show_plots: bool) -> None:
        """Plot advantages over episodes with enhanced styling."""
        try:
            advantages = np.array(metrics.get('training', {}).get('advantages', []))
            if advantages.size == 0:
                self.logger.warning("Advantages data is empty. Skipping plot.")
                return

            if advantages.ndim == 1:
                advantages = advantages.reshape(-1, 1)

            episodes = np.arange(1, advantages.shape[0] + 1)
            mean_advantages = np.mean(advantages, axis=1)
            std_advantages = np.std(advantages, axis=1)

            mean_advantages = np.nan_to_num(mean_advantages)
            std_advantages = np.nan_to_num(std_advantages)

            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            ax.plot(episodes, mean_advantages, label='Mean Advantage', color='tab:blue', linewidth=1.5)
            ax.fill_between(
                episodes,
                mean_advantages - std_advantages,
                mean_advantages + std_advantages,
                alpha=0.3,
                color='tab:blue',
                label='Std Deviation'
            )
            ax.set_title('Advantages Over Episodes', fontsize=14)
            ax.set_xlabel('Episodes', fontsize=12)
            ax.set_ylabel('Advantage', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

            plt.tight_layout()
            self.save_figure(fig, 'advantages_plot')
            self.logger.info("Advantages plot saved successfully.")
            if show_plots:
                plt.show()
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error plotting advantages: {str(e)}")
            raise

    def _plot_value_loss(self, metrics: Dict, show_plots: bool) -> None:
        """Plot value loss over episodes with exponential trendline and enhanced styling."""
        try:
            value_loss = np.array(metrics['training']['critic_loss'])
            episodes = np.arange(1, len(value_loss) + 1)

            # Ensure no zero or negative values for log transformation
            value_loss = np.clip(value_loss, a_min=1e-8, a_max=None)

            # Perform logarithmic transformation of value loss
            log_value_loss = np.log(value_loss)

            # Fit a linear model on the log-transformed value loss
            poly_coeffs = np.polyfit(episodes, log_value_loss, 1)
            exponential_fit = np.exp(poly_coeffs[1]) * np.exp(poly_coeffs[0] * episodes)

            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            ax.plot(episodes, value_loss, label='Raw Value Loss', color='tab:red', alpha=0.5, linewidth=1.5)
            ax.plot(episodes, exponential_fit, label='Exponential Fit', color='tab:blue', linewidth=2)

            ax.set_title('Value Loss Over Episodes (Exponential Trendline)', fontsize=14)
            ax.set_xlabel('Episodes', fontsize=12)
            ax.set_ylabel('Value Loss', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

            plt.tight_layout()
            self.save_figure(fig, 'value_loss_exponential_fit')
            self.logger.info("Value loss plot saved successfully.")
            if show_plots:
                plt.show()
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error plotting value loss: {str(e)}")
            raise

    # Add any other plotting functions as needed with similar enhancements
# Add the following methods to the TrainingMetrics class:

    def _plot_normalized_cumulative_reward_per_agent(self, metrics: Dict, show_plots: bool) -> None:
        """Plot normalized cumulative reward per agent over episodes with enhanced styling."""
        try:
            cumulative_rewards_per_agent = metrics['rewards']['per_agent_cumulative']
            num_agents = cumulative_rewards_per_agent.shape[0]
            episodes = np.arange(1, cumulative_rewards_per_agent.shape[1] + 1)
    
            fig, axs = plt.subplots(num_agents, 1, figsize=(10, 3 * num_agents), dpi=300, sharex=True)
            if num_agents == 1:
                axs = [axs]
            fig.suptitle('Normalized Cumulative Reward Per Agent Over Episodes', fontsize=16)
    
            colors = self.color_palette
    
            for agent_idx in range(num_agents):
                rewards = cumulative_rewards_per_agent[agent_idx]
                # Normalize the rewards
                min_reward = np.min(rewards)
                max_reward = np.max(rewards)
                if max_reward - min_reward > 1e-8:
                    normalized_rewards = (rewards - min_reward) / (max_reward - min_reward)
                else:
                    normalized_rewards = rewards - min_reward  # All zeros in this case
                axs[agent_idx].plot(episodes, normalized_rewards, label=f'Agent {agent_idx + 1}', color=colors(agent_idx), linewidth=1.5)
    
                # Apply moving average smoothing
                window = max(1, len(episodes) // 50)
                if window > 1:
                    normalized_rewards_smooth = np.convolve(normalized_rewards, np.ones(window) / window, mode='valid')
                    axs[agent_idx].plot(episodes[window - 1:], normalized_rewards_smooth, label='Smoothed', color='tab:red', linewidth=2, linestyle='--')
    
                axs[agent_idx].set_title(f'Agent {agent_idx + 1}', fontsize=14)
                axs[agent_idx].set_ylabel('Normalized Cumulative Reward', fontsize=12)
                axs[agent_idx].grid(True, alpha=0.3)
                axs[agent_idx].legend(fontsize=10)
    
            axs[-1].set_xlabel('Episodes', fontsize=12)
    
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self.save_figure(fig, 'normalized_cumulative_reward_per_agent')
            self.logger.info("Normalized cumulative reward per agent plot saved successfully.")
            if show_plots:
                plt.show()
            plt.close(fig)
    
        except Exception as e:
            self.logger.error(f"Error plotting normalized cumulative reward per agent: {str(e)}")
            raise

    def _plot_advantages(self, metrics: Dict, show_plots: bool) -> None:
        """Plot advantages over episodes with enhanced styling."""
        try:
            advantages = np.array(metrics.get('training', {}).get('advantages', []))
            if advantages.size == 0:
                self.logger.warning("Advantages data is empty. Skipping plot.")
                return

            if advantages.ndim == 1:
                advantages = advantages.reshape(-1, 1)

            episodes = np.arange(1, advantages.shape[0] + 1)
            mean_advantages = np.mean(advantages, axis=1)
            std_advantages = np.std(advantages, axis=1)

            mean_advantages = np.nan_to_num(mean_advantages)
            std_advantages = np.nan_to_num(std_advantages)

            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            ax.plot(episodes, mean_advantages, label='Mean Advantage', color='tab:blue', linewidth=1.5)
            ax.fill_between(
            episodes,
            mean_advantages - std_advantages,
            mean_advantages + std_advantages,
            alpha=0.3,
            color='tab:blue',
            label='Std Deviation'
        )


            # Apply moving average smoothing
            window = max(1, len(episodes) // 50)
            if window > 1:
                mean_adv_smooth = np.convolve(mean_advantages, np.ones(window) / window, mode='valid')
                ax.plot(episodes[window - 1:], mean_adv_smooth, linestyle='--', color='tab:red', linewidth=2, label='Smoothed Mean')

            ax.set_title('Advantages Over Episodes', fontsize=14)
            ax.set_xlabel('Episodes', fontsize=12)
            ax.set_ylabel('Advantage', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

            plt.tight_layout()
            self.save_figure(fig, 'advantages_plot')
            self.logger.info("Advantages plot saved successfully.")
            if show_plots:
                plt.show()
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error plotting advantages: {str(e)}")
            raise
    def _plot_convergence(self, metrics: Dict, show_plots: bool) -> None:
        """Plot convergence of the training process over episodes."""
        try:
            episodes = np.arange(1, len(metrics['success_rate']['per_episode']) + 1)
            success_rates = metrics['success_rate']['per_episode']

            # Calculate moving average of success rates
            window = min(100, len(episodes))
            moving_avg_success_rate = np.convolve(success_rates, np.ones(window) / window, mode='valid')
            adjusted_episodes = episodes[window - 1:]

            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

            # Plot raw success rates
            ax.plot(episodes, success_rates, label='Success Rate', color='tab:blue', alpha=0.3, linewidth=1)

            # Plot moving average success rate
            ax.plot(adjusted_episodes, moving_avg_success_rate, label=f'{window}-Episode Moving Average', color='tab:red', linewidth=2)

            # Plot convergence threshold
            ax.axhline(y=self.convergence_threshold, color='tab:green', linestyle='--', label=f'Convergence Threshold ({self.convergence_threshold})', linewidth=2)

            # Mark the convergence point if convergence was achieved
            if self.converged and self.episodes_to_converge is not None:
                ax.axvline(x=self.episodes_to_converge, color='tab:purple', linestyle='--', label=f'Converged at Episode {self.episodes_to_converge}', linewidth=2)

            ax.set_xlabel('Episodes', fontsize=12)
            ax.set_ylabel('Success Rate', fontsize=12)
            ax.set_title('Training Convergence', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

            plt.tight_layout()
            self.save_figure(fig, 'training_convergence')
            self.logger.info("Convergence plot saved successfully.")
            if show_plots:
                plt.show()
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error plotting convergence: {str(e)}")
            raise
    def _plot_reward_std_over_time(self, metrics: Dict, show_plots: bool) -> None:
        """Plot the standard deviation of rewards over episodes."""
        try:
            episodes = np.arange(1, len(metrics['rewards']['mean']) + 1)
            reward_std = metrics['rewards']['std_per_episode']  # Per-episode std of rewards

            # Plot the std over episodes
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            ax.plot(episodes, reward_std, label='Reward Standard Deviation', color='tab:blue', linewidth=1.5)

            # Optionally, apply moving average smoothing
            window = max(1, len(episodes) // 50)
            if window > 1:
                smoothed_std = np.convolve(reward_std, np.ones(window) / window, mode='valid')
                adjusted_episodes = episodes[window - 1:]
                ax.plot(adjusted_episodes, smoothed_std, label=f'Smoothed Std (window={window})', color='tab:red', linewidth=2, linestyle='--')

            ax.set_xlabel('Episodes', fontsize=12)
            ax.set_ylabel('Standard Deviation', fontsize=12)
            ax.set_title('Reward Standard Deviation Over Episodes', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

            plt.tight_layout()
            self.save_figure(fig, 'reward_std_over_time')
            self.logger.info("Reward standard deviation plot saved successfully.")
            if show_plots:
                plt.show()
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error plotting reward standard deviation: {str(e)}")
            raise

    def _plot_reward_std_per_agent(self, metrics: Dict, show_plots: bool) -> None:
        """Plot the standard deviation of rewards per agent over episodes."""
        try:
            std_per_agent_per_episode = metrics['rewards']['std_per_agent_per_episode']  # Shape: (episodes, num_agents)
            num_episodes, num_agents = std_per_agent_per_episode.shape
            episodes = np.arange(1, num_episodes + 1)

            # Determine subplot grid dimensions
            if num_agents % 2 == 0:
                ncols = 2
                nrows = num_agents // 2
            else:
                ncols = 1
                nrows = num_agents

            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 3 * nrows), dpi=300)
            axes = np.atleast_1d(axes).flatten()

            colors = self.color_palette

            for agent_idx in range(num_agents):
                ax = axes[agent_idx]
                std_over_episodes = std_per_agent_per_episode[:, agent_idx]

                # Optionally, apply moving average smoothing
                window = max(1, num_episodes // 50)
                if window > 1:
                    smoothed_std = np.convolve(std_over_episodes, np.ones(window) / window, mode='valid')
                    adjusted_episodes = episodes[window - 1:]
                    ax.plot(adjusted_episodes, smoothed_std, label=f'Smoothed Std (window={window})', color='tab:red', linewidth=2, linestyle='--')
                else:
                    ax.plot(episodes, std_over_episodes, label=f'Agent {agent_idx + 1} Reward Std', color=colors(agent_idx % 10), linewidth=1.5)

                ax.set_xlabel('Episodes', fontsize=12)
                ax.set_ylabel('Standard Deviation', fontsize=12)
                ax.set_title(f'Agent {agent_idx + 1} Reward Std Over Time', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10)

            # Remove any unused subplots
            for idx in range(num_agents, nrows * ncols):
                fig.delaxes(axes[idx])

            plt.tight_layout()
            self.save_figure(fig, 'reward_std_per_agent_over_time')
            self.logger.info("Reward standard deviation per agent plot saved successfully.")
            if show_plots:
                plt.show()
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error plotting reward standard deviation per agent: {str(e)}")
            raise

    def clip_rewards(self, rewards, lower_bound=-10.0, upper_bound=300.0):
        """
        Clip rewards using fixed bounds.
        
        Args:
            rewards: Array of rewards
            lower_bound (float): Lower bound for clipping (default: -10.0)
            upper_bound (float): Upper bound for clipping (default: 30.0)
            
        Returns:
            Array of clipped rewards
        """
        try:
            rewards = np.array(rewards)
            if rewards.size == 0:
                return rewards
                
            # Remove any non-finite values
            rewards = rewards[np.isfinite(rewards)]
            if rewards.size == 0:
                return np.zeros_like(rewards)
                
            # Clip rewards to fixed bounds
            clipped_rewards = np.clip(rewards, lower_bound, upper_bound)
            return clipped_rewards
            
        except Exception as e:
            self.logger.warning(f"Error in clip_rewards: {str(e)}. Returning unclipped rewards.")
            return rewards
        
    def _plot_actor_loss_per_actor(self, metrics: Dict, show_plots: bool) -> None:
        """Plot actor loss per agent over episodes with enhanced styling and robust handling."""
        try:
            actor_loss_per_agent = metrics.get('actor_loss_per_agent', None)
            if actor_loss_per_agent is None:
                self.logger.warning("No actor_loss_per_agent data found in metrics. Skipping plot.")
                return

            num_agents, num_episodes = actor_loss_per_agent.shape
            episodes = np.arange(1, num_episodes + 1)

            if num_agents == 0 or num_episodes == 0:
                self.logger.warning("No actor loss data available for agents.")
                return

            # Determine subplot grid dimensions
            ncols = 2 if num_agents > 1 else 1
            nrows = (num_agents + 1) // 2

            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 3 * nrows), dpi=300)
            axes = np.atleast_1d(axes).flatten()

            # Plot for each agent
            for agent_idx in range(num_agents):
                ax = axes[agent_idx]
                raw_data = actor_loss_per_agent[agent_idx]
                # Clip data to a reasonable range if desired
                raw_data = np.clip(raw_data, -5, 10)
                ax.plot(episodes, raw_data, label='Raw Data', color='tab:orange', alpha=0.3, linewidth=1)

                # Add different smoothing levels
                windows = [5, 20, 50]
                colors = ['tab:red', 'tab:blue', 'tab:green']
                for window, color in zip(windows, colors):
                    if len(episodes) > window:
                        alpha_val = 2 / (window + 1)
                        smoothed = np.zeros_like(raw_data)
                        smoothed[0] = raw_data[0]
                        for i in range(1, len(raw_data)):
                            smoothed[i] = alpha_val * raw_data[i] + (1 - alpha_val) * smoothed[i-1]

                        ax.plot(episodes, smoothed, label=f'EMA (window={window})', color=color, linewidth=2)

                # Add trend line
                if len(raw_data) > 1:
                    z = np.polyfit(episodes, raw_data, 1)
                    p = np.poly1d(z)
                    ax.plot(episodes, p(episodes), '--', color='tab:purple', label='Trend', linewidth=2)

                # Calculate and display statistics
                stats_text = (
                    f'Mean: {np.mean(raw_data):.4f}\n'
                    f'Std: {np.std(raw_data):.4f}\n'
                    f'Min: {np.min(raw_data):.4f}\n'
                    f'Max: {np.max(raw_data):.4f}'
                )
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

                ax.set_xlabel('Episodes', fontsize=12)
                ax.set_ylabel('Actor Loss', fontsize=12)
                ax.set_title(f'Agent {agent_idx + 1} Actor Loss Over Time', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=10)

            # Hide unused subplots
            for ax in axes[num_agents:]:
                ax.axis('off')

            plt.tight_layout()
            self.save_figure(fig, 'actor_loss_per_agent_smoothed')
            self.logger.info("Smoothed actor loss per agent plot saved successfully.")
            if show_plots:
                plt.show()
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error plotting actor loss per agent: {str(e)}")
            raise
