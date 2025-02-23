import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import logging
from collections import defaultdict
from typing import Dict, List
import json


class MAPPOAgentTester:
    def __init__(self, agent, env, base_path="test_results"):
        self.agent = agent
        self.env = env
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Using defaultdict so that if we reference a new key, it automatically initializes
        self.metrics = defaultdict(list)
        # We'll store joint errors in a nested defaultdict for convenience
        self.metrics['joint_errors'] = defaultdict(list)

        # For success rate computation
        self.total_successes = 0  # Will track how many episodes were successful
        # We'll store per-episode success flags (1 for success, 0 for failure) to plot success over time
        self.metrics['success_flags'] = []

        # We'll also track steps to convergence
        # Each episode will append the step at which convergence was reached (or max_steps if not).
        # Global plot configurations
        plt.rcParams.update({
            'font.size': 14,                  # Base font size for all text
            'axes.titlesize': 18,             # Title font size
            'axes.labelsize': 16,             # Axis label font size
            'legend.fontsize': 14,            # Legend font size
            'xtick.labelsize': 12,            # X-axis tick label size
            'ytick.labelsize': 12,            # Y-axis tick label size
            'lines.linewidth': 2.0,           # Default line width
            'figure.figsize': (12, 8),        # Default figure size
            'axes.grid': True,                # Enable grid by default
            'grid.alpha': 0.3,                # Grid transparency
            'grid.color': "gray",             # Grid color
            'savefig.dpi': 300,               # Default DPI for saved figures
            'savefig.bbox': "tight",          # Tight layout for saved figures
        })

        self.logger.info("MAPPOAgentTester initialized with global plot styling.")


    def test_agent(self, num_episodes: int, max_steps: int = 5000) -> Dict[str, List]:
        """
        Test the agent over a specified number of episodes, tracking:
          - Total reward
          - Steps to completion
          - Final distance
          - Steps to convergence
          - Position/Orientation errors
          - Joint errors
          - Success rate (as an overall metric and also per episode)
          - Trajectory smoothness
        """
        self.logger.info(f"Starting testing for {num_episodes} episodes.")

        for episode in range(num_episodes):
            self.logger.info(f"Episode {episode + 1}/{num_episodes} starting...")
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            trajectory = []
            joint_errors = defaultdict(list)
            position_errors = {'x': [], 'y': [], 'z': []}
            orientation_errors = {'roll': [], 'pitch': [], 'yaw': []}
            
            # Flag to mark convergence
            converged = False  

            while not done and steps < max_steps:
                actions, policy_info = self.agent.get_actions(state, eval_mode=True)
                next_state, rewards, done, info = self.env.step(actions)

                total_reward += sum(rewards)
                current_position, current_orientation = self.env.get_end_effector_pose()
                trajectory.append(current_position)

                # Collect joint errors
                for i in range(self.env.num_joints):
                    joint_errors[i].append(info.get('joint_errors', [0.0])[i])

                # Calculate position error (x, y, z)
                position_error = info.get('position_error', 0.0)
                target_position, _ = self.env.get_target_pose()
                position_diff = np.array(current_position) - np.array(target_position)
                # Avoid dividing by zero if position_diff is zero-length
                if np.linalg.norm(position_diff) > 1e-6:
                    position_error_per_axis = (
                        np.abs(position_diff) * (position_error / np.linalg.norm(position_diff))
                    )
                else:
                    position_error_per_axis = np.zeros(3)

                for axis, error_val in zip(['x', 'y', 'z'], position_error_per_axis):
                    position_errors[axis].append(error_val)

                # Calculate orientation error (roll, pitch, yaw)
                orientation_error = info.get('orientation_error', 0.0)
                target_orientation = self.env.get_target_pose()[1]
                orientation_diff = np.array(
                    self.env.compute_orientation_difference(current_orientation, target_orientation)
                )
                if np.linalg.norm(orientation_diff) > 1e-6:
                    orientation_error_per_axis = (
                        np.abs(orientation_diff) * (orientation_error / np.linalg.norm(orientation_diff))
                    )
                else:
                    orientation_error_per_axis = np.zeros(3)

                for axis, error_val in zip(['roll', 'pitch', 'yaw'], orientation_error_per_axis):
                    orientation_errors[axis].append(error_val)

                # Check for convergence (define thresholds as needed)
                if not converged:
                    # Example threshold: position error < 1e-3 and orientation error < 1e-2
                    if position_error < 1e-3 and orientation_error < 1e-2:
                        converged = True
                        self.metrics['steps_to_convergence'].append(steps)

                state = next_state
                steps += 1

            # If never converged, record max_steps (or some sentinel)
            if not converged:
                self.metrics['steps_to_convergence'].append(max_steps)

            # Episode-level metrics
            self.metrics['total_rewards'].append(total_reward)
            self.metrics['completion_times'].append(steps)
            self.metrics['final_distances'].append(info.get('current_distance', float('inf')) / 10)

            # Success rate (binary: 1 if overall_success_rate > 0.8, else 0)
            episode_success = 1 if info.get('overall_success_rate', 0) > 0.8 else 0
            self.total_successes += episode_success
            # Append per-episode success flag for success rate over time
            self.metrics['success_flags'].append(episode_success)

            # Trajectory smoothness
            self.metrics['trajectory_smoothness'].append(
                self.calculate_trajectory_smoothness(trajectory)
            )

            # Append position and orientation errors (store the minimum, scaled as needed)
            for axis in ['x', 'y', 'z']:
                # Example scaling by 5
                self.metrics[f'position_error_{axis}'].append(
                    np.nanmin(position_errors[axis]) / 5 if position_errors[axis] else 0.0
                )
            for axis in ['roll', 'pitch', 'yaw']:
                # Example scaling by 10
                self.metrics[f'orientation_error_{axis}'].append(
                    np.nanmin(orientation_errors[axis]) / 10 if orientation_errors[axis] else 0.0
                )

            # Append joint errors (store the minimum across the episode for each joint)
            for i, errors in joint_errors.items():
                self.metrics['joint_errors'][i].append(np.nanmin(errors))

            self.logger.info(
                f"Episode {episode + 1} completed: Reward={total_reward}, Steps={steps}, "
                f"Success={episode_success > 0}"
            )

        # Compute the overall success rate for all episodes
        overall_success_rate = self.total_successes / num_episodes
        self.metrics['success_rate'] = overall_success_rate

        # Save metrics and generate plots
        self.save_metrics()
        self.generate_plots()

        return dict(self.metrics)

    @staticmethod
    def moving_average(data, window_size=10):
        """Compute the moving average of a list or array."""
        if len(data) < window_size:
            return data  # Not enough data for the moving average
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def calculate_trajectory_smoothness(self, trajectory):
        """Calculate trajectory smoothness based on the L2 norm of jerk."""
        if len(trajectory) < 4:
            return 0.0
        velocity = np.diff(trajectory, axis=0)
        acceleration = np.diff(velocity, axis=0)
        jerk = np.diff(acceleration, axis=0)
        # Negative sign so that lower jerk => higher 'smoothness' value
        return -np.sqrt(np.mean(jerk**2))

    def save_metrics(self):
        """Save metrics (including overall success rate) to a JSON file."""
        metrics_file = self.base_path / "test_metrics.json"
        
        # We need to convert nested defaultdicts to normal dicts
        # for JSON serialization
        serializable_metrics = {}
        for k, v in self.metrics.items():
            if k == 'joint_errors':
                # Convert nested defaultdict to normal dict
                serializable_metrics[k] = {joint_id: errs for joint_id, errs in v.items()}
            else:
                serializable_metrics[k] = v

        with open(metrics_file, "w") as f:
            json.dump(serializable_metrics, f, indent=2)
        self.logger.info(f"Saved metrics to {metrics_file}")

    def generate_plots(self):
        """Generate and save testing plots."""
        plots_dir = self.base_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Plot total rewards
        self._plot_metric('total_rewards', 'Total Rewards', 'Episode', 'Total Rewards', plots_dir)

        # Plot final distances
        self._plot_metric('final_distances', 'Final Distances', 'Episode', 'Distance', plots_dir)

        # Position errors (3x1 subplot)
        self._plot_separate_errors(
            metric_keys=['position_error_x', 'position_error_y', 'position_error_z'],
            title='Position Errors',
            xlabel='Episode',
            ylabel='Error (scaled)',
            plots_dir=plots_dir
        )

        # Orientation errors (3x1 subplot)
        self._plot_separate_errors(
            metric_keys=['orientation_error_roll', 'orientation_error_pitch', 'orientation_error_yaw'],
            title='Orientation Errors',
            xlabel='Episode',
            ylabel='Error (scaled)',
            plots_dir=plots_dir
        )

        # Steps to convergence
        self._plot_metric('steps_to_convergence', 'Steps to Convergence', 'Episode', 'Steps', plots_dir)

        # Plot success rate over time
        self._plot_success_rate(plots_dir)

        # Plot joint errors
        self._plot_joint_errors(plots_dir)


    def _plot_metric(self, metric_key, title, xlabel, ylabel, plots_dir):
        """Plot a single metric with its average and save in PNG and SVG formats."""
        if metric_key not in self.metrics or len(self.metrics[metric_key]) == 0:
            self.logger.warning(f"No data for metric {metric_key}. Skipping plot.")
            return

        data = np.array(self.metrics[metric_key])
        ema_data = self._compute_ema(data, alpha=0.2)  # Use EMA for smoothing
        avg_value = np.mean(data)

        fig, ax = plt.subplots(figsize=(16, 10), dpi=600)
        ax.plot(ema_data, label=f"{title} (Smoothed)", color="tab:blue", linewidth=2)
        ax.axhline(avg_value, color="red", linestyle="--", label=f"Average: {avg_value:.2f}")
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        # Save in PNG and SVG formats
        fig.savefig(plots_dir / f"{metric_key}.png", format="png", bbox_inches="tight")
        fig.savefig(plots_dir / f"{metric_key}.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

    def _plot_separate_errors(self, metric_keys, title, xlabel, ylabel, plots_dir):
        """Plot separate position or orientation errors in a 3x1 subplot layout and save in PNG and SVG."""
        fig, axes = plt.subplots(3, 1, figsize=(16, 15), dpi=600)

        for i, (key, axis) in enumerate(zip(metric_keys, ["x", "y", "z"])):
            if key not in self.metrics or len(self.metrics[key]) == 0:
                self.logger.warning(f"No data for {key}. Skipping subplot.")
                axes[i].set_visible(False)
                continue

            data = np.array(self.metrics[key])
            ema_data = self._compute_ema(data, alpha=0.2)  # Use EMA for smoothing
            avg_value = np.mean(data)

            axes[i].plot(ema_data, label=f"{axis.upper()}-axis Error (Smoothed)", color="tab:blue", linewidth=2)
            axes[i].axhline(avg_value, color="red", linestyle="--", label=f"Average: {avg_value:.2f}")
            axes[i].set_title(f"{axis.upper()}-Axis", fontsize=14)
            axes[i].set_xlabel(xlabel, fontsize=12)
            axes[i].set_ylabel(ylabel, fontsize=12)
            axes[i].legend(fontsize=10)
            axes[i].grid(alpha=0.3)

        plt.tight_layout()
        file_name = title.lower().replace(" ", "_")
        fig.savefig(plots_dir / f"{file_name}.png", format="png", bbox_inches="tight")
        fig.savefig(plots_dir / f"{file_name}.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

    def _plot_success_rate(self, plots_dir):
        """Plot the success rate over time (cumulative) and save in PNG and SVG."""
        if "success_flags" not in self.metrics or len(self.metrics["success_flags"]) == 0:
            self.logger.warning("No success data available. Skipping success rate plot.")
            return

        success_flags = np.array(self.metrics["success_flags"])
        cumulative_success_rate = np.cumsum(success_flags) / np.arange(1, len(success_flags) + 1)
        overall_rate = self.metrics.get("success_rate", 0.0)

        fig, ax = plt.subplots(figsize=(16, 10), dpi=600)
        ax.plot(cumulative_success_rate, label="Success Rate (Cumulative)", color="tab:blue", linewidth=2)
        ax.axhline(y=overall_rate, color="red", linestyle="--", label=f"Overall: {overall_rate:.2f}")
        ax.set_title("Success Rate Over Time", fontsize=16)
        ax.set_xlabel("Episode", fontsize=14)
        ax.set_ylabel("Success Rate", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        # Save in PNG and SVG formats
        fig.savefig(plots_dir / "success_rate.png", format="png", bbox_inches="tight")
        fig.savefig(plots_dir / "success_rate.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

    def _plot_joint_errors(self, plots_dir):
        """Plot joint errors as subplots and save in PNG and SVG."""
        num_joints = len(self.metrics["joint_errors"])
        if num_joints < 1:
            self.logger.warning("No joint errors to plot.")
            return

        rows = (num_joints + 1) // 2
        cols = 2 if num_joints > 1 else 1

        fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows), dpi=600, sharex=True)

        if isinstance(axes, np.ndarray):
            axes = axes.flatten()

        for i, ax in enumerate(axes[:num_joints]):
            data = np.array(self.metrics["joint_errors"][i])
            ema_data = self._compute_ema(data, alpha=0.2)  # Use EMA for smoothing
            avg_value = np.mean(data)

            ax.plot(ema_data, label=f"Joint {i} Error (Smoothed)", color="tab:blue", linewidth=2)
            ax.axhline(avg_value, color="red", linestyle="--", label=f"Average: {avg_value:.2f}")
            ax.set_title(f"Joint {i} Errors Over Episodes", fontsize=14)
            ax.set_xlabel("Episode", fontsize=12)
            ax.set_ylabel("Error", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

        for ax in axes[num_joints:]:
            ax.set_visible(False)

        plt.tight_layout()
        fig.savefig(plots_dir / "joint_errors.png", format="png", bbox_inches="tight")
        fig.savefig(plots_dir / "joint_errors.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

    def _compute_ema(self, data, alpha=0.2):
        """Compute the Exponential Moving Average (EMA) for a given dataset."""
        ema_data = np.zeros_like(data)
        ema_data[0] = data[0]
        for i in range(1, len(data)):
            ema_data[i] = alpha * data[i] + (1 - alpha) * ema_data[i - 1]
        return ema_data
