import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import logging
from collections import defaultdict, deque
from typing import Dict, List, Any
import json
import datetime
import platform
import sys
import torch
from scipy import stats
import warnings


class MAPPOAgentTester:
    def __init__(self, agent, env, base_path="test_results"):
        self.agent = agent
        self.env = env
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Using defaultdict for automatic initialization
        self.metrics = defaultdict(list)
        self.metrics['joint_errors'] = defaultdict(list)

        # Success tracking
        self.total_successes = 0
        self.metrics['success_flags'] = []

        # IMPROVED: Convergence criteria configuration
        self.convergence_config = {
            # Position thresholds (in meters, typically)
            'position_threshold': 0.01,      # 1cm accuracy
            'position_threshold_strict': 0.005,  # 5mm for strict convergence
            
            # Orientation thresholds (in radians)
            'orientation_threshold': 0.05,   # ~2.9 degrees
            'orientation_threshold_strict': 0.02,  # ~1.1 degrees
            
            # Joint error thresholds (in radians)
            'joint_error_threshold': 0.05,   # ~2.9 degrees per joint
            'joint_error_threshold_strict': 0.02,  # ~1.1 degrees per joint
            
            # Overall distance threshold (combined position + orientation)
            'overall_distance_threshold': 0.03,
            'overall_distance_threshold_strict': 0.015,
            
            # Stability requirements
            'stability_window': 5,           # Must maintain convergence for N steps
            'max_oscillation': 0.001,       # Maximum allowed oscillation during stability check
            
            # Success rate threshold for episode success
            'episode_success_threshold': 0.8,  # 80% of joints must succeed
        }
        
        # IMPROVED: Multiple convergence tracking
        self.convergence_types = [
            'position_convergence',
            'orientation_convergence', 
            'joint_convergence',
            'overall_convergence',
            'stable_convergence'
        ]
        
        for conv_type in self.convergence_types:
            self.metrics[f'steps_to_{conv_type}'] = []

        # NEW: Performance-based metrics to fix reward paradox
        self.metrics['performance_scores'] = []
        self.metrics['efficiency_scores'] = []
        self.metrics['normalized_rewards'] = []

        # Global plot configurations
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'legend.fontsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'lines.linewidth': 2.0,
            'figure.figsize': (12, 8),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': "gray",
            'savefig.dpi': 300,
            'savefig.bbox': "tight",
        })

        self.logger.info("MAPPOAgentTester initialized with improved convergence criteria.")

    def test_agent(self, num_episodes: int, max_steps: int = 5000) -> Dict[str, List]:
        """
        IMPROVED: Test agent with comprehensive convergence tracking and performance-based scoring.
        """
        self.logger.info(f"Starting testing for {num_episodes} episodes with improved convergence criteria.")

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
            
            # IMPROVED: Convergence tracking
            convergence_tracker = self._initialize_convergence_tracker()
            
            # IMPROVED: Stability tracking for robust convergence detection
            stability_tracker = {
                'position_history': deque(maxlen=self.convergence_config['stability_window']),
                'orientation_history': deque(maxlen=self.convergence_config['stability_window']),
                'joint_history': deque(maxlen=self.convergence_config['stability_window']),
                'overall_history': deque(maxlen=self.convergence_config['stability_window'])
            }

            while not done and steps < max_steps:
                # Get actions - handle different return signatures
                action_result = self.agent.get_actions(state, eval_mode=True)
                if len(action_result) == 3:
                    actions, policy_info, _ = action_result
                elif len(action_result) == 2:
                    actions, policy_info = action_result
                else:
                    actions = action_result
                    
                next_state, rewards, done, info = self.env.step(actions)

                total_reward += sum(rewards)
                current_position, current_orientation = self.env.get_end_effector_pose()
                trajectory.append(current_position)

                # IMPROVED: Extract error metrics with better handling
                position_error, orientation_error, current_joint_errors = self._extract_error_metrics(
                    info, current_position, current_orientation
                )
                
                # Store joint errors
                for i in range(self.env.num_joints):
                    if i < len(current_joint_errors):
                        joint_errors[i].append(current_joint_errors[i])
                    else:
                        joint_errors[i].append(0.0)

                # IMPROVED: Calculate detailed position and orientation errors
                position_error_per_axis, orientation_error_per_axis = self._calculate_detailed_errors(
                    current_position, current_orientation, position_error, orientation_error
                )
                
                # Store detailed errors
                for axis, error_val in zip(['x', 'y', 'z'], position_error_per_axis):
                    position_errors[axis].append(error_val)
                for axis, error_val in zip(['roll', 'pitch', 'yaw'], orientation_error_per_axis):
                    orientation_errors[axis].append(error_val)

                # IMPROVED: Check multiple convergence criteria
                self._check_convergence_criteria(
                    convergence_tracker, stability_tracker, steps,
                    position_error, orientation_error, current_joint_errors,
                    info.get('current_distance', float('inf'))
                )

                state = next_state
                steps += 1

            # IMPROVED: Record convergence results for all criteria
            self._record_convergence_results(convergence_tracker, max_steps)

            # Episode-level metrics
            self.metrics['total_rewards'].append(total_reward)
            self.metrics['completion_times'].append(steps)
            self.metrics['final_distances'].append(info.get('current_distance', float('inf')) / 10)

            # IMPROVED: Success determination with configurable threshold
            episode_success = self._determine_episode_success(info, current_joint_errors, position_error, orientation_error)
            self.total_successes += episode_success
            self.metrics['success_flags'].append(episode_success)

            # NEW: Calculate performance-based metrics to fix reward paradox
            performance_score, efficiency_score, normalized_reward = self._calculate_performance_metrics(
                total_reward, steps, max_steps, episode_success, position_error, orientation_error
            )
            self.metrics['performance_scores'].append(performance_score)
            self.metrics['efficiency_scores'].append(efficiency_score)
            self.metrics['normalized_rewards'].append(normalized_reward)

            # Trajectory smoothness
            self.metrics['trajectory_smoothness'].append(
                self.calculate_trajectory_smoothness(trajectory)
            )

            # Store error metrics
            self._store_error_metrics(position_errors, orientation_errors, joint_errors)

            self.logger.info(
                f"Episode {episode + 1} completed: Reward={total_reward:.2f}, Steps={steps}, "
                f"Success={episode_success > 0}, Performance={performance_score:.2f}, "
                f"Pos_err={position_error:.4f}, Orient_err={orientation_error:.4f}"
            )

        # Compute overall success rate
        overall_success_rate = self.total_successes / num_episodes
        self.metrics['success_rate'] = overall_success_rate

        # IMPROVED: Log convergence statistics
        self._log_convergence_statistics()

        # Save metrics and generate plots
        self.save_metrics()
        self.generate_plots()

        # IMPROVED: Generate comprehensive report automatically
        self.generate_comprehensive_report()

        return dict(self.metrics)

    def _calculate_performance_metrics(self, total_reward, steps, max_steps, success, position_error, orientation_error):
        """
        NEW: Calculate performance-based metrics that properly rank episodes.
        
        This fixes the reward paradox where failed long episodes get higher rewards than successful short ones.
        """
        # Base performance score
        performance_score = 0.0
        
        # 1. Success bonus (large positive reward for completing the task)
        if success:
            success_bonus = 10000  # Large bonus for success
            performance_score += success_bonus
            
            # 2. Efficiency bonus (reward for completing quickly)
            efficiency_ratio = (max_steps - steps) / max_steps
            efficiency_bonus = efficiency_ratio * 5000  # Up to 5000 bonus for efficiency
            performance_score += efficiency_bonus
            
            # 3. Accuracy bonus (reward for low errors)
            accuracy_bonus = 0
            if position_error < 0.001:  # Very accurate position
                accuracy_bonus += 1000
            elif position_error < 0.005:  # Good position
                accuracy_bonus += 500
                
            if orientation_error < 0.01:  # Very accurate orientation
                accuracy_bonus += 1000
            elif orientation_error < 0.02:  # Good orientation
                accuracy_bonus += 500
                
            performance_score += accuracy_bonus
            
        else:
            # 4. Failure penalty (negative score for not completing)
            failure_penalty = -5000  # Large penalty for failure
            performance_score += failure_penalty
            
            # 5. Partial credit based on progress (small reward for getting closer)
            # Use normalized reward as partial credit, but cap it low
            progress_credit = min(total_reward / steps, 100)  # Max 100 points for progress
            performance_score += progress_credit

        # 6. Normalize the raw reward by episode length to get reward per step
        normalized_reward = total_reward / steps if steps > 0 else 0

        # 7. Calculate efficiency score (0-1, where 1 is most efficient)
        if success:
            efficiency_score = (max_steps - steps) / max_steps
        else:
            efficiency_score = 0.0  # No efficiency credit for failed episodes

        return performance_score, efficiency_score, normalized_reward

    def _initialize_convergence_tracker(self):
        """Initialize convergence tracking for an episode."""
        return {
            'position_converged': False,
            'orientation_converged': False,
            'joint_converged': False,
            'overall_converged': False,
            'stable_converged': False,
            'position_convergence_step': None,
            'orientation_convergence_step': None,
            'joint_convergence_step': None,
            'overall_convergence_step': None,
            'stable_convergence_step': None,
        }

    def _extract_error_metrics(self, info, current_position, current_orientation):
        """Extract error metrics from environment info with fallbacks."""
        try:
            # Position error
            if 'position_error' in info:
                position_error = float(info['position_error'])
            else:
                target_position, _ = self.env.get_target_pose()
                position_diff = np.array(current_position) - np.array(target_position)
                position_error = np.linalg.norm(position_diff)
            
            # Orientation error
            if 'orientation_error' in info:
                orientation_error = float(info['orientation_error'])
            else:
                _, target_orientation = self.env.get_target_pose()
                orientation_diff = self.env.compute_orientation_difference(
                    current_orientation, target_orientation
                )
                orientation_error = np.linalg.norm(orientation_diff)
            
            # Joint errors
            if 'joint_errors' in info:
                joint_errors = info['joint_errors']
            else:
                # Fallback: use zero errors
                joint_errors = [0.0] * self.env.num_joints
                
            return position_error, orientation_error, joint_errors
            
        except Exception as e:
            self.logger.warning(f"Error extracting metrics: {e}")
            return 1.0, 1.0, [1.0] * self.env.num_joints

    def _calculate_detailed_errors(self, current_position, current_orientation, position_error, orientation_error):
        """Calculate detailed per-axis errors."""
        try:
            target_position, target_orientation = self.env.get_target_pose()
            
            # Position errors per axis
            position_diff = np.array(current_position) - np.array(target_position)
            if np.linalg.norm(position_diff) > 1e-6:
                position_error_per_axis = np.abs(position_diff) * (position_error / np.linalg.norm(position_diff))
            else:
                position_error_per_axis = np.zeros(3)

            # Orientation errors per axis
            orientation_diff = np.array(
                self.env.compute_orientation_difference(current_orientation, target_orientation)
            )
            if np.linalg.norm(orientation_diff) > 1e-6:
                orientation_error_per_axis = np.abs(orientation_diff) * (orientation_error / np.linalg.norm(orientation_diff))
            else:
                orientation_error_per_axis = np.zeros(3)
                
            return position_error_per_axis, orientation_error_per_axis
            
        except Exception as e:
            self.logger.warning(f"Error calculating detailed errors: {e}")
            return np.zeros(3), np.zeros(3)

    def _check_convergence_criteria(self, convergence_tracker, stability_tracker, steps,
                                   position_error, orientation_error, joint_errors, overall_distance):
        """IMPROVED: Check multiple convergence criteria with stability requirements."""
        
        # 1. Position convergence
        if not convergence_tracker['position_converged']:
            if position_error < self.convergence_config['position_threshold']:
                convergence_tracker['position_converged'] = True
                convergence_tracker['position_convergence_step'] = steps
                self.logger.debug(f"Position converged at step {steps}")

        # 2. Orientation convergence  
        if not convergence_tracker['orientation_converged']:
            if orientation_error < self.convergence_config['orientation_threshold']:
                convergence_tracker['orientation_converged'] = True
                convergence_tracker['orientation_convergence_step'] = steps
                self.logger.debug(f"Orientation converged at step {steps}")

        # 3. Joint convergence (all joints must be within threshold)
        if not convergence_tracker['joint_converged']:
            max_joint_error = max(joint_errors) if joint_errors else float('inf')
            if max_joint_error < self.convergence_config['joint_error_threshold']:
                convergence_tracker['joint_converged'] = True
                convergence_tracker['joint_convergence_step'] = steps
                self.logger.debug(f"Joints converged at step {steps}")

        # 4. Overall convergence (combined position + orientation)
        if not convergence_tracker['overall_converged']:
            if overall_distance < self.convergence_config['overall_distance_threshold']:
                convergence_tracker['overall_converged'] = True
                convergence_tracker['overall_convergence_step'] = steps
                self.logger.debug(f"Overall converged at step {steps}")

        # 5. IMPROVED: Stable convergence (must maintain convergence for stability_window steps)
        if not convergence_tracker['stable_converged']:
            # Update stability tracking
            stability_tracker['position_history'].append(position_error)
            stability_tracker['orientation_history'].append(orientation_error)
            stability_tracker['joint_history'].append(max(joint_errors) if joint_errors else float('inf'))
            stability_tracker['overall_history'].append(overall_distance)
            
            # Check if we have enough history and all criteria are stable
            if len(stability_tracker['position_history']) == self.convergence_config['stability_window']:
                is_stable = self._check_stability(stability_tracker)
                if is_stable:
                    convergence_tracker['stable_converged'] = True
                    convergence_tracker['stable_convergence_step'] = steps
                    self.logger.debug(f"Stable convergence at step {steps}")

    def _check_stability(self, stability_tracker):
        """Check if errors are stable within the stability window."""
        try:
            # Check position stability
            pos_errors = list(stability_tracker['position_history'])
            pos_stable = (max(pos_errors) < self.convergence_config['position_threshold_strict'] and
                         (max(pos_errors) - min(pos_errors)) < self.convergence_config['max_oscillation'])
            
            # Check orientation stability
            orient_errors = list(stability_tracker['orientation_history'])
            orient_stable = (max(orient_errors) < self.convergence_config['orientation_threshold_strict'] and
                           (max(orient_errors) - min(orient_errors)) < self.convergence_config['max_oscillation'])
            
            # Check joint stability
            joint_errors = list(stability_tracker['joint_history'])
            joint_stable = (max(joint_errors) < self.convergence_config['joint_error_threshold_strict'] and
                          (max(joint_errors) - min(joint_errors)) < self.convergence_config['max_oscillation'])
            
            # Check overall stability
            overall_errors = list(stability_tracker['overall_history'])
            overall_stable = (max(overall_errors) < self.convergence_config['overall_distance_threshold_strict'] and
                            (max(overall_errors) - min(overall_errors)) < self.convergence_config['max_oscillation'])
            
            return pos_stable and orient_stable and joint_stable and overall_stable
            
        except Exception as e:
            self.logger.warning(f"Error checking stability: {e}")
            return False

    def _record_convergence_results(self, convergence_tracker, max_steps):
        """Record convergence results for the episode."""
        for conv_type in self.convergence_types:
            step_key = f'{conv_type.replace("_convergence", "")}_convergence_step'
            if convergence_tracker.get(step_key) is not None:
                self.metrics[f'steps_to_{conv_type}'].append(convergence_tracker[step_key])
            else:
                self.metrics[f'steps_to_{conv_type}'].append(max_steps)

    def _determine_episode_success(self, info, joint_errors, position_error, orientation_error):
        """IMPROVED: Determine episode success with multiple criteria."""
        try:
            # Method 1: Use info if available
            if 'overall_success_rate' in info:
                success_rate = info['overall_success_rate']
                return 1 if success_rate > self.convergence_config['episode_success_threshold'] else 0
            
            # Method 2: Calculate based on individual joint success
            if 'success_per_joint' in info:
                joint_successes = info['success_per_joint']
                success_rate = sum(joint_successes) / len(joint_successes) if joint_successes else 0
                return 1 if success_rate > self.convergence_config['episode_success_threshold'] else 0
            
            # Method 3: Calculate based on error thresholds
            position_success = position_error < self.convergence_config['position_threshold']
            orientation_success = orientation_error < self.convergence_config['orientation_threshold']
            
            if joint_errors:
                joint_success_rate = sum(1 for e in joint_errors 
                                       if e < self.convergence_config['joint_error_threshold']) / len(joint_errors)
            else:
                joint_success_rate = 0
            
            # Overall success requires all criteria
            overall_success = (position_success and orientation_success and 
                             joint_success_rate > self.convergence_config['episode_success_threshold'])
            
            return 1 if overall_success else 0
            
        except Exception as e:
            self.logger.warning(f"Error determining episode success: {e}")
            return 0

    def _store_error_metrics(self, position_errors, orientation_errors, joint_errors):
        """Store error metrics for analysis."""
        # Position errors (scaled)
        for axis in ['x', 'y', 'z']:
            self.metrics[f'position_error_{axis}'].append(
                np.nanmin(position_errors[axis]) / 5 if position_errors[axis] else 0.0
            )
        
        # Orientation errors (scaled)
        for axis in ['roll', 'pitch', 'yaw']:
            self.metrics[f'orientation_error_{axis}'].append(
                np.nanmin(orientation_errors[axis]) / 10 if orientation_errors[axis] else 0.0
            )

        # Joint errors (minimum across episode for each joint)
        for i, errors in joint_errors.items():
            self.metrics['joint_errors'][i].append(np.nanmin(errors) if errors else 0.0)

    def _log_convergence_statistics(self):
        """Log convergence statistics for analysis."""
        self.logger.info("=== CONVERGENCE STATISTICS ===")
        
        for conv_type in self.convergence_types:
            steps_key = f'steps_to_{conv_type}'
            if steps_key in self.metrics and self.metrics[steps_key]:
                steps_data = np.array(self.metrics[steps_key])
                
                # Calculate statistics
                mean_steps = np.mean(steps_data)
                median_steps = np.median(steps_data)
                std_steps = np.std(steps_data)
                min_steps = np.min(steps_data)
                max_steps = np.max(steps_data)
                
                # Count successful convergences (not max_steps)
                successful_convergences = np.sum(steps_data < max(steps_data))
                success_rate = successful_convergences / len(steps_data)
                
                self.logger.info(f"{conv_type.replace('_', ' ').title()}:")
                self.logger.info(f"  Success Rate: {success_rate:.2%}")
                self.logger.info(f"  Mean Steps: {mean_steps:.1f}")
                self.logger.info(f"  Median Steps: {median_steps:.1f}")
                self.logger.info(f"  Std Dev: {std_steps:.1f}")
                self.logger.info(f"  Range: [{min_steps:.0f}, {max_steps:.0f}]")
                self.logger.info("")

    @staticmethod
    def moving_average(data, window_size=10):
        """Compute the moving average of a list or array."""
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def calculate_trajectory_smoothness(self, trajectory):
        """Calculate trajectory smoothness based on the L2 norm of jerk."""
        if len(trajectory) < 4:
            return 0.0
        velocity = np.diff(trajectory, axis=0)
        acceleration = np.diff(velocity, axis=0)
        jerk = np.diff(acceleration, axis=0)
        return -np.sqrt(np.mean(jerk**2))

    def save_metrics(self):
        """Save metrics to JSON file with convergence configuration."""
        metrics_file = self.base_path / "test_metrics.json"
        
        # Convert to serializable format
        serializable_metrics = {}
        for k, v in self.metrics.items():
            if k == 'joint_errors':
                serializable_metrics[k] = {joint_id: errs for joint_id, errs in v.items()}
            else:
                serializable_metrics[k] = v
        
        # Add convergence configuration for reference
        serializable_metrics['convergence_config'] = self.convergence_config

        with open(metrics_file, "w") as f:
            json.dump(serializable_metrics, f, indent=2)
        self.logger.info(f"Saved metrics to {metrics_file}")

    def generate_plots(self):
        """Generate all plots including convergence analysis."""
        plots_dir = self.base_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Existing plots
        self._plot_metric('total_rewards', 'Total Rewards', 'Episode', 'Total Rewards', plots_dir)
        # NEW: Performance-based plots
        self._plot_metric('performance_scores', 'Performance Scores', 'Episode', 'Performance Score', plots_dir)
        self._plot_metric('efficiency_scores', 'Efficiency Scores', 'Episode', 'Efficiency Score', plots_dir)
        self._plot_metric('normalized_rewards', 'Normalized Rewards (per step)', 'Episode', 'Reward per Step', plots_dir)
        
        self._plot_metric('final_distances', 'Final Distances', 'Episode', 'Distance', plots_dir)
        
        # Position and orientation error plots
        self._plot_separate_errors(
            ['position_error_x', 'position_error_y', 'position_error_z'],
            'Position Errors', 'Episode', 'Error (scaled)', plots_dir
        )
        self._plot_separate_errors(
            ['orientation_error_roll', 'orientation_error_pitch', 'orientation_error_yaw'],
            'Orientation Errors', 'Episode', 'Error (scaled)', plots_dir
        )

        # IMPROVED: Convergence analysis plots
        self._plot_convergence_analysis(plots_dir)
        
        # Success rate and joint errors
        self._plot_success_rate(plots_dir)
        self._plot_joint_errors(plots_dir)
        
        # NEW: Performance comparison plot
        self._plot_performance_comparison(plots_dir)

    def _plot_performance_comparison(self, plots_dir):
        """NEW: Plot comparing different performance metrics."""
        if not all(key in self.metrics for key in ['total_rewards', 'performance_scores', 'success_flags']):
            self.logger.warning("Insufficient data for performance comparison plot.")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=600)
        
        episodes = np.arange(1, len(self.metrics['total_rewards']) + 1)
        success_flags = np.array(self.metrics['success_flags'])
        
        # Create colors based on success/failure
        colors = ['green' if s else 'red' for s in success_flags]
        
        # Plot 1: Total Rewards vs Performance Score
        ax1.scatter(self.metrics['total_rewards'], self.metrics['performance_scores'], 
                   c=colors, alpha=0.7, s=50)
        ax1.set_xlabel('Total Reward')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Total Reward vs Performance Score')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Success'),
                          Patch(facecolor='red', label='Failure')]
        ax1.legend(handles=legend_elements)
        
        # Plot 2: Episode Performance Scores over time
        ax2.scatter(episodes, self.metrics['performance_scores'], c=colors, alpha=0.7, s=50)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Performance Score')
        ax2.set_title('Performance Score Over Episodes')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Efficiency vs Total Reward
        ax3.scatter(self.metrics['total_rewards'], self.metrics['efficiency_scores'],
                   c=colors, alpha=0.7, s=50)
        ax3.set_xlabel('Total Reward')
        ax3.set_ylabel('Efficiency Score')
        ax3.set_title('Total Reward vs Efficiency')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Completion Time vs Performance Score
        completion_times = self.metrics.get('completion_times', [])
        if completion_times:
            ax4.scatter(completion_times, self.metrics['performance_scores'],
                       c=colors, alpha=0.7, s=50)
            ax4.set_xlabel('Completion Time (steps)')
            ax4.set_ylabel('Performance Score')
            ax4.set_title('Completion Time vs Performance Score')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No completion time data', ha='center', va='center', 
                    transform=ax4.transAxes)
        
        plt.tight_layout()
        fig.savefig(plots_dir / "performance_comparison.png", format="png", bbox_inches="tight")
        fig.savefig(plots_dir / "performance_comparison.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

    def _plot_convergence_analysis(self, plots_dir):
        """IMPROVED: Plot comprehensive convergence analysis."""
        # Plot convergence steps for different criteria
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=600)
        axes = axes.flatten()
        
        for i, conv_type in enumerate(self.convergence_types):
            if i >= len(axes):
                break
                
            steps_key = f'steps_to_{conv_type}'
            if steps_key in self.metrics and self.metrics[steps_key]:
                data = np.array(self.metrics[steps_key])
                
                # Plot histogram
                axes[i].hist(data, bins=20, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{conv_type.replace("_", " ").title()}')
                axes[i].set_xlabel('Steps to Convergence')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = np.mean(data)
                axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
                axes[i].legend()
        
        # Hide unused subplot
        if len(self.convergence_types) < len(axes):
            axes[-1].set_visible(False)
            
        plt.tight_layout()
        fig.savefig(plots_dir / "convergence_analysis.png", format="png", bbox_inches="tight")
        fig.savefig(plots_dir / "convergence_analysis.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

    def _plot_metric(self, metric_key, title, xlabel, ylabel, plots_dir):
        """Plot a single metric with its average and save in PNG and SVG formats."""
        if metric_key not in self.metrics or len(self.metrics[metric_key]) == 0:
            self.logger.warning(f"No data for metric {metric_key}. Skipping plot.")
            return

        data = np.array(self.metrics[metric_key])
        ema_data = self._compute_ema(data, alpha=0.2)
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

        fig.savefig(plots_dir / f"{metric_key}.png", format="png", bbox_inches="tight")
        fig.savefig(plots_dir / f"{metric_key}.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

    def _plot_separate_errors(self, metric_keys, title, xlabel, ylabel, plots_dir):
        """Plot separate errors in subplots."""
        fig, axes = plt.subplots(3, 1, figsize=(16, 15), dpi=600)

        for i, (key, axis) in enumerate(zip(metric_keys, ["x", "y", "z"])):
            if key not in self.metrics or len(self.metrics[key]) == 0:
                self.logger.warning(f"No data for {key}. Skipping subplot.")
                axes[i].set_visible(False)
                continue

            data = np.array(self.metrics[key])
            ema_data = self._compute_ema(data, alpha=0.2)
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
        """Plot the success rate over time."""
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

        fig.savefig(plots_dir / "success_rate.png", format="png", bbox_inches="tight")
        fig.savefig(plots_dir / "success_rate.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

    def _plot_joint_errors(self, plots_dir):
        """Plot joint errors as subplots."""
        num_joints = len(self.metrics["joint_errors"])
        if num_joints < 1:
            self.logger.warning("No joint errors to plot.")
            return

        rows = (num_joints + 1) // 2
        cols = 2 if num_joints > 1 else 1

        fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows), dpi=600, sharex=True)

        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for i in range(min(num_joints, len(axes))):
            data = np.array(self.metrics["joint_errors"][i])
            ema_data = self._compute_ema(data, alpha=0.2)
            avg_value = np.mean(data)

            axes[i].plot(ema_data, label=f"Joint {i} Error (Smoothed)", color="tab:blue", linewidth=2)
            axes[i].axhline(avg_value, color="red", linestyle="--", label=f"Average: {avg_value:.2f}")
            axes[i].set_title(f"Joint {i} Errors Over Episodes", fontsize=14)
            axes[i].set_xlabel("Episode", fontsize=12)
            axes[i].set_ylabel("Error", fontsize=12)
            axes[i].legend(fontsize=10)
            axes[i].grid(alpha=0.3)

        # Hide unused subplots
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

    def get_convergence_summary(self):
        """
        IMPROVED: Get a comprehensive summary of convergence performance.
        
        Returns:
            dict: Detailed convergence analysis
        """
        summary = {
            'convergence_config': self.convergence_config,
            'convergence_performance': {},
            'overall_performance': {
                'total_episodes': len(self.metrics.get('success_flags', [])),
                'overall_success_rate': self.metrics.get('success_rate', 0.0),
                'average_reward': np.mean(self.metrics.get('total_rewards', [0])),
                'average_completion_time': np.mean(self.metrics.get('completion_times', [0])),
                # NEW: Performance-based metrics
                'average_performance_score': np.mean(self.metrics.get('performance_scores', [0])),
                'average_efficiency_score': np.mean(self.metrics.get('efficiency_scores', [0])),
                'average_normalized_reward': np.mean(self.metrics.get('normalized_rewards', [0]))
            }
        }
        
        # Analyze each convergence type
        for conv_type in self.convergence_types:
            steps_key = f'steps_to_{conv_type}'
            if steps_key in self.metrics and self.metrics[steps_key]:
                steps_data = np.array(self.metrics[steps_key])
                max_steps = np.max(steps_data)
                
                # Calculate success rate (episodes that converged before max_steps)
                successful_convergences = np.sum(steps_data < max_steps)
                success_rate = successful_convergences / len(steps_data)
                
                # Calculate statistics only for successful convergences
                successful_steps = steps_data[steps_data < max_steps]
                
                convergence_stats = {
                    'success_rate': success_rate,
                    'total_episodes': len(steps_data),
                    'successful_episodes': successful_convergences,
                }
                
                if len(successful_steps) > 0:
                    convergence_stats.update({
                        'mean_steps_to_convergence': np.mean(successful_steps),
                        'median_steps_to_convergence': np.median(successful_steps),
                        'std_steps_to_convergence': np.std(successful_steps),
                        'min_steps_to_convergence': np.min(successful_steps),
                        'max_steps_to_convergence': np.max(successful_steps),
                        'percentile_25': np.percentile(successful_steps, 25),
                        'percentile_75': np.percentile(successful_steps, 75)
                    })
                else:
                    convergence_stats.update({
                        'mean_steps_to_convergence': None,
                        'median_steps_to_convergence': None,
                        'std_steps_to_convergence': None,
                        'min_steps_to_convergence': None,
                        'max_steps_to_convergence': None,
                        'percentile_25': None,
                        'percentile_75': None
                    })
                
                summary['convergence_performance'][conv_type] = convergence_stats
        
        return summary

    def print_convergence_report(self):
        """
        IMPROVED: Print a comprehensive convergence report.
        """
        summary = self.get_convergence_summary()
        
        print("\n" + "="*80)
        print("                    CONVERGENCE ANALYSIS REPORT")
        print("="*80)
        
        # Overall performance
        overall = summary['overall_performance']
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total Episodes Tested: {overall['total_episodes']}")
        print(f"   Overall Success Rate: {overall['overall_success_rate']:.1%}")
        print(f"   Average Reward: {overall['average_reward']:.2f}")
        print(f"   Average Performance Score: {overall['average_performance_score']:.2f}")
        print(f"   Average Efficiency Score: {overall['average_efficiency_score']:.2f}")
        print(f"   Average Completion Time: {overall['average_completion_time']:.1f} steps")
        
        # Convergence thresholds
        config = summary['convergence_config']
        print(f"\n⚙️  CONVERGENCE THRESHOLDS:")
        print(f"   Position Threshold: {config['position_threshold']:.4f} m")
        print(f"   Orientation Threshold: {config['orientation_threshold']:.4f} rad ({np.degrees(config['orientation_threshold']):.1f}°)")
        print(f"   Joint Error Threshold: {config['joint_error_threshold']:.4f} rad ({np.degrees(config['joint_error_threshold']):.1f}°)")
        print(f"   Overall Distance Threshold: {config['overall_distance_threshold']:.4f}")
        print(f"   Stability Window: {config['stability_window']} steps")
        
        # Detailed convergence analysis
        print(f"\n🎯 CONVERGENCE PERFORMANCE:")
        print("-" * 80)
        
        convergence_names = {
            'position_convergence': 'Position Convergence',
            'orientation_convergence': 'Orientation Convergence',
            'joint_convergence': 'Joint Convergence',
            'overall_convergence': 'Overall Convergence',
            'stable_convergence': 'Stable Convergence'
        }
        
        for conv_type, stats in summary['convergence_performance'].items():
            name = convergence_names.get(conv_type, conv_type.replace('_', ' ').title())
            print(f"\n📈 {name}:")
            print(f"   Success Rate: {stats['success_rate']:.1%} ({stats['successful_episodes']}/{stats['total_episodes']} episodes)")
            
            if stats['mean_steps_to_convergence'] is not None:
                print(f"   Mean Steps: {stats['mean_steps_to_convergence']:.1f}")
                print(f"   Median Steps: {stats['median_steps_to_convergence']:.1f}")
                print(f"   Std Deviation: {stats['std_steps_to_convergence']:.1f}")
                print(f"   Range: [{stats['min_steps_to_convergence']:.0f}, {stats['max_steps_to_convergence']:.0f}]")
                print(f"   25th-75th Percentile: [{stats['percentile_25']:.0f}, {stats['percentile_75']:.0f}]")
            else:
                print("   No successful convergences recorded")
        
        print("\n" + "="*80)

    def generate_comprehensive_report(self):
        """Generate comprehensive testing report automatically."""
        try:
            generator = TestingReportGenerator(self, dict(self.metrics))
            report_content = generator.generate_report()
            return report_content
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return None


# UPDATED: TestingReportGenerator with performance-based episode ranking
class TestingReportGenerator:
    """
    Generates comprehensive testing reports for MAPPO agent evaluation.
    """
    
    def __init__(self, tester, test_results: Dict[str, Any]):
        self.tester = tester
        self.results = test_results
        self.convergence_summary = tester.get_convergence_summary()
        self.timestamp = datetime.datetime.now()
        
    def generate_report(self, output_path: str = None) -> str:
        """
        Generate a comprehensive testing report.
        
        Args:
            output_path: Path to save the report file
            
        Returns:
            str: The generated report content
        """
        if output_path is None:
            output_path = self.tester.base_path / "testing_report.txt"
        
        report_content = self._build_report()
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        print(f"📄 Comprehensive testing report saved to: {output_path}")
        return report_content
    
    def _build_report(self) -> str:
        """Build the complete report content."""
        sections = [
            self._build_header(),
            self._build_executive_summary(),
            self._build_test_configuration(),
            self._build_convergence_analysis(),
            self._build_performance_ranking(),
            self._build_detailed_metrics(),
            self._build_episode_breakdown(),
            self._build_analysis_recommendations(),
            self._build_statistical_analysis(),
            self._build_environment_details(),
            self._build_conclusion()
        ]
        
        return "\n\n".join(sections)
    
    def _build_header(self) -> str:
        """Build report header."""
        test_duration = self._calculate_test_duration()
        total_episodes = self.convergence_summary['overall_performance']['total_episodes']
        
        return f"""# MAPPO Agent Testing Report

**Generated:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Test Duration:** {test_duration}
**Episodes Tested:** {total_episodes}
**Max Steps Per Episode:** 5000

{"="*80}"""

    def _build_executive_summary(self) -> str:
        """Build executive summary section."""
        overall = self.convergence_summary['overall_performance']
        rewards = self.results.get('total_rewards', [0])
        performance_scores = self.results.get('performance_scores', [0])
        
        # Calculate key findings
        key_findings = self._generate_key_findings()
        
        return f"""## 📊 EXECUTIVE SUMMARY

**Overall Performance:**
- Success Rate: {overall['overall_success_rate']:.1%}
- Average Reward: {overall['average_reward']:.2f}
- Average Performance Score: {overall.get('average_performance_score', 0):.2f}
- Average Efficiency Score: {overall.get('average_efficiency_score', 0):.2f}
- Average Episode Duration: {overall['average_completion_time']:.1f} steps
- Best Episode Reward: {max(rewards):.2f}
- Best Performance Score: {max(performance_scores):.2f}

**Key Findings:**
{chr(10).join(f'- {finding}' for finding in key_findings)}"""

    def _build_test_configuration(self) -> str:
        """Build test configuration section."""
        config = self.convergence_summary['convergence_config']
        
        return f"""## ⚙️ TEST CONFIGURATION

**Environment Settings:**
- Robot Model: {getattr(self.tester.env, 'robot_name', 'xarm')}
- Number of Joints: {self.tester.env.num_joints}
- Environment Type: Inverse Kinematics

**Convergence Thresholds:**
- Position Threshold: {config['position_threshold']:.4f} m ({config['position_threshold']*1000:.1f} mm)
- Orientation Threshold: {config['orientation_threshold']:.4f} rad ({np.degrees(config['orientation_threshold']):.1f}°)
- Joint Error Threshold: {config['joint_error_threshold']:.4f} rad ({np.degrees(config['joint_error_threshold']):.1f}°)
- Overall Distance Threshold: {config['overall_distance_threshold']:.4f}
- Stability Window: {config['stability_window']} steps
- Maximum Oscillation: {config['max_oscillation']:.4f}
- Episode Success Threshold: {config['episode_success_threshold']:.1%}

**Agent Configuration:**
- Algorithm: Multi-Agent Proximal Policy Optimization (MAPPO)
- Evaluation Mode: True
- PD Controller Integration: {hasattr(self.tester.agent, 'pd_controllers')}

**Performance Scoring:**
- Success Bonus: 10,000 points
- Efficiency Bonus: Up to 5,000 points
- Accuracy Bonus: Up to 2,000 points
- Failure Penalty: -5,000 points"""

    def _build_convergence_analysis(self) -> str:
        """Build detailed convergence analysis."""
        conv_perf = self.convergence_summary['convergence_performance']
        
        sections = []
        convergence_names = {
            'position_convergence': 'Position Convergence',
            'orientation_convergence': 'Orientation Convergence',
            'joint_convergence': 'Joint Convergence',
            'overall_convergence': 'Overall Convergence',
            'stable_convergence': 'Stable Convergence'
        }
        
        for conv_type, stats in conv_perf.items():
            name = convergence_names.get(conv_type, conv_type.replace('_', ' ').title())
            
            if stats['mean_steps_to_convergence'] is not None:
                section = f"""### {name}
- **Success Rate:** {stats['success_rate']:.1%} ({stats['successful_episodes']}/{stats['total_episodes']} episodes)
- **Mean Steps to Convergence:** {stats['mean_steps_to_convergence']:.1f} ± {stats['std_steps_to_convergence']:.1f}
- **Median Steps:** {stats['median_steps_to_convergence']:.1f}
- **Best Performance:** {stats['min_steps_to_convergence']:.0f} steps
- **Worst Performance:** {stats['max_steps_to_convergence']:.0f} steps
- **25th-75th Percentile:** [{stats['percentile_25']:.0f}, {stats['percentile_75']:.0f}] steps"""
            else:
                section = f"""### {name}
- **Success Rate:** {stats['success_rate']:.1%} ({stats['successful_episodes']}/{stats['total_episodes']} episodes)
- **Mean Steps to Convergence:** No successful convergences
- **Status:** Needs significant improvement"""
            
            sections.append(section)
        
        return f"""## 🎯 CONVERGENCE ANALYSIS

{chr(10).join(sections)}"""

    def _build_performance_ranking(self) -> str:
        """Build performance ranking section."""
        conv_perf = self.convergence_summary['convergence_performance']
        convergence_names = {
            'position_convergence': 'Position Convergence',
            'orientation_convergence': 'Orientation Convergence',
            'joint_convergence': 'Joint Convergence',
            'overall_convergence': 'Overall Convergence',
            'stable_convergence': 'Stable Convergence'
        }
        
        # Sort by success rate
        ranked = sorted(conv_perf.items(), key=lambda x: x[1]['success_rate'], reverse=True)
        
        ranking_lines = []
        for i, (conv_type, stats) in enumerate(ranked, 1):
            name = convergence_names.get(conv_type, conv_type.replace('_', ' ').title())
            mean_steps = stats['mean_steps_to_convergence']
            if mean_steps is not None:
                mean_str = f" (avg: {mean_steps:.1f} steps)"
            else:
                mean_str = " (no successes)"
            ranking_lines.append(f"{i}. {name}: {stats['success_rate']:.1%}{mean_str}")
        
        return f"""## 🏆 PERFORMANCE RANKING

**Convergence Success Rates (Highest to Lowest):**
{chr(10).join(ranking_lines)}"""

    def _build_detailed_metrics(self) -> str:
        """Build detailed metrics section."""
        # Calculate error statistics
        pos_errors = self._calculate_position_error_stats()
        orient_errors = self._calculate_orientation_error_stats()
        joint_errors_table = self._build_joint_errors_table()
        trajectory_stats = self._calculate_trajectory_stats()
        reward_stats = self._calculate_reward_stats()
        performance_stats = self._calculate_performance_stats()
        
        return f"""## 📈 DETAILED METRICS

### Error Analysis
**Position Errors (Average per Axis):**
- X-Axis Error: {pos_errors['x']['mean']:.6f} ± {pos_errors['x']['std']:.6f}
- Y-Axis Error: {pos_errors['y']['mean']:.6f} ± {pos_errors['y']['std']:.6f}
- Z-Axis Error: {pos_errors['z']['mean']:.6f} ± {pos_errors['z']['std']:.6f}

**Orientation Errors (Average per Axis):**
- Roll Error: {orient_errors['roll']['mean']:.6f} ± {orient_errors['roll']['std']:.6f} rad
- Pitch Error: {orient_errors['pitch']['mean']:.6f} ± {orient_errors['pitch']['std']:.6f} rad
- Yaw Error: {orient_errors['yaw']['mean']:.6f} ± {orient_errors['yaw']['std']:.6f} rad

**Joint Errors (Average per Joint):**
{joint_errors_table}

### Performance Analysis
- **Average Performance Score:** {performance_stats['mean_performance']:.2f}
- **Best Performance Score:** {performance_stats['max_performance']:.2f}
- **Worst Performance Score:** {performance_stats['min_performance']:.2f}
- **Average Efficiency Score:** {performance_stats['mean_efficiency']:.3f}
- **Performance Consistency:** {performance_stats['performance_std']:.2f}

### Trajectory Analysis
- **Average Trajectory Smoothness:** {trajectory_stats['mean']:.6f}
- **Smoothness Std Deviation:** {trajectory_stats['std']:.6f}
- **Best Trajectory Smoothness:** {trajectory_stats['max']:.6f}
- **Worst Trajectory Smoothness:** {trajectory_stats['min']:.6f}

### Reward Analysis
- **Total Cumulative Reward:** {reward_stats['total']:.2f}
- **Reward Standard Deviation:** {reward_stats['std']:.2f}
- **Reward Trend:** {reward_stats['trend']}
- **Reward Consistency:** {reward_stats['consistency']:.1%}"""

    def _build_episode_breakdown(self) -> str:
        """Build episode-by-episode breakdown with FIXED ranking based on performance score."""
        best_episodes = self._get_best_episodes_by_performance()
        worst_episodes = self._get_worst_episodes_by_performance()
        fastest_convergence = self._get_fastest_convergence_episodes()
        failure_analysis = self._analyze_failures()
        
        return f"""## 📊 EPISODE-BY-EPISODE BREAKDOWN

**Top 5 Best Episodes (by Performance Score):**
{best_episodes}

**Top 5 Worst Episodes (by Performance Score):**
{worst_episodes}

**Episodes with Fastest Convergence:**
{fastest_convergence}

**Episodes with Failed Convergence:**
- Total Failed Episodes: {failure_analysis['count']}
- Failure Rate: {failure_analysis['rate']:.1%}
- Common Failure Patterns: {failure_analysis['patterns']}"""

    def _get_best_episodes_by_performance(self) -> str:
        """FIXED: Get table of best episodes ranked by performance score instead of raw reward."""
        if 'performance_scores' not in self.results:
            return "No performance score data available"
        
        performance_scores = np.array(self.results['performance_scores'])
        total_rewards = np.array(self.results.get('total_rewards', []))
        completion_times = np.array(self.results.get('completion_times', []))
        success_flags = np.array(self.results.get('success_flags', []))
        
        # Get top 5 episodes by PERFORMANCE SCORE (not raw reward)
        top_indices = np.argsort(performance_scores)[-5:][::-1]
        
        lines = ["Episode | Performance | Reward  | Steps | Success"]
        lines.append("--------|-------------|---------|-------|--------")
        
        for i, idx in enumerate(top_indices):
            episode_num = idx + 1
            performance = performance_scores[idx]
            reward = total_rewards[idx] if len(total_rewards) > idx else "N/A"
            steps = completion_times[idx] if len(completion_times) > idx else "N/A"
            success = "Yes" if len(success_flags) > idx and success_flags[idx] else "No"
            lines.append(f"{episode_num:7d} | {performance:11.2f} | {reward:7.2f} | {steps:5.0f} | {success:7s}")
        
        return "\n".join(lines)

    def _get_worst_episodes_by_performance(self) -> str:
        """FIXED: Get table of worst episodes ranked by performance score instead of raw reward."""
        if 'performance_scores' not in self.results:
            return "No performance score data available"
        
        performance_scores = np.array(self.results['performance_scores'])
        total_rewards = np.array(self.results.get('total_rewards', []))
        completion_times = np.array(self.results.get('completion_times', []))
        success_flags = np.array(self.results.get('success_flags', []))
        
        # Get bottom 5 episodes by PERFORMANCE SCORE (not raw reward)
        bottom_indices = np.argsort(performance_scores)[:5]
        
        lines = ["Episode | Performance | Reward  | Steps | Success"]
        lines.append("--------|-------------|---------|-------|--------")
        
        for i, idx in enumerate(bottom_indices):
            episode_num = idx + 1
            performance = performance_scores[idx]
            reward = total_rewards[idx] if len(total_rewards) > idx else "N/A"
            steps = completion_times[idx] if len(completion_times) > idx else "N/A"
            success = "Yes" if len(success_flags) > idx and success_flags[idx] else "No"
            lines.append(f"{episode_num:7d} | {performance:11.2f} | {reward:7.2f} | {steps:5.0f} | {success:7s}")
        
        return "\n".join(lines)

    def _calculate_performance_stats(self) -> dict:
        """Calculate performance-based statistics."""
        if 'performance_scores' in self.results:
            perf_scores = np.array(self.results['performance_scores'])
            efficiency_scores = np.array(self.results.get('efficiency_scores', [0]))
            
            return {
                'mean_performance': np.mean(perf_scores),
                'max_performance': np.max(perf_scores),
                'min_performance': np.min(perf_scores),
                'performance_std': np.std(perf_scores),
                'mean_efficiency': np.mean(efficiency_scores),
            }
        return {
            'mean_performance': 0.0,
            'max_performance': 0.0,
            'min_performance': 0.0,
            'performance_std': 0.0,
            'mean_efficiency': 0.0,
        }

    # Continue with remaining methods...