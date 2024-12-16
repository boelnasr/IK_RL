#!/usr/bin/env python3

import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import logging
import time
import traceback
from collections import defaultdict
import pandas as pd
from pathlib import Path
import torch
from typing import Dict, List, Tuple    
import json
from datetime import datetime
import os
from ik_solver.environment import InverseKinematicsEnv
from ik_solver.mappo import MAPPOAgent

class AgentTester:
    def __init__(self, agent, env, base_path="test_results"):
        self.agent = agent
        self.env = env
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics = []
        self.episode_metrics = defaultdict(list)
        self.convergence_times = []  # Convergence time tracking

        self.setup_logging()
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = self.base_path / f"session_{self.timestamp}"
        self.session_path.mkdir(exist_ok=True)

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(self.base_path / 'testing.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)

    # def run_systematic_tests(self, num_episodes=100):
    #     try:
    #         self.logger.info("Starting systematic testing suite")
            
    #         results = {
    #             'basic_performance': self.test_basic_performance(num_episodes),
    #             'speed_tests': self.test_speed(num_episodes)
    #         }
            
    #         self.save_results(results)
    #         return results
            
    #     except Exception as e:
    #         self.logger.error(f"Error in systematic testing: {str(e)}")
    #         raise

    def run_systematic_tests(self, num_episodes=100):
        try:
            self.logger.info("Starting systematic testing suite")
            
            results = {
                'basic_performance': self.test_basic_performance(num_episodes),
                'speed_tests': self.test_speed(num_episodes),
                'random_joint_tests': self.test_random_joint_sets(num_episodes),
            }
            
            self.save_results(results)
            return results
        except Exception as e:
            self.logger.error(f"Error in systematic testing: {str(e)}")
            raise

    def test_basic_performance(self, num_episodes):
        results = {
            'success_rate': [],
            'completion_times': [],
            'final_distances': [],
            'trajectory_smoothness': [],
            'joint_errors': {f'joint_{i}': [] for i in range(self.env.num_joints)},
            'min_joint_errors': {f'joint_{i}': [] for i in range(self.env.num_joints)},  # Added for min joint errors
            'position_errors': np.zeros((num_episodes, 3)),
            'positions': [],
            'target_positions': [],
            'orientations': [],
            'target_orientations': [],
            'orientation_errors': []
        }

        for episode in range(num_episodes):
            state = self.env.reset()
            target_position, target_orientation = self.env.get_target_pose()
            target_joint_angles = self.compute_target_joint_angles(target_position, target_orientation)

            done = False
            steps = 0
            trajectory = []
            episode_joint_errors = {f'joint_{i}': [] for i in range(self.env.num_joints)}
            episode_positions = []
            episode_orientations = []
            episode_orientation_errors = []

            while not done and steps < self.env.max_episode_steps:
                actions, _ = self.agent.get_actions(state, eval_mode=True)
                next_state, rewards, done, info = self.env.step(actions)

                # Track positions and orientations
                current_pos, current_orient = self.env.get_end_effector_pose()
                episode_positions.append(current_pos)
                episode_orientations.append(current_orient)

                # Track orientation errors
                orientation_error = self.env.compute_orientation_difference(current_orient, target_orientation)
                episode_orientation_errors.append(orientation_error)

                # Track joint errors
                for i in range(self.env.num_joints):
                    current_angle = p.getJointState(self.env.robot_id, self.env.joint_indices[i])[0]
                    target_angle = target_joint_angles[i]
                    joint_error = abs(current_angle - target_angle)
                    episode_joint_errors[f'joint_{i}'].append(joint_error)

                trajectory.append(current_pos)
                state = next_state
                steps += 1

            # Store episode results
            results['positions'].append(episode_positions)
            results['orientations'].append(episode_orientations)
            results['target_positions'].append(target_position)
            results['target_orientations'].append(target_orientation)
            results['orientation_errors'].append(episode_orientation_errors)
            results['success_rate'].append(info['success'])
            results['completion_times'].append(steps)
            results['final_distances'].append(info['current_distance'])
            results['trajectory_smoothness'].append(self.calculate_trajectory_smoothness(trajectory))

            # Calculate final position error
            final_pos = np.array(trajectory[-1])
            target_pos = np.array(target_position)
            results['position_errors'][episode] = final_pos - target_pos

            # Store minimum and mean joint errors for each joint
            for joint_id, errors in episode_joint_errors.items():
                if errors:
                    results['joint_errors'][joint_id].append(np.mean(errors))  # Mean joint error
                    results['min_joint_errors'][joint_id].append(min(errors))  # Minimum joint error

            if (episode + 1) % 5 == 0:
                self.logger.info(f"Episode {episode + 1}/{num_episodes} complete")

        return results


    # def test_basic_performance(self, num_episodes):
    #     results = {
    #         'success_rate': [],  # Store scalar success status
    #         'completion_times': [],
    #         'final_distances': [],
    #         'trajectory_smoothness': [],
    #         'joint_errors': {f'joint_{i}': [] for i in range(self.env.num_joints)},
    #         'min_joint_errors': {f'joint_{i}': [] for i in range(self.env.num_joints)},  # Added for min joint errors
    #         'position_errors': np.zeros((num_episodes, 3)),
    #         'positions': [],
    #         'target_positions': [],
    #         'orientations': [],
    #         'target_orientations': [],
    #         'orientation_errors': []
    #     }

    #     for episode in range(num_episodes):
    #         state = self.env.reset()
    #         target_position, target_orientation = self.env.get_target_pose()
    #         target_joint_angles = self.compute_target_joint_angles(target_position, target_orientation)

    #         steps = 0
    #         trajectory = []
    #         episode_joint_errors = {f'joint_{i}': [] for i in range(self.env.num_joints)}
    #         episode_positions = []
    #         episode_orientations = []
    #         episode_orientation_errors = []

    #         success = False  # Initialize success for the episode

    #         while steps < self.env.max_episode_steps:  # Custom termination condition
    #             actions, _ = self.agent.get_actions(state, eval_mode=True)
    #             next_state, rewards, _, info = self.env.step(actions)

    #             # Custom success condition
    #             position_error = np.linalg.norm(np.array(target_position) - np.array(self.env.get_end_effector_pose()[0]))
    #             orientation_error = np.linalg.norm(self.env.compute_orientation_difference(
    #                 self.env.get_end_effector_pose()[1], target_orientation))

    #             if position_error < self.env.success_threshold and orientation_error < self.env.orientation_threshold:
    #                 success = True
    #                 break  # Terminate the episode if success is achieved

    #             # Track positions and orientations
    #             current_pos, current_orient = self.env.get_end_effector_pose()
    #             episode_positions.append(current_pos)
    #             episode_orientations.append(current_orient)

    #             # Track orientation errors
    #             orientation_error = self.env.compute_orientation_difference(current_orient, target_orientation)
    #             episode_orientation_errors.append(orientation_error)

    #             # Track joint errors
    #             for i in range(self.env.num_joints):
    #                 current_angle = p.getJointState(self.env.robot_id, self.env.joint_indices[i])[0]
    #                 target_angle = target_joint_angles[i]
    #                 joint_error = abs(current_angle - target_angle)
    #                 episode_joint_errors[f'joint_{i}'].append(joint_error)

    #             trajectory.append(current_pos)
    #             state = next_state
    #             steps += 1

    #         # Store episode results
    #         results['positions'].append(episode_positions)
    #         results['orientations'].append(episode_orientations)
    #         results['target_positions'].append(target_position)
    #         results['target_orientations'].append(target_orientation)
    #         results['orientation_errors'].append(episode_orientation_errors)
    #         results['success_rate'].append(success)  # Append scalar success
    #         results['completion_times'].append(steps)
    #         results['final_distances'].append(position_error)
    #         results['trajectory_smoothness'].append(self.calculate_trajectory_smoothness(trajectory))

    #         # Calculate final position error
    #         final_pos = np.array(trajectory[-1]) if len(trajectory) > 0 else np.zeros(3)
    #         target_pos = np.array(target_position)
    #         results['position_errors'][episode] = final_pos - target_pos

    #         # Store minimum and mean joint errors for each joint
    #         for joint_id, errors in episode_joint_errors.items():
    #             if errors:
    #                 results['joint_errors'][joint_id].append(np.mean(errors))  # Mean joint error
    #                 results['min_joint_errors'][joint_id].append(min(errors))  # Minimum joint error

    #         if (episode + 1) % 5 == 0:
    #             self.logger.info(f"Episode {episode + 1}/{num_episodes} complete")

    #     return results

    def test_random_joint_sets(self, num_episodes=100):
        results = {
            'success_rate': [],
            'completion_times': [],
            'final_distances': [],
            'convergence_times': [],
            'policy_entropies': [],
        }

        for episode in range(num_episodes):
            # Generate random joint angles
            random_joint_angles = self.generate_random_joint_angles()
            self.env.reset_to_joint_positions(random_joint_angles)

            target_position, target_orientation = self.env.get_end_effector_pose()

            # Reset to target for test start
            state = self.env.reset_to_target(target_position, target_orientation)

            done = False
            steps = 0
            converged = False
            while not done and steps < self.env.max_episode_steps:
                actions, policy_info = self.agent.get_actions(state, eval_mode=True)
                entropy = policy_info.get("entropy", 0.0)
                self.entropies.append(entropy)

                next_state, _, done, info = self.env.step(actions)

                # Check convergence
                current_position, current_orientation = self.env.get_end_effector_pose()
                position_error = np.linalg.norm(np.array(current_position) - np.array(target_position))
                orientation_error = np.linalg.norm(self.env.compute_orientation_difference(
                    current_orientation, target_orientation))

                if not converged and position_error < self.env.success_threshold and \
                        orientation_error < self.env.orientation_threshold:
                    converged = True
                    results['convergence_times'].append(steps)

                state = next_state
                steps += 1

            if not converged:
                results['convergence_times'].append(self.env.max_episode_steps)

            results['policy_entropies'].append(np.mean(self.entropies))
            results['completion_times'].append(steps)
            results['success_rate'].append(info['success'])
            results['final_distances'].append(info['current_distance'])

        return results

    def generate_random_joint_angles(self):
        """Generate random joint angles within valid joint limits."""
        lower_limits, upper_limits = zip(*self.env.joint_limits)
        return np.random.uniform(lower_limits, upper_limits)


    def test_speed(self, num_episodes):
        results = {
            'raw_results': defaultdict(list),
            'aggregate_stats': {}
        }
        
        for episode in range(num_episodes):
            state = self.env.reset()
            start_time = time.time()
            trajectory = []
            velocities = []
            
            initial_pos, _ = self.env.get_end_effector_pose()
            target_pos = self.env.target_position
            direct_distance = np.linalg.norm(target_pos - initial_pos)
            
            done = False
            steps = 0
            
            while not done and steps < self.env.max_episode_steps:
                actions, _ = self.agent.get_actions(state, eval_mode=True)
                next_state, _, done, _ = self.env.step(actions)
                
                current_pos, _ = self.env.get_end_effector_pose()
                trajectory.append(current_pos)
                
                if len(trajectory) >= 2:
                    velocity = np.linalg.norm(current_pos - trajectory[-2]) / self.env.sim_timestep
                    velocities.append(velocity)
                
                state = next_state
                steps += 1
            
            reaching_time = time.time() - start_time
            trajectory = np.array(trajectory)
            path_length = sum(np.linalg.norm(trajectory[i+1] - trajectory[i]) 
                            for i in range(len(trajectory)-1))
            path_efficiency = direct_distance / (path_length + 1e-6)
            
            results['raw_results']['reaching_times'].append(reaching_time)
            results['raw_results']['path_efficiencies'].append(path_efficiency)
            results['raw_results']['velocities'].append(np.mean(velocities) if velocities else 0)
            
        # Calculate aggregate statistics
        for key, values in results['raw_results'].items():
            results['aggregate_stats'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
            
        return results

    def save_results(self, results):
        try:
            save_dir = self.base_path / f"test_session_{self.timestamp}"
            save_dir.mkdir(exist_ok=True)
            
            metrics_df = pd.DataFrame(self.metrics)
            metrics_df.to_csv(save_dir / 'metrics.csv', index=False)
            
            with open(save_dir / 'test_results.json', 'w') as f:
                json.dump(self.convert_to_serializable(results), f, indent=2)
                
            self.generate_and_save_plots(results, save_dir)
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise


    def generate_and_save_plots(self, results, save_dir):
        """
        Generate and save various plots based on test results.
        """
        plots_dir = save_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        # Generate all plots
        self._plot_basic_performance(results['basic_performance'], plots_dir)
        self._plot_speed_results(results['speed_tests'], plots_dir)
        self._plot_joint_errors(results['basic_performance'], plots_dir)
        self._plot_position_error(results['basic_performance'], plots_dir)
        self._plot_position_tracking(results['basic_performance'], plots_dir)
        self._plot_orientation_error(results['basic_performance'], plots_dir)
        self._plot_minimum_joint_errors(results['basic_performance'], plots_dir)
        self._plot_success_rate_per_agent(results['basic_performance'], plots_dir)
        self._plot_convergence_time(results, plots_dir)


    def _plot_basic_performance(self, basic_results, plots_dir):
        metrics_data = {
            'Success Rate': np.mean(basic_results['success_rate']),
            'Avg Steps': np.mean(basic_results['completion_times']),
            'Avg Distance': np.mean(basic_results['final_distances']),
            'Avg Smoothness': np.mean(basic_results['trajectory_smoothness'])
        }
        
        plt.figure(figsize=(10, 6))
        plt.bar(metrics_data.keys(), metrics_data.values(), color='#2E86AB')
        plt.title('Basic Performance Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'basic_performance.png')
        plt.close()

    def _plot_speed_results(self, speed_results, plots_dir):
        if not speed_results['aggregate_stats']:
            return
            
        plt.figure(figsize=(10, 6))
        
        metrics = list(speed_results['aggregate_stats'].keys())
        means = [speed_results['aggregate_stats'][m]['mean'] for m in metrics]
        stds = [speed_results['aggregate_stats'][m]['std'] for m in metrics]
        
        plt.bar(metrics, means, yerr=stds, capsize=5)
        plt.title('Speed Performance Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'speed_metrics.png')
        plt.close()

    def _plot_joint_errors(self, basic_results, plots_dir):
        try:
            if 'joint_errors' not in basic_results:
                self.logger.error("No joint error data found")
                return

            num_joints = len(basic_results['joint_errors'])
            episodes = range(len(next(iter(basic_results['joint_errors'].values()))))

            fig, axes = plt.subplots(num_joints, 1, figsize=(12, 4 * num_joints), dpi=300)
            if num_joints == 1:
                axes = [axes]

            for i, (joint_id, errors) in enumerate(basic_results['joint_errors'].items()):
                ax = axes[i]
                errors = np.array(errors)

                # Plot minimum error
                min_error = np.min(errors)
                min_error_idx = np.argmin(errors)
                ax.plot(min_error_idx, min_error, 'g*', markersize=15, label=f'Minimum Error: {min_error:.4f}')
                ax.axhline(y=min_error, color='g', linestyle='--', alpha=0.5)

                # Plot all raw errors
                ax.plot(episodes, errors, 'b-', alpha=0.6, label='Joint Error')

                # Annotate the minimum value
                stats_text = f'Minimum Error: {min_error:.4f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax.set_title(f'Joint {i + 1} Error')
                ax.set_xlabel('Episodes')
                ax.set_ylabel('Error (radians)')
                ax.grid(True, alpha=0.3)
                ax.legend()

            plt.tight_layout()
            plt.savefig(plots_dir / 'joint_errors_analysis.png')
            plt.close(fig)
        except Exception as e:
            self.logger.error(f"Error in joint error plotting: {str(e)}")
            plt.close('all')


    def compute_target_joint_angles(self, target_position, target_orientation):
        try:
            target_joint_angles = p.calculateInverseKinematics(
                bodyUniqueId=self.env.robot_id,
                endEffectorLinkIndex=self.env.joint_indices[-1],
                targetPosition=target_position,
                targetOrientation=target_orientation,
                lowerLimits=[limit[0] for limit in self.env.joint_limits],
                upperLimits=[limit[1] for limit in self.env.joint_limits],
                jointRanges=[limit[1] - limit[0] for limit in self.env.joint_limits],
                restPoses=[0.0] * self.env.num_joints
            )
            return np.array(target_joint_angles[:self.env.num_joints])
        except Exception as e:
            self.logger.error(f"Error computing target joint angles: {str(e)}")
            return np.zeros(self.env.num_joints)


    def _plot_orientation_error(self, basic_results, plots_dir):
        try:
            orientation_errors = basic_results.get('orientation_errors', [])
            
            if len(orientation_errors) == 0:
                self.logger.error("No orientation error data found")
                return

            # Normalize orientation errors (pad sequences to have equal length)
            max_length = max(len(episode) for episode in orientation_errors)
            padded_errors = np.array([
                np.pad(episode, ((0, max_length - len(episode)), (0, 0)), mode='constant', constant_values=0)
                if len(episode) < max_length else np.array(episode)
                for episode in orientation_errors
            ])

            # Check if padded_errors is valid
            if padded_errors.ndim != 3:
                self.logger.error("Padded orientation errors do not have the correct shape")
                return

            fig, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=300, sharex=True)
            axis_labels = ['Roll', 'Pitch', 'Yaw']
            colors = ['tab:red', 'tab:blue', 'tab:green']
            time_steps = np.arange(padded_errors.shape[1])

            for i, ax in enumerate(axes):
                mean_errors = padded_errors[:, :, i].mean(axis=0)
                std_errors = padded_errors[:, :, i].std(axis=0)

                ax.plot(time_steps, mean_errors, label=f'{axis_labels[i]} Error', color=colors[i], linewidth=1.5)
                #ax.fill_between(time_steps, mean_errors - std_errors, mean_errors + std_errors,
                                # alpha=0.2, color=colors[i], label='Standard Deviation')

                ax.set_ylabel(f'{axis_labels[i]} Error')
                ax.grid(True, alpha=0.3)
                ax.legend()

            axes[-1].set_xlabel('Time Steps')
            plt.suptitle('Orientation Error Tracking', fontsize=14)
            plt.tight_layout()
            plt.savefig(plots_dir / 'orientation_errors.png')
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error in orientation error plotting: {str(e)}")
            plt.close('all')



    def calculate_trajectory_smoothness(self, trajectory):
        if len(trajectory) < 4:
            return 0.0
        velocity = np.diff(trajectory, axis=0)
        acceleration = np.diff(velocity, axis=0)
        jerk = np.diff(acceleration, axis=0)
        return -np.sqrt(np.mean(jerk**2))

    def convert_to_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj

    def _plot_position_tracking(self, basic_results, plots_dir):
        try:
            positions = basic_results['positions']
            target_positions = np.array(basic_results['target_positions'])

            # Normalize positions data (pad sequences to have equal length)
            max_length = max(len(episode) for episode in positions)
            padded_positions = np.array([
                np.pad(episode, ((0, max_length - len(episode)), (0, 0)), mode='edge')
                for episode in positions
            ])

            fig, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=300, sharex=True)
            axis_labels = ['X', 'Y', 'Z']
            colors = ['tab:red', 'tab:blue', 'tab:green']
            time_steps = np.arange(padded_positions.shape[1])

            for i, ax in enumerate(axes):
                mean_positions = padded_positions[:, :, i].mean(axis=0)
                std_positions = padded_positions[:, :, i].std(axis=0)

                ax.plot(time_steps, mean_positions, label=f'Actual {axis_labels[i]}', color=colors[i], linewidth=1.5)
                ax.fill_between(time_steps, mean_positions - std_positions, mean_positions + std_positions,
                                alpha=0.2, color=colors[i], label='Standard Deviation')
                ax.axhline(y=target_positions[0, i], color='black', linestyle='--', label='Target')

                final_error = np.abs(mean_positions[-1] - target_positions[0, i])
                stats_text = f'Final Error: {final_error:.4f}\n'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax.set_ylabel(f'{axis_labels[i]} (units)')
                ax.grid(True, alpha=0.3)
                ax.legend()

            axes[-1].set_xlabel('Time Steps')
            plt.suptitle('End-Effector Position Tracking', fontsize=14)
            plt.tight_layout()
            plt.savefig(plots_dir / 'position_tracking.png')
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error in position tracking plotting: {str(e)}")
            plt.close('all')


    def save_figure(self, fig, filename):
        """Save a figure to the session directory."""
        try:
            fig.savefig(self.session_path / f"{filename}.png", dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved plot: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save plot {filename}: {str(e)}")

    def _plot_position_error(self, basic_results, plots_dir):
        try:
            position_errors = basic_results['position_errors']
            if len(position_errors) == 0:
                self.logger.error("No position error data found")
                return

            fig, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=300, sharex=True)
            axis_labels = ['X', 'Y', 'Z']
            colors = ['tab:red', 'tab:blue', 'tab:green']
            episodes = np.arange(1, len(position_errors) + 1)

            for i, ax in enumerate(axes):
                errors = position_errors[:, i]
                window = min(10, len(episodes) // 10) if len(episodes) > 50 else 1
                
                # Raw errors
                ax.plot(episodes, errors, label=f'{axis_labels[i]}-Axis Error', 
                    color=colors[i], alpha=0.6, linewidth=1.5)
                
                # Moving average
                if window > 1:
                    smoothed_errors = np.convolve(errors, np.ones(window) / window, mode='valid')
                    ax.plot(episodes[window-1:], smoothed_errors, 
                        label=f'Moving Average', color='black', linestyle='--')

                # Min value marker
                min_error = np.min(errors)
                min_idx = np.argmin(errors)
                ax.plot(min_idx + 1, min_error, 'g*', markersize=15, 
                    label=f'Minimum: {min_error:.4f}')
                
                # Stats
                stats_text = (f'Mean: {np.mean(errors):.4f}\n'
                            f'Std: {np.std(errors):.4f}\n'
                            f'Min: {min_error:.4f}\n'
                            f'Max: {np.max(errors):.4f}')
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_ylabel(f'{axis_labels[i]}-Axis Error')
                ax.grid(True, alpha=0.3)
                ax.legend()

            axes[-1].set_xlabel('Episodes')
            plt.tight_layout()
            plt.savefig(plots_dir / 'position_errors.png')
            plt.close(fig)

        except Exception as e:
            self.logger.error(f"Error in position error plotting: {str(e)}")
            plt.close('all')

    def _plot_minimum_joint_errors(self, basic_results, plots_dir):
        try:
            min_joint_errors = basic_results.get('min_joint_errors', {})
            if not min_joint_errors:
                self.logger.error("No minimum joint error data found")
                return

            num_joints = len(min_joint_errors)
            episodes = np.arange(1, len(next(iter(min_joint_errors.values()))) + 1)

            fig, axes = plt.subplots(num_joints, 1, figsize=(12, 4 * num_joints), dpi=300)

            if num_joints == 1:
                axes = [axes]

            for i, (joint_id, errors) in enumerate(min_joint_errors.items()):
                ax = axes[i]
                errors = np.array(errors)

                # Plot minimum errors
                ax.plot(episodes, errors, label='Minimum Error', color='tab:blue', linewidth=1.5)

                # Highlight the absolute minimum
                min_error = np.min(errors)
                min_error_idx = np.argmin(errors)
                ax.scatter(min_error_idx + 1, min_error, color='red', s=100, label='Best Performance')

                # Add annotations
                stats_text = (
                    f'Min: {min_error:.4f}\n'
                    f'Final: {errors[-1]:.4f}'
                )
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

                ax.set_title(f'Joint {i + 1} Minimum Errors Over Time')
                ax.set_xlabel('Episodes')
                ax.set_ylabel('Minimum Error (radians)')
                ax.legend()
                ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(plots_dir /  'minimum_joint_errors')
            plt.close(fig)
        except Exception as e:
            self.logger.error(f"Error plotting minimum joint errors: {str(e)}")
            raise

    def _plot_success_rate_per_agent(self, basic_results, plots_dir):
        """
        Plot the success rate per agent (joint) as a bar chart.

        Args:
            basic_results (dict): The results from the basic performance test.
            plots_dir (Path): Directory to save the plot.
        """
        try:
            # Extract success rates from basic results
            success_rates = []
            for joint_id in basic_results['joint_errors']:
                success_count = sum(
                    1 for errors in basic_results['joint_errors'][joint_id] if np.mean(errors) < self.env.success_threshold
                )
                success_rate = success_count / len(basic_results['joint_errors'][joint_id])
                success_rates.append(success_rate)

            joint_labels = [f'Joint {i + 1}' for i in range(len(success_rates))]

            # Plot the bar chart
            plt.figure(figsize=(10, 6))
            plt.bar(joint_labels, success_rates, color='tab:blue', alpha=0.8)
            plt.ylim(0, 1)
            plt.title('Success Rate Per Agent (Joint)', fontsize=16)
            plt.xlabel('Agents (Joints)', fontsize=14)
            plt.ylabel('Success Rate', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.6)

            # Add success rate values above the bars
            for i, rate in enumerate(success_rates):
                plt.text(i, rate + 0.02, f'{rate:.2f}', ha='center', fontsize=12)

            # Save the plot
            plt.tight_layout()
            plt.savefig(plots_dir / 'success_rate_per_agent.png')
            plt.close()
            self.logger.info("Saved success rate per agent plot.")

        except Exception as e:
            self.logger.error(f"Error plotting success rate per agent: {str(e)}")
            plt.close('all')

    def _plot_convergence_time(self, results, plots_dir):
        convergence_times = results.get('convergence_times', [])
        if not convergence_times:
            self.logger.error("No convergence time data found")
            return

        plt.figure(figsize=(10, 6))
        plt.hist(convergence_times, bins=20, color='tab:green', alpha=0.7)
        plt.axvline(x=np.mean(convergence_times), color='red', linestyle='--', label=f'Mean: {np.mean(convergence_times):.2f}')
        plt.xlabel('Convergence Time (steps)')
        plt.ylabel('Frequency')
        plt.title('Convergence Time Distribution')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'convergence_time.png')
        plt.close()


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_results_dir = Path(f"test_results_{timestamp}")
        base_results_dir.mkdir(parents=True, exist_ok=True)
        
        robot_config = {
            'robot_name': 'xarm',
            'sim_timestep': 1./240.,
            'max_episode_steps': 1000
        }
        
        logger.info("Initializing environment and agent...")
        env = InverseKinematicsEnv(**robot_config)
        agent = MAPPOAgent(env=env, config={
            'hidden_dim': 256,
            'lr': 1e-4,
            'gamma': 0.99,
            'tau': 0.95
        })
        
        logger.info("Setting up testing framework...")
        tester = AgentTester(agent=agent, env=env, base_path=base_results_dir)
        
        test_episodes = 100
        logger.info(f"Starting systematic testing with {test_episodes} episodes...")
        results = tester.run_systematic_tests(num_episodes=test_episodes)
        
        logger.info("Testing completed successfully")
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        try:
            env.close()
            logger.info("Environment closed successfully")
        except:
            pass
        
        logger.info("Testing session completed")