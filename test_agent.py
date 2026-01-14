#!/usr/bin/env python3


import os
import torch
import logging
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ik_solver.mappo import MAPPOAgent
from ik_solver.environment import InverseKinematicsEnv


# Ensure that the logging is set up
logging.basicConfig(level=logging.INFO)

def load_best_agents(agent, base_path="models", agent_prefix="best_agent_joint_"):
    """
    Load the best agents' state dictionaries from the saved files.

    Args:
        agent (MAPPOAgent): The MAPPO agent instance containing the agents.
        base_path (str): Base directory where the models are stored.
        agent_prefix (str): Prefix of the model filenames.
    """
    # Create a list to hold the file paths for the best models
    best_agent_paths = [os.path.join(base_path, f"{agent_prefix}{i}.pth") for i in range(agent.num_agents)]

    for agent_idx, path in enumerate(best_agent_paths):
        if os.path.exists(path):
            logging.info(f"Loading best agent for joint {agent_idx} from {path}")
            try:
                # Use strict=False to ignore size mismatches
                agent.agents[agent_idx].load_state_dict(torch.load(path), strict=False)
                logging.info(f"Successfully loaded best agent for joint {agent_idx}")
            except Exception as e:
                logging.error(f"Error loading state dict for joint {agent_idx} from {path}: {e}")
        else:
            logging.warning(f"Best agent file for joint {agent_idx} not found at {path}. Skipping.")

def test_best_agents(
    agent,
    env,
    num_episodes=10,
    max_steps=1000,
    *,
    position_threshold=0.05,
    orientation_threshold=0.1
):
    """
    Test the best-performing agents in the environment.

    Args:
        agent (MAPPOAgent): The MAPPO agent instance containing the trained agents.
        env (InverseKinematicsEnv): The testing environment.
        num_episodes (int): Number of testing episodes.
        max_steps (int): Maximum number of steps per episode.

    Returns:
        dict: A dictionary containing success rate and joint error metrics.
    """
    # Initialize results dictionary
    results = {
        'mean_joint_errors': [],
        'max_joint_errors': [],
        'success_rate': [],
        'cumulative_rewards': [],
        'mean_episode_rewards': [],
        'position_error_mean_abs': [],
        'orientation_error_mean_abs': [],
        'custom_success_rate': [],
    }

    # Test the agent over multiple episodes
    for episode in range(num_episodes):
        logging.info(f"Starting test episode {episode+1}/{num_episodes}")
        state = env.reset()
        done = False
        step = 0
        episode_rewards = []
        joint_errors = []
        success_count = 0
        custom_success_count = 0
        position_errors = []
        orientation_errors = []

        while not done and step < max_steps:
            actions, _, _ = agent.get_actions(state, eval_mode=True)
            next_state, rewards, done, info = env.step(actions)

            # Collect joint errors and rewards
            if hasattr(env, "joint_errors"):
                joint_errors.append(np.array(env.joint_errors, copy=True))
            episode_rewards.append(sum(rewards))  # Sum rewards across agents

            # Track position and orientation errors per step
            pos_err = np.array(getattr(env, "position_error", np.zeros(3)), dtype=np.float32).flatten()
            ori_err = np.array(getattr(env, "orientation_error", np.zeros(3)), dtype=np.float32).flatten()
            if pos_err.size < 3:
                pos_err = np.pad(pos_err, (0, 3 - pos_err.size), constant_values=0.0)
            else:
                pos_err = pos_err[:3]
            if ori_err.size < 3:
                ori_err = np.pad(ori_err, (0, 3 - ori_err.size), constant_values=0.0)
            else:
                ori_err = ori_err[:3]
            position_errors.append(pos_err)
            orientation_errors.append(ori_err)

            # Check success using environment info
            per_joint_success = info.get('success_per_joint', info.get('success_per_agent', []))
            success_count += sum(per_joint_success)

            # Custom success criteria based on pose error norms
            pos_norm = np.linalg.norm(pos_err)
            ori_norm = np.linalg.norm(ori_err)
            if pos_norm <= position_threshold and ori_norm <= orientation_threshold:
                custom_success_count += 1

            # Update state and step count
            state = next_state
            step += 1

        # Calculate mean and max joint errors
        if joint_errors:
            stacked_joint_errors = np.stack(joint_errors, axis=0)
            mean_joint_error = np.mean(stacked_joint_errors, axis=0)
            max_joint_error = np.max(stacked_joint_errors, axis=0)
        else:
            mean_joint_error = np.zeros(env.num_joints)
            max_joint_error = np.zeros(env.num_joints)
        results['mean_joint_errors'].append(mean_joint_error)
        results['max_joint_errors'].append(max_joint_error)

        # Calculate cumulative and mean rewards
        cumulative_reward = sum(episode_rewards) if episode_rewards else 0.0
        mean_episode_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        results['cumulative_rewards'].append(cumulative_reward)
        results['mean_episode_rewards'].append(mean_episode_reward)

        # Calculate success rate
        success_rate = success_count / (env.num_joints * step) if step > 0 else 0.0
        results['success_rate'].append(success_rate)
        custom_rate = custom_success_count / step if step > 0 else 0.0
        results['custom_success_rate'].append(custom_rate)

        # Aggregate per-axis position/orientation errors (mean absolute over episode)
        if position_errors:
            pos_stack = np.stack(position_errors, axis=0)
            mean_abs_pos = np.mean(np.abs(pos_stack), axis=0)
        else:
            mean_abs_pos = np.zeros(3, dtype=np.float32)
        if orientation_errors:
            ori_stack = np.stack(orientation_errors, axis=0)
            mean_abs_ori = np.mean(np.abs(ori_stack), axis=0)
        else:
            mean_abs_ori = np.zeros(3, dtype=np.float32)
        results['position_error_mean_abs'].append(mean_abs_pos)
        results['orientation_error_mean_abs'].append(mean_abs_ori)

        print(
            f"Test Episode {episode+1} - Mean Joint Error: {mean_joint_error}, "
            f"Env Success Rate: {success_rate:.2f}, "
            f"Custom Success Rate: {custom_rate:.2f}"
        )

    # Convert results lists to numpy arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


def plot_error_metrics(results, output_dir):
    """Plot per-axis position and orientation error trends."""
    os.makedirs(output_dir, exist_ok=True)
    episodes = np.arange(1, len(results['position_error_mean_abs']) + 1)
    axis_labels = ['X', 'Y', 'Z']

    def _plot(data, title, ylabel, filename):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        for axis_idx in range(data.shape[1]):
            ax.plot(episodes, data[:, axis_idx], label=f'{axis_labels[axis_idx]} axis')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)

    _plot(
        results['position_error_mean_abs'],
        'Mean Absolute Position Error per Axis',
        'Mean |Position Error| (m)',
        'position_error_axes.png'
    )
    _plot(
        results['orientation_error_mean_abs'],
        'Mean Absolute Orientation Error per Axis',
        'Mean |Orientation Error| (rad)',
        'orientation_error_axes.png'
    )

def main():
    logging.basicConfig(level=logging.INFO)

    # Import actual config used during training
    from config import config as training_config

    # Initialize FRESH TEST environment with episode counter reset to 0
    # This ensures thresholds start at max_success_threshold (0.1m) instead of min (0.01m)
    env = InverseKinematicsEnv()
    env.episode_number = 0  # Reset episode counter for testing with reasonable thresholds

    agent = MAPPOAgent(env, training_config)

    # Define the base path and model file prefix
    base_path = "models"  # Adjust this if your models are stored elsewhere

    # Load the best-performing agents
    load_best_agents(agent, base_path=base_path)

    # Test the best agents and collect results
    test_results = test_best_agents(agent, env, num_episodes=100, max_steps=1000)

    # Display the results
    for key, values in test_results.items():
        logging.info(f"{key}: {values}")   

    # Calculate average success rate and joint error across all episodes
    avg_success_rate = float(np.mean(test_results['success_rate']))
    avg_custom_success = float(np.mean(test_results['custom_success_rate']))
    avg_joint_error = np.mean(test_results['mean_joint_errors'], axis=0)

    print(f"Average Env Success Rate: {avg_success_rate:.3f}")
    print(f"Average Custom Success Rate: {avg_custom_success:.3f}")
    print(f"Average Joint Error per Joint: {avg_joint_error}")

    # Generate plots for position and orientation error metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("test_reports", timestamp)
    plot_error_metrics(test_results, output_dir)
    logging.info(f"Saved error plots to {output_dir}")

if __name__ == "__main__":
    main()
