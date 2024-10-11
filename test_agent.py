#!/usr/bin/env python3


import os
import torch
import logging
import numpy as np
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

def test_best_agents(agent, env, num_episodes=10, max_steps=1000):
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

        while not done and step < max_steps:
            actions, _ = agent.get_actions(state)
            next_state, rewards, done, info = env.step(actions)

            # Collect joint errors and rewards
            joint_errors.append(env.joint_errors)
            episode_rewards.append(sum(rewards))  # Sum rewards across agents

            # Check success
            success_count += sum(info['success_per_agent'])

            # Update state and step count
            state = next_state
            step += 1

        # Calculate mean and max joint errors
        mean_joint_error = np.mean(joint_errors, axis=0)
        max_joint_error = np.max(joint_errors, axis=0)
        results['mean_joint_errors'].append(mean_joint_error)
        results['max_joint_errors'].append(max_joint_error)

        # Calculate cumulative and mean rewards
        cumulative_reward = sum(episode_rewards)
        mean_episode_reward = np.mean(episode_rewards)
        results['cumulative_rewards'].append(cumulative_reward)
        results['mean_episode_rewards'].append(mean_episode_reward)

        # Calculate success rate
        success_rate = success_count / (env.num_joints * step)
        results['success_rate'].append(success_rate)

        logging.info(f"Test Episode {episode+1} - Mean Joint Error: {mean_joint_error}, Success Rate: {success_rate:.2f}")

    # Convert results lists to numpy arrays
    for key in results:
        results[key] = np.array(results[key])

    return results

def main():
    logging.basicConfig(level=logging.INFO)

    # Initialize the environment and MAPPOAgent (adjust the config and env setup as needed)
    env = InverseKinematicsEnv()  # Replace with actual environment initialization
    config = {
        'hidden_dim': 128,
        'lr': 1e-4,
        'gamma': 0.99,
        'tau': 0.95,
        'ppo_epochs': 15,
        'batch_size': 64,
        'clip_param': 0.1,
    }
    agent = MAPPOAgent(env, config)

    # Define the base path and model file prefix
    base_path = "models"  # Adjust this if your models are stored elsewhere

    # Load the best-performing agents
    load_best_agents(agent, base_path=base_path)

    # Test the best agents and collect results
    test_results = test_best_agents(agent, env, num_episodes=10, max_steps=1000)

    # Display the results
    for key, values in test_results.items():
        logging.info(f"{key}: {values}")

    # Calculate average success rate and joint error across all episodes
    avg_success_rate = np.mean(test_results['success_rate'], axis=0)
    avg_joint_error = np.mean(test_results['mean_joint_errors'], axis=0)

    logging.info(f"Average Success Rate per Joint: {avg_success_rate}")
    logging.info(f"Average Joint Error per Joint: {avg_joint_error}")

if __name__ == "__main__":
    main()
