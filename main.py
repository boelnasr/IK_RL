#!/usr/bin/env python3

import numpy as np
from ik_solver.environment import InverseKinematicsEnv
from ik_solver.mappo import MAPPOAgent
from config import config
import logging
import traceback

def main():
    """
    Main function that initializes the environment and agent, and runs the training process.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    env = None  # Initialize `env` to ensure it's available in the `finally` block

    try:
        # Initialize the PyBullet environment with the robot (e.g., "kuka_iiwa")
        robot_name = config.get('robot_name', 'kuka_iiwa')
        logging.info(f"Initializing environment with robot: {robot_name}")

        # Initialize the environment
        env = InverseKinematicsEnv(robot_name=robot_name)

        # Initialize the MAPPO agent with the environment and configuration
        agent = MAPPOAgent(env, config)

        # Train the agent
        num_episodes = config.get('num_episodes', 5000)
        max_steps_per_episode = config.get('max_steps_per_episode', 300)
        logging.info(f"Starting training for {num_episodes} episodes with a maximum of {max_steps_per_episode} steps per episode...")

        agent.train(num_episodes=num_episodes, max_steps_per_episode=max_steps_per_episode)

        # After training, optionally test the agent if specified in the configuration
        if config.get('test_agent_after_training', True):
            num_tests = config.get('num_tests', 5)
            logging.info(f"Testing agent for {num_tests} test episodes after training...")
            agent.test_agent(env, num_episodes=num_tests)  # Pass `env` and use `num_episodes`

    except Exception as e:
        # Log any errors that occur during the setup or training process
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()  # This will print the full error stack trace to the console

    finally:
        # Ensure that the environment is alw.ays closed properly
        if env is not None:
            env.close()  # Properly close the environment
        logging.info("Environment closed.")

if __name__ == "__main__":
    main()
