#!/usr/bin/env python3

import numpy as np
import logging
import traceback
from ik_solver.environment import InverseKinematicsEnv
from ik_solver.mappo import MAPPOAgent
from config import config

def main():
    """
    Main function to initialize the environment, train the agent, and optionally test it.
    """
    # Configure logging
    logging.basicConfig(
        filename='training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    env = None  # Ensure `env` is defined for the `finally` block
    try:
        # Retrieve configuration parameters
        robot_name = config.get('robot_name', 'xarm')
        num_episodes = config.get('num_episodes',2000)
        max_steps_per_episode = config.get('max_steps_per_episode', 5000)
        test_agent_after_training = config.get('test_agent_after_training', True)
        num_tests = config.get('num_tests', 5)

        # Log the configuration
        logging.info(f"Initializing environment with robot: {robot_name}")
        logging.info(f"Training configuration: num_episodes={num_episodes}, max_steps_per_episode={max_steps_per_episode}")

        # Initialize the environment
        env = InverseKinematicsEnv(robot_name=robot_name)
        logging.info("Environment initialized successfully.")

        # Initialize the MAPPO agent
        agent = MAPPOAgent(env, config)
        logging.info("MAPPOAgent initialized successfully.")

        # Start training
        logging.info("Starting training...")
        agent.train()
        logging.info("Training completed successfully.")

        # Test the agent after training if specified
        if test_agent_after_training:
            logging.info(f"Starting testing for {num_tests} episodes...")
            agent.test_agent(env, num_episodes=num_tests)
            logging.info("Testing completed successfully.")

    except Exception as e:
        # Log and print the error stack trace
        logging.error(f"An error occurred during training or testing: {e}")
        traceback.print_exc()

    finally:
        # Ensure the environment is closed properly
        if env is not None:
            env.close()
            logging.info("Environment closed.")

if __name__ == "__main__":
    main()
