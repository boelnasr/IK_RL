#!/usr/bin/env python3

import os
import numpy as np
import logging
import traceback
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ik_solver.environment import InverseKinematicsEnv
from ik_solver.mappo import MAPPOAgent
from config import config

def setup_distributed(rank, world_size):
    """
    Set up the process group for distributed training.
    """
    os.environ['MASTER_ADDR'] = 'localhost'  # Set the master node address
    os.environ['MASTER_PORT'] = '12355'     # Set the master node port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    """
    Clean up the distributed process group.
    """
    dist.destroy_process_group()

def train_agent(rank, world_size, num_episodes, max_steps_per_episode, test_agent_after_training, num_tests):
    """
    Train the MAPPO agent using distributed training.
    """
    
    # Set up distributed environment
    setup_distributed(rank, world_size)

    # Configure logging
    logging.basicConfig(
        filename=f'training_rank_{rank}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    env = None  # Ensure `env` is defined for the `finally` block
    try:
        # Initialize the environment
        robot_name = config.get('robot_name', 'xarm')
        logging.info(f"[Rank {rank}] Initializing environment with robot: {robot_name}")
        env = InverseKinematicsEnv(robot_name=robot_name)
        logging.info(f"[Rank {rank}] Environment initialized successfully.")

        # Initialize the MAPPO agent
        agent = MAPPOAgent(env, config)
        logging.info(f"[Rank {rank}] MAPPOAgent initialized successfully.")

        # Assign agent to specific device for distributed training
        device = torch.device(f"cuda:{rank}")
        agent.device = device
        agent.agents = [a.to(device) for a in agent.agents]
        agent.critic.to(device)

        # Start training
        logging.info(f"[Rank {rank}] Starting training...")
        agent.train()
        logging.info(f"[Rank {rank}] Training completed successfully.")

        # Test the agent after training if specified
        if rank == 0 and test_agent_after_training:  # Only rank 0 performs testing
            logging.info("[Rank 0] Starting testing...")
            agent.test_agent(env, num_episodes=num_tests)
            logging.info("[Rank 0] Testing completed successfully.")

    except Exception as e:
        # Log and print the error stack trace
        logging.error(f"[Rank {rank}] An error occurred: {e}")
        traceback.print_exc()

    finally:
        # Ensure the environment is closed properly
        if env is not None:
            env.close()
            logging.info(f"[Rank {rank}] Environment closed.")

        # Clean up distributed training setup
        cleanup_distributed()

def main():
    """
    Main function to initialize distributed training or fallback to single GPU.
    """
    world_size = torch.cuda.device_count()  # Detect available GPUs
    if world_size < 2:
        logging.info("Single GPU detected. Running training on a single GPU.")
        
        # Retrieve configuration parameters
        num_episodes = config.get('num_episodes', 2000)
        max_steps_per_episode = config.get('max_steps_per_episode', 5000)
        test_agent_after_training = config.get('test_agent_after_training', True)
        num_tests = config.get('num_tests', 5)
        
        # Run single-GPU training
        train_agent(
            rank=0,  # Single process
            world_size=1,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            test_agent_after_training=test_agent_after_training,
            num_tests=num_tests
        )
    else:
        # Start the distributed training processes
        mp.spawn(
            train_agent,
            args=(world_size, num_episodes, max_steps_per_episode, test_agent_after_training, num_tests),
            nprocs=world_size,
            join=True
        )



if __name__ == "__main__":
    main()
