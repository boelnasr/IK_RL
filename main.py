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
from ik_solver.reward_function import get_reward_parameters_snapshot
from ik_solver.reporting import write_reward_report
from config import config, attention_config, encoder_config
import random

def setup_distributed(rank, world_size):
    """
    Set up the process group for distributed training.
    """
    os.environ['MASTER_ADDR'] = 'localhost'  # Set the master node address
    # Use a random port to avoid conflicts
    os.environ['MASTER_PORT'] = str(random.randint(12000, 65000))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    """
    Clean up the distributed process group.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def _build_config_snapshot():
    """
    Collect a copy of configuration data for reporting.
    """
    return {
        "config": dict(config),
        "attention_config": dict(attention_config),
        "encoder_config": dict(encoder_config),
    }

def train_agent_single_gpu(num_episodes, max_steps_per_episode, test_agent_after_training, num_tests):
    """
    Train the MAPPO agent on single GPU without distributed training.
    """
    # Configure logging for single GPU
    logging.basicConfig(
        filename='training_single_gpu.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    env = None  # Ensure `env` is defined for the `finally` block
    try:
        # Initialize the environment
        robot_name = config.get('robot_name', 'xarm')
        logging.info(f"Initializing environment with robot: {robot_name}")
        env = InverseKinematicsEnv(robot_name=robot_name)
        logging.info("Environment initialized successfully.")

        # Initialize the MAPPO agent
        agent = MAPPOAgent(env, config)
        logging.info("MAPPOAgent initialized successfully.")

        # Assign agent to GPU 0 for single GPU training
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        agent.device = device
        agent.agents = [a.to(device) for a in agent.agents]
        agent.critic.to(device)
        
        logging.info(f"Using device: {device}")

        report_dir = os.path.join(agent.base_path, "report_logs")
        report_path = write_reward_report(
            config_snapshot=_build_config_snapshot(),
            reward_snapshot=get_reward_parameters_snapshot(),
            env_snapshot=env.get_reward_logging_snapshot(),
            output_dir=report_dir,
            run_label=f"single_gpu::{os.path.basename(agent.base_path)}",
        )
        logging.info(f"Reward configuration report saved to {report_path}")

        # Start training
        logging.info("Starting training...")
        agent.train()
        logging.info("Training completed successfully.")

        # Test the agent after training if specified
        if test_agent_after_training:
            logging.info("Starting testing...")
            agent.test_agent(env, num_episodes=num_tests)
            logging.info("Testing completed successfully.")

    except Exception as e:
        # Log and print the error stack trace
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()

    finally:
        # Ensure the environment is closed properly
        if env is not None:
            env.close()
            logging.info("Environment closed.")

def train_agent_distributed(rank, world_size, num_episodes, max_steps_per_episode, test_agent_after_training, num_tests):
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

        if rank == 0:
            report_dir = os.path.join(agent.base_path, "report_logs")
            report_path = write_reward_report(
                config_snapshot=_build_config_snapshot(),
                reward_snapshot=get_reward_parameters_snapshot(),
                env_snapshot=env.get_reward_logging_snapshot(),
                output_dir=report_dir,
                run_label=f"distributed_rank_{rank}::{os.path.basename(agent.base_path)}",
            )
            logging.info(f"[Rank {rank}] Reward configuration report saved to {report_path}")

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
    
    # Retrieve configuration parameters
    num_episodes = config.get('num_episodes', 100)  # Reduced for testing
    max_steps_per_episode = config.get('max_steps_per_episode', 500)  # Reduced for testing
    test_agent_after_training = config.get('test_agent_after_training', True)
    num_tests = config.get('num_tests', 5)
    
    print(f"Detected {world_size} GPU(s)")
    
    if world_size < 2:
        print("Running single GPU training...")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("Single GPU detected. Running training on a single GPU.")
        
        # Run single-GPU training WITHOUT distributed setup
        train_agent_single_gpu(
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            test_agent_after_training=test_agent_after_training,
            num_tests=num_tests
        )
    else:
        print("Running distributed training...")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info(f"Multiple GPUs detected ({world_size}). Running distributed training.")
        
        # Start the distributed training processes
        mp.spawn(
            train_agent_distributed,
            args=(world_size, num_episodes, max_steps_per_episode, test_agent_after_training, num_tests),
            nprocs=world_size,
            join=True
        )

if __name__ == "__main__":
    main()
