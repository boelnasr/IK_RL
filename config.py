#!/usr/bin/env python3
import logging

# Configure logging
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Configuration dictionary
config = {
    'robot_name': 'kuka_iiwa',             # Robot model to be used
    'hidden_dim': 1024,                     # Hidden layer dimension for the actor-critic network
    'lr': 1e-4,                            # Global default learning rate
    'gamma': 0.98,                         # Discount factor for rewards
    'tau': 0.92,                           # GAE parameter for advantage estimation
    'clip_param': 0.2,                     # Global default PPO clip parameter
    'ppo_epochs': 200,                      # Number of PPO epochs per update
    'batch_size': 1024,                      # Batch size for training
    'buffer_size': 4096,                   # Size of the replay buffer

    'num_episodes': 1000,                  # Number of episodes to train
    'max_steps_per_episode': 200,          # Maximum number of steps per episode

    'test_agent_after_training': True,     # Whether to test the agent after training
    'num_tests': 5,                        # Number of test episodes to run after training

    # Per-joint learning rates (optional, fall back to global 'lr' if not provided)
    'lr_joint_0': 1e-4,                    # Learning rate for joint 0
    'lr_joint_1': 3e-4,                    # Learning rate for joint 1
    'lr_joint_2': 5e-4,                    # Learning rate for joint 2
    'lr_joint_3': 3e-4,                    # Learning rate for joint 3
    'lr_joint_4': 1e-4,                    # Learning rate for joint 4
    'lr_joint_5': 3e-4,                    # Learning rate for joint 5
    'lr_joint_6': 5e-4,                    # Learning rate for joint 6

    # Per-joint PPO clip parameters (optional, fall back to global 'clip_param' if not provided)
    'clip_joint_0': 0.1,                   # PPO clip parameter for joint 0
    'clip_joint_1': 0.2,                   # PPO clip parameter for joint 1
    'clip_joint_2': 0.3,                   # PPO clip parameter for joint 2
    'clip_joint_3': 0.2,                   # PPO clip parameter for joint 3
    'clip_joint_4': 0.1,                   # PPO clip parameter for joint 4
    'clip_joint_5': 0.2,                   # PPO clip parameter for joint 5
    'clip_joint_6': 0.3                    # PPO clip parameter for joint 6
}
