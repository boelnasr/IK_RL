#!/usr/bin/env python3
import logging

# Configure logging
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Configuration dictionary
config = {
    'robot_name': 'ur5',#'kuka_iiwa',             # Robot model to be used
    'hidden_dim': 256,                     # Hidden layer dimension for the actor-critic network
    'lr': 1e-4,                            # Global default learning rate
    'gamma': 0.99,                         # Discount factor for rewards
    'tau': 0.95,                           # GAE parameter for advantage estimation
    'clip_param': 0.2,                     # Global default PPO clip parameter
    'ppo_epochs': 15,                      # Number of PPO epochs per update
    'batch_size': 128,                      # Batch size for training
    'buffer_size': 4096,                   # Size of the replay buffer
    'initial_epsilon': 1.0,
    'epsilon_decay': 0.999,
    'min_epsilon':0.01,
    'num_episodes': 1000,                  # Number of episodes to train
    'max_steps_per_episode':5000,          # Maximum number of steps per episode

    'test_agent_after_training': True,     #  Whether to test the agent after training
    'num_tests': 15,                        # Number of test episodes to run after training

    # Per-joint learning rates (optional, fall back to global 'lr' if not provided)
    'lr_joint_0': 1e-4,                    # Learning rate for joint 0
    'lr_joint_1': 1e-4,                    # Learning rate for joint 1
    'lr_joint_2': 1e-4,                    # Learning rate for joint 2
    'lr_joint_3': 1e-4,                    # Learning rate for joint 3
    'lr_joint_4': 1e-4,                    # Learning rate for joint 4
    'lr_joint_5': 1e-4,                    # Learning rate for joint 5
    'lr_joint_6': 1e-4,                    # Learning rate for joint 6

    # Per-joint PPO clip parameters (optional, fall back to global 'clip_param' if not provided)
    'clip_joint_0': 0.2,                   # PPO clip parameter for joint 0
    'clip_joint_1': 0.2,                   # PPO clip parameter for joint 1
    'clip_joint_2': 0.2,                   # PPO clip parameter for joint 2
    'clip_joint_3': 0.2,                   # PPO clip parameter for joint 3
    'clip_joint_4': 0.2,                   # PPO clip parameter for joint 4
    'clip_joint_5': 0.2,                   # PPO clip parameter for joint 5
    'clip_joint_6': 0.2                    # PPO clip parameter for joint 6
}
attention_config = {
    'num_heads': 4,               # Number of attention heads
    'head_dim': 64,              # Dimension of each attention head
    'attention_dropout': 0.1,     # Dropout rate for attention
    'key_dim': 64,               # Key dimension
    'value_dim': 64,             # Value dimension
    'query_dim': 64,             # Query dimension
    'output_dim': 256,           # Output dimension after attention
    'use_bias': True             # Whether to use bias in projections
}
encoder_config = {
    'input_dim': 7,        # 1 joint angle + 3 position error + 3 orientation error
    'hidden_dim': 128,     # Hidden layer dimension
    'embedding_dim': 64,   # Embedding dimension
    'num_layers': 2,       # Number of encoder layers
    'dropout': 0.1,        # Dropout rate
    'use_layer_norm': True # Whether to use layer normalization
}
