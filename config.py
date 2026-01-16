#!/usr/bin/env python3
import logging

# Configure logging
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Configuration dictionary
config = {
    'robot_name': 'xarm',             # Robot model to be used
    'hidden_dim': 256,                     # OPTIMIZED: Reduced from 512 for faster training
    'lr': 1e-4,                            # FIXED: Further reduced from 2e-4 for more stable policy updates
    'gamma': 0.99,                         # Discount factor for rewards
    'tau': 0.95,                           # GAE parameter for advantage estimation
    'clip_param': 0.15,                    # Global default PPO clip parameter (reduced from 0.2)
    'ppo_epochs': 5,                       # IMPROVED: Increased from 3 for better learning
    'batch_size': 256,                     # IMPROVED: Larger batches for more stable gradients
    'buffer_size': 4096,                   # Size of the replay buffer
    'initial_epsilon': 0.50,               # IMPROVED: Increased from 0.40 for better exploration
    'epsilon_decay': 0.998,                # IMPROVED: Slower decay for longer exploration
    'min_epsilon': 0.15,                   # IMPROVED: Increased from 0.10 to maintain exploration
    'num_episodes': 500,                   # IMPROVED: Increased from 100 - agents need more time to learn!
    'max_steps_per_episode': 1000,         # Steps per episode for training
    'enable_anomaly_detection': False,     # Enable PyTorch anomaly detection during debugging
    'jacobian_update_tolerance': 1e-3,     # Angle change threshold before recomputing Jacobians

    'test_agent_after_training': True,     #  Whether to test the agent after training
    'num_tests': 10,                        # Number of test episodes to run after training
    'use_cross_validation' : False,
    # Per-joint learning rates (optional, fall back to global 'lr' if not provided)
    'lr_joint_0': 1e-4,                    # FIXED: Reduced from 2e-4 for stability
    'lr_joint_1': 1e-4,                    # FIXED: Reduced from 2e-4 for stability
    'lr_joint_2': 1e-4,                    # FIXED: Reduced from 2e-4 for stability
    'lr_joint_3': 1e-4,                    # FIXED: Reduced from 2e-4 for stability
    'lr_joint_4': 1e-4,                    # FIXED: Reduced from 2e-4 for stability
    'lr_joint_5': 1e-4,                    # FIXED: Reduced from 2e-4 for stability
    'lr_joint_6': 1e-4,                    # FIXED: Reduced from 2e-4 for stability

    # Per-joint PPO clip parameters (optional, fall back to global 'clip_param' if not provided)
    'clip_joint_0': 0.15,                  # PPO clip parameter for joint 0
    'clip_joint_1': 0.15,                  # PPO clip parameter for joint 1
    'clip_joint_2': 0.15,                  # PPO clip parameter for joint 2
    'clip_joint_3': 0.15,                  # PPO clip parameter for joint 3
    'clip_joint_4': 0.15,                  # PPO clip parameter for joint 4
    'clip_joint_5': 0.15,                  # PPO clip parameter for joint 5
    'clip_joint_6': 0.15,                   # PPO clip parameter for joint 6
    # Your existing config parameters
    'value_loss_scale': 1.5,     # FIXED: Increased from 0.8 to prevent critic collapse!
    'entropy_scale': 0.05,       # FIXED: Increased from 0.01 to maintain exploration!
    'max_grad_norm': 0.5,        # Maximum gradient norm
    'ratio_clip': 0.15,          # Maximum policy ratio (tightened from 0.20)
    'advantage_clip': 10.0,      # FIXED: Increased from 1.0 - was clipping too aggressively!
    'reward_scale': 1.0,         # FIXED: Increased from 0.1 - don't scale down rewards!
    'value_loss_clip': 10.0,     # NEW: Clip value loss to prevent collapse
    'normalize_advantages': True,  # NEW: Normalize advantages for stability
    'use_scheduler': True,     # Whether to use a learning rate scheduler
    #GPU config
    'num_envs': 4,              # Number of parallel environments
    'world_size': 1,            # Number of GPUs (1 for single GPU)
    'rank': 0,                  # GPU rank (0 for single GPU)
    #Cross validation config
    'validation_episodes': 10,
    'k_folds': 3,
    # HER parameters - FULLY DISABLED for simpler, more stable learning
    'use_her': False,                      # DISABLED: HER buffer turned off
    'use_prioritized_replay': False,       # DISABLED: Prioritized replay buffer turned off
    'buffer_update_freq': 10,              # REDUCED frequency (was 3)
    'num_buffer_updates': 1,               # Number of mini-batch updates per replay trigger
    'her_update_freq': 20,                 # REDUCED frequency (was 8)
    'her_batch_size': 256,                 # Increased from 128 for better HER utilization
    'her_k_future': 4,                     # REDUCED: Back to 4 from 6
    'her_reward_type': 'dense',            # 'dense' or 'sparse'
    'her_success_threshold': 0.05,

    # SUCCESS THRESHOLD CONFIGURATION (curriculum learning)
    # CRITICAL: Start with achievable threshold, then tighten as agent improves
    'max_success_threshold': 0.1,     # 100mm - START HERE (achievable with 40-80mm errors)
    'min_success_threshold': 0.005,   # 5mm - TARGET (tighten as agent succeeds)

    # PD Controller - DISABLED for pure RL learning
    'use_pd_controller': False,            # DISABLED: Turn off PD controller
    'pd_weight': 0.0,                      # Set to 0 to disable PD blending
    'pd_kp': 1.0,
    'pd_kd': 0.2,
    'pd_dt': 0.01,
    # Best-model selection tuning
    'best_model_success_weight': 1.0,      # Weight for success-rate component
    'best_model_error_weight': 0.5,        # Weight for mean-joint-error penalty
    'best_model_reward_weight': 0.0,       # Optional reward contribution (kept neutral)
    'best_model_min_delta': 1e-3,          # Minimum score improvement to overwrite best model
    # Early stopping
    'early_stop_enabled': False,
    'early_stop_metric': 'success_rate',  # 'success_rate' or 'mean_joint_error'
    'early_stop_patience': 40,
    'early_stop_min_progress': 0.8,  # Start checking once this fraction of training is done
    'early_stop_tolerance': 1e-3,
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
