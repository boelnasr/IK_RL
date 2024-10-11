# IK_RL
# Multi-Agent Proximal Policy Optimization (MAPPO) for Inverse Kinematics

## Project Overview

This project implements a Multi-Agent Proximal Policy Optimization (MAPPO) algorithm to solve inverse kinematics problems for robotic arms. The system uses PyBullet for physics simulation and PyTorch for deep learning.

## Key Components

### 1. Environment (`InverseKinematicsEnv`)

- Custom OpenAI Gym environment
- Simulates a robotic arm (default: KUKA IIWA) in PyBullet
- Handles state observations, action applications, and reward calculations

### 2. MAPPO Agent (`MAPPOAgent`)

- Implements the MAPPO algorithm
- Manages multiple agents, one for each joint of the robotic arm
- Uses a centralized critic and decentralized actors

### 3. Neural Network Models

#### Actor Network (`JointActor`)
- Predicts action mean and standard deviation for each joint
- Uses tanh activation for bounded actions

#### Critic Network (`CentralizedCritic`)
- Estimates the value function for the entire state

### 4. Training Process

- Episodic training loop
- Collects trajectories and updates policy using PPO
- Implements Generalized Advantage Estimation (GAE)
- Uses separate learning rates and clip parameters for each joint

### 5. Metrics and Logging (`TrainingMetrics`)

- Tracks various performance metrics during training
- Generates plots and saves logs for analysis

## Key Features

- Multi-agent approach for controlling individual joints
- Centralized training with decentralized execution
- Dynamic difficulty adjustment during training
- Best model saving based on joint error performance
- Comprehensive logging and visualization of training metrics

## Usage

1. Configure the environment and training parameters in `config.py`
2. Run the main training script: main.py
3. Monitor training progress through logged metrics and generated plots
4. Use the trained model for testing or deployment

## Dependencies

- PyTorch
- PyBullet
- OpenAI Gym
- NumPy
- Matplotlib
## Future Improvements

- Implement more advanced exploration strategies
- Add support for different robot models
- Optimize hyperparameters for better performance
- Implement multi-task learning for various IK problems