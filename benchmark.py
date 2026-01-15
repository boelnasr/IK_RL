#!/usr/bin/env python3
"""
Benchmark script to estimate training time for IK_RL.

Runs a small number of episodes to measure performance and extrapolates
to estimate total training time.

Usage:
    python benchmark.py [--episodes N] [--steps N]
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import platform
from datetime import timedelta

# Suppress PyBullet GUI
os.environ['PYBULLET_SILENT'] = '1'


def get_system_info():
    """Gather system information."""
    info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'python_version': platform.python_version(),
        'processor': platform.processor(),
        'torch_version': torch.__version__,
    }

    # GPU info
    if torch.cuda.is_available():
        info['gpu_available'] = True
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        info['cuda_version'] = torch.version.cuda
    else:
        info['gpu_available'] = False
        info['gpu_name'] = 'N/A (CPU only)'
        info['gpu_memory'] = 'N/A'
        info['cuda_version'] = 'N/A'

    # CPU info
    try:
        import multiprocessing
        info['cpu_cores'] = multiprocessing.cpu_count()
    except:
        info['cpu_cores'] = 'Unknown'

    return info


def run_benchmark(num_episodes=5, max_steps=100):
    """Run benchmark episodes and measure timing."""

    print("\n" + "="*60)
    print("IK_RL Training Benchmark")
    print("="*60)

    # System info
    print("\n[System Information]")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"  {key}: {value}")

    # Import after system info (to catch import errors)
    print("\n[Loading modules...]")
    load_start = time.time()

    try:
        from config import config
        from ik_solver.environment import InverseKinematicsEnv
        from ik_solver.mappo import MAPPOAgent
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure you're running from the IK_RL directory.")
        sys.exit(1)

    load_time = time.time() - load_start
    print(f"  Module load time: {load_time:.2f}s")

    # Override config for benchmark
    benchmark_config = config.copy()
    benchmark_config['num_episodes'] = num_episodes
    benchmark_config['max_steps_per_episode'] = max_steps

    print(f"\n[Benchmark Configuration]")
    print(f"  Robot: {benchmark_config['robot_name']}")
    print(f"  Benchmark episodes: {num_episodes}")
    print(f"  Steps per episode: {max_steps}")
    print(f"  Target episodes: {config['num_episodes']}")

    # Initialize environment
    print("\n[Initializing environment...]")
    env_start = time.time()

    try:
        env = InverseKinematicsEnv(
            robot_name=benchmark_config['robot_name'],
            config=benchmark_config
        )
    except Exception as e:
        print(f"Error initializing environment: {e}")
        sys.exit(1)

    env_init_time = time.time() - env_start
    print(f"  Environment init time: {env_init_time:.2f}s")

    # Initialize agent
    print("\n[Initializing agent...]")
    agent_start = time.time()

    try:
        agent = MAPPOAgent(env, benchmark_config)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        env.close()
        sys.exit(1)

    agent_init_time = time.time() - agent_start
    print(f"  Agent init time: {agent_init_time:.2f}s")

    # Run benchmark episodes
    print(f"\n[Running {num_episodes} benchmark episodes...]")

    episode_times = []
    step_times = []

    for ep in range(num_episodes):
        ep_start = time.time()

        state, _ = env.reset()  # Gymnasium API returns (obs, info)
        done = False
        step_count = 0
        ep_step_times = []

        while not done and step_count < max_steps:
            step_start = time.time()

            actions, log_probs = agent.get_actions(state)
            next_state, rewards, terminated, truncated, info = env.step(actions)  # Gymnasium API
            done = terminated or truncated
            state = next_state
            step_count += 1

            ep_step_times.append(time.time() - step_start)

        ep_time = time.time() - ep_start
        episode_times.append(ep_time)
        step_times.extend(ep_step_times)

        print(f"  Episode {ep+1}/{num_episodes}: {ep_time:.2f}s ({step_count} steps)")

    # Cleanup
    env.close()

    # Calculate statistics
    print("\n" + "="*60)
    print("Benchmark Results")
    print("="*60)

    avg_episode_time = np.mean(episode_times)
    std_episode_time = np.std(episode_times)
    avg_step_time = np.mean(step_times) * 1000  # ms

    print(f"\n[Timing Statistics]")
    print(f"  Average episode time: {avg_episode_time:.2f}s (±{std_episode_time:.2f}s)")
    print(f"  Average step time: {avg_step_time:.2f}ms")
    print(f"  Steps per second: {1000/avg_step_time:.1f}")

    # Estimate total training time
    target_episodes = config['num_episodes']

    # Account for policy updates (roughly 20% overhead)
    policy_update_overhead = 1.2

    estimated_training_time = avg_episode_time * target_episodes * policy_update_overhead
    estimated_time_delta = timedelta(seconds=int(estimated_training_time))

    # Best/worst case estimates
    best_case = (avg_episode_time - std_episode_time) * target_episodes * policy_update_overhead
    worst_case = (avg_episode_time + std_episode_time) * target_episodes * policy_update_overhead

    print(f"\n[Training Time Estimates for {target_episodes} episodes]")
    print(f"  Estimated time: {estimated_time_delta} ({estimated_training_time/60:.1f} min)")
    print(f"  Best case:      {timedelta(seconds=int(best_case))} ({best_case/60:.1f} min)")
    print(f"  Worst case:     {timedelta(seconds=int(worst_case))} ({worst_case/60:.1f} min)")

    # Estimates for different episode counts
    print(f"\n[Estimates for Different Episode Counts]")
    print(f"  {'Episodes':<12} {'Est. Time':<15} {'Minutes':<10}")
    print(f"  {'-'*12} {'-'*15} {'-'*10}")

    for ep_count in [100, 200, 300, 500, 1000]:
        est_time = avg_episode_time * ep_count * policy_update_overhead
        print(f"  {ep_count:<12} {str(timedelta(seconds=int(est_time))):<15} {est_time/60:.1f}")

    # Memory usage
    print(f"\n[Memory Usage]")
    if torch.cuda.is_available():
        gpu_mem_used = torch.cuda.memory_allocated() / 1e9
        gpu_mem_cached = torch.cuda.memory_reserved() / 1e9
        print(f"  GPU memory allocated: {gpu_mem_used:.2f} GB")
        print(f"  GPU memory cached: {gpu_mem_cached:.2f} GB")

    try:
        import psutil
        process = psutil.Process()
        ram_used = process.memory_info().rss / 1e9
        print(f"  RAM used: {ram_used:.2f} GB")
    except ImportError:
        print("  (Install psutil for RAM usage: pip install psutil)")

    # Recommendations
    print(f"\n[Recommendations]")
    if not sys_info['gpu_available']:
        print("  ⚠ No GPU detected. Training will be slower on CPU.")
        print("    Consider using Google Colab with GPU runtime.")

    if avg_step_time > 50:  # More than 50ms per step
        print("  ⚠ Step time is slow. Consider:")
        print("    - Reducing max_steps_per_episode")
        print("    - Using a simpler robot model")

    if estimated_training_time > 3600:  # More than 1 hour
        print(f"  ℹ Training will take >{estimated_training_time/3600:.1f} hours.")
        print("    Consider reducing num_episodes for initial testing.")

    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60 + "\n")

    return {
        'avg_episode_time': avg_episode_time,
        'avg_step_time': avg_step_time,
        'estimated_total_time': estimated_training_time,
        'system_info': sys_info
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark IK_RL training time')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of benchmark episodes (default: 5)')
    parser.add_argument('--steps', type=int, default=100,
                        help='Max steps per episode (default: 100)')

    args = parser.parse_args()

    results = run_benchmark(num_episodes=args.episodes, max_steps=args.steps)
