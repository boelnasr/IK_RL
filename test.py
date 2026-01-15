#!/usr/bin/env python3

import numpy as np
import pybullet as p
import logging
import time
from collections import defaultdict
from pathlib import Path
import json
from datetime import datetime
from ik_solver.environment import InverseKinematicsEnv
from ik_solver.mappo import MAPPOAgent

class AgentTester:
    """
    A comprehensive testing framework for evaluating trained MAPPO agents in robotic control.
    Includes systematic evaluation of precision, robustness, speed, and basic performance metrics.
    """

    def __init__(
        self,
        agent,
        env,
        base_path="test_results",
        num_joints=6,
        convergence_patience=5,
        position_threshold=None,
        orientation_threshold=None,
    ):
        """
        Initialize the testing framework with the trained agent and environment.
        """
        self.agent = agent
        self.env = env
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.num_joints = num_joints
        self.metrics = defaultdict(list)
        self.setup_logging()

        # Session directory for organizing results
        self.session_path = self.base_path / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path.mkdir(exist_ok=True)

        # Early convergence configuration
        self.convergence_patience = convergence_patience
        env_position_threshold = getattr(self.env, "position_threshold", None)
        env_orientation_threshold = getattr(self.env, "orientation_threshold", None)
        env_success_threshold = getattr(self.env, "success_threshold", None)

        default_threshold = env_success_threshold if env_success_threshold is not None else 0.02
        self.position_threshold = (
            position_threshold
            if position_threshold is not None
            else env_position_threshold
            if env_position_threshold is not None
            else default_threshold
        )
        self.orientation_threshold = (
            orientation_threshold
            if orientation_threshold is not None
            else env_orientation_threshold
            if env_orientation_threshold is not None
            else default_threshold
        )

    def setup_logging(self):
        """Configure logging for the testing framework."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(self.base_path / "testing.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        self.logger.addHandler(console_handler)

    def run_systematic_tests(self, num_episodes=100):
        """
        Run comprehensive tests on the agent and environment.
        Includes evaluations for precision, robustness, speed, and basic performance.
        """
        try:
            self.logger.info("Starting systematic tests...")

            # Run test categories
            test_results = {
                "basic_performance": self.test_basic_performance(num_episodes),
                "precision_tests": self.test_precision(num_episodes),
                "robustness_tests": self.test_robustness(num_episodes),
                "speed_tests": self.test_speed(num_episodes),
            }

            # Save and process results
            self.save_results(test_results)
            return test_results

        except Exception as e:
            self.logger.error(f"Error during testing: {e}")
            raise

    def test_basic_performance(self, num_episodes):
        """Evaluate the agent's basic task completion performance."""
        results = {
            "success_rate": [],
            "completion_times": [],
            "final_distances": [],
            "trajectory_smoothness": [],
            "joint_errors": {f"joint_{i}": [] for i in range(self.num_joints)},
        }

        for episode in range(num_episodes):
            state, _ = self.env.reset()  # Gymnasium API returns (obs, info)
            done = False
            steps = 0
            joint_errors = defaultdict(list)
            convergence_counter = 0

            while not done and steps < self.env.max_episode_steps:
                actions, _ = self.agent.get_actions(state, eval_mode=True)
                next_state, _, terminated, truncated, info = self.env.step(actions)  # Gymnasium API
                done = terminated or truncated
                state = next_state

                # Record joint errors
                for i in range(self.num_joints):
                    joint_state = p.getJointState(self.env.robot_id, self.env.joint_indices[i])
                    joint_errors[f"joint_{i}"].append(abs(joint_state[0] - info["target_joint_angles"][i]))

                convergence_counter = self._update_convergence_counter(info, convergence_counter)
                if convergence_counter >= self.convergence_patience:
                    done = True
                    self.logger.info(
                        "Basic performance test episode %d: early stopping after convergence at step %d",
                        episode + 1,
                        steps + 1,
                    )

                steps += 1

            # Record results
            results["success_rate"].append(info["success"])
            results["completion_times"].append(steps)
            results["final_distances"].append(info["current_distance"])
            for joint, errors in joint_errors.items():
                results["joint_errors"][joint].append(np.mean(errors))

            self.logger.info(f"Completed basic performance test for episode {episode + 1}/{num_episodes}")

        return results

    def test_precision(self, num_episodes):
        """Evaluate the agent's ability to achieve precise positioning."""
        thresholds = [0.1, 0.05, 0.02, 0.01, 0.005]
        results = {"success_rates": {f"threshold_{t}": [] for t in thresholds}}

        for episode in range(num_episodes):
            state, _ = self.env.reset()  # Gymnasium API returns (obs, info)
            done = False
            convergence_counter = 0

            while not done:
                actions, _ = self.agent.get_actions(state, eval_mode=True)
                next_state, _, terminated, truncated, info = self.env.step(actions)  # Gymnasium API
                done = terminated or truncated
                state = next_state

                convergence_counter = self._update_convergence_counter(info, convergence_counter)
                if convergence_counter >= self.convergence_patience:
                    done = True
                    self.logger.info(
                        "Precision test episode %d: early stopping after convergence",
                        episode + 1,
                    )

            for threshold in thresholds:
                results["success_rates"][f"threshold_{threshold}"].append(info["current_distance"] <= threshold)

            self.logger.info(f"Completed precision test for episode {episode + 1}/{num_episodes}")

        return results

    def test_robustness(self, num_episodes):
        """Evaluate the agent's performance under different configurations."""
        configs = {
            "stretched": {"angles": [0.0] * self.num_joints},
            "folded": {"angles": [np.pi / 2] * self.num_joints},
            "random": {"angles": None},
        }
        results = {name: {"success_rate": [], "position_error": [], "orientation_error": []} for name in configs}

        for config_name, config in configs.items():
            for episode in range(num_episodes):
                _, _ = self.env.reset()  # Gymnasium API returns (obs, info)
                if config["angles"] is not None:
                    for i, angle in enumerate(config["angles"]):
                        p.resetJointState(self.env.robot_id, self.env.joint_indices[i], angle)

                done = False
                convergence_counter = 0
                while not done:
                    actions, _ = self.agent.get_actions(self.env.get_observation(), eval_mode=True)
                    _, _, terminated, truncated, info = self.env.step(actions)  # Gymnasium API
                    done = terminated or truncated

                    convergence_counter = self._update_convergence_counter(info, convergence_counter)
                    if convergence_counter >= self.convergence_patience:
                        done = True
                        self.logger.info(
                            "Robustness test '%s' episode %d: early stopping after convergence",
                            config_name,
                            episode + 1,
                        )

                results[config_name]["success_rate"].append(info["success"])
                results[config_name]["position_error"].append(info["current_distance"])
                results[config_name]["orientation_error"].append(info["orientation_error"])

            self.logger.info(f"Completed robustness test for config '{config_name}'")

        return results

    def test_speed(self, num_episodes):
        """Evaluate the agent's speed and efficiency."""
        results = {"reaching_time": [], "path_efficiency": []}

        for episode in range(num_episodes):
            start_time = time.time()
            _, _ = self.env.reset()  # Gymnasium API returns (obs, info)
            done = False
            convergence_counter = 0

            while not done:
                actions, _ = self.agent.get_actions(self.env.get_observation(), eval_mode=True)
                _, _, terminated, truncated, info = self.env.step(actions)  # Gymnasium API
                done = terminated or truncated

                convergence_counter = self._update_convergence_counter(info, convergence_counter)
                if convergence_counter >= self.convergence_patience:
                    done = True
                    self.logger.info(
                        "Speed test episode %d: early stopping after convergence",
                        episode + 1,
                    )

            reaching_time = time.time() - start_time
            results["reaching_time"].append(reaching_time)

            # Example metric: efficiency (straight-line vs actual path length)
            results["path_efficiency"].append(self.env.compute_path_efficiency())

            self.logger.info(f"Completed speed test for episode {episode + 1}/{num_episodes}")

        return results

    def save_results(self, results):
        """Save test results to disk."""
        results_path = self.session_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        self.logger.info(f"Results saved to {results_path}")

    def _update_convergence_counter(self, info, counter):
        """Update and return the convergence counter based on the latest env info."""
        if info is None:
            return 0

        return counter + 1 if self._has_converged(info) else 0

    def _has_converged(self, info):
        """Return True when current metrics indicate the agent has converged."""
        if not isinstance(info, dict):
            return False

        if info.get("success"):
            return True

        success_ratio = info.get("overall_success_rate")
        if success_ratio is not None and success_ratio >= 1.0:
            return True

        evaluated = False
        metrics_met = True

        success_threshold = getattr(self.env, "success_threshold", None)
        distance = info.get("current_distance")
        if distance is not None and success_threshold is not None:
            evaluated = True
            metrics_met = metrics_met and distance <= success_threshold

        position_error = info.get("position_error")
        if position_error is not None and self.position_threshold is not None:
            evaluated = True
            metrics_met = metrics_met and position_error <= self.position_threshold

        orientation_error = info.get("orientation_error")
        if orientation_error is not None and self.orientation_threshold is not None:
            evaluated = True
            metrics_met = metrics_met and orientation_error <= self.orientation_threshold

        return metrics_met if evaluated else False

    def close(self):
        """Clean up resources."""
        self.env.close()
        self.logger.info("Environment closed successfully")


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logger = logging.getLogger(__name__)

        # Initialize environment and agent
        logger.info("Initializing environment...")
        env = InverseKinematicsEnv(robot_name="xarm", sim_timestep=1 / 240, max_episode_steps=1000)

        logger.info("Initializing agent...")
        agent = MAPPOAgent(env)

        # Set up tester
        tester = AgentTester(agent, env)

        # Run tests
        logger.info("Starting tests...")
        tester.run_systematic_tests(num_episodes=100)

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise

    finally:
        tester.close()
