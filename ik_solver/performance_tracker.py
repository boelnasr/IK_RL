"""
Performance-based checkpoint and early stopping tracker.

This module provides robust tracking of training performance using rolling windows
and curriculum-aware early stopping criteria.
"""

import numpy as np
from collections import deque
import logging
import torch
import os
from typing import Dict, Optional, Tuple


class PerformanceTracker:
    """
    Tracks performance metrics with rolling windows for robust early stopping.

    Key features:
    - Rolling window averaging to reduce noise
    - Curriculum-aware: won't stop until curriculum is complete
    - Tracks success rate AND pose error separately
    - Checkpoints best models automatically
    """

    def __init__(
        self,
        window_size: int = 50,
        patience: int = 100,
        min_delta_success: float = 0.01,  # 1% improvement required
        min_delta_error: float = 0.001,    # 0.001 rad improvement required
        save_dir: str = None,
        num_agents: int = 7
    ):
        """
        Initialize the performance tracker.

        Args:
            window_size: Number of episodes for rolling average
            patience: Episodes to wait after last improvement before stopping
            min_delta_success: Minimum improvement in success rate to count as better
            min_delta_error: Minimum improvement in pose error to count as better
            save_dir: Directory to save checkpoints
            num_agents: Number of agents/joints
        """
        self.window_size = window_size
        self.patience = patience
        self.min_delta_success = min_delta_success
        self.min_delta_error = min_delta_error
        self.save_dir = save_dir
        self.num_agents = num_agents

        # Rolling windows for metrics
        self.success_history = deque(maxlen=window_size)
        self.pose_error_history = deque(maxlen=window_size)
        self.position_error_history = deque(maxlen=window_size)
        self.orientation_error_history = deque(maxlen=window_size)

        # Best performance tracking
        self.best_success_rate = 0.0
        self.best_pose_error = float('inf')
        self.best_checkpoint_episode = 0
        self.episodes_since_improvement = 0

        # Checkpoint paths
        self.best_checkpoint_path = None
        self.last_checkpoint_path = None

        # Statistics
        self.total_episodes = 0
        self.checkpoints_saved = 0

        self.logger = logging.getLogger(__name__)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def update(
        self,
        success: bool,
        position_error: float,
        orientation_error: float,
        episode: int,
        models: Dict[int, torch.nn.Module] = None,
        critic: torch.nn.Module = None,
        curriculum_complete: bool = False
    ) -> Dict[str, any]:
        """
        Update performance tracker with new episode data.

        Args:
            success: Whether episode was successful
            position_error: End-effector position error
            orientation_error: End-effector orientation error
            episode: Current episode number
            models: Dict mapping agent_idx -> actor model (for checkpointing)
            critic: Critic model (for checkpointing)
            curriculum_complete: Whether curriculum ramping is finished

        Returns:
            Dict with tracking info including whether to stop
        """
        self.total_episodes += 1

        # Add to rolling windows
        self.success_history.append(1.0 if success else 0.0)
        self.position_error_history.append(position_error)
        self.orientation_error_history.append(orientation_error)

        # Combined pose error (position + orientation)
        pose_error = position_error + orientation_error
        self.pose_error_history.append(pose_error)

        # Calculate rolling averages (only if we have enough data)
        if len(self.success_history) >= min(10, self.window_size):
            rolling_success = np.mean(self.success_history)
            rolling_pose_error = np.mean(self.pose_error_history)
            rolling_position_error = np.mean(self.position_error_history)
            rolling_orientation_error = np.mean(self.orientation_error_history)
        else:
            # Not enough data yet
            return {
                'should_stop': False,
                'reason': 'insufficient_data',
                'rolling_success_rate': 0.0,
                'rolling_pose_error': float('inf'),
                'episodes_since_improvement': 0,
                'best_success_rate': self.best_success_rate,
                'best_pose_error': self.best_pose_error
            }

        # Check for improvement (either metric improving counts)
        improved = False
        improvement_type = None

        # Success rate improvement
        if rolling_success > (self.best_success_rate + self.min_delta_success):
            self.best_success_rate = rolling_success
            improved = True
            improvement_type = 'success_rate'

        # Pose error improvement (lower is better)
        if rolling_pose_error < (self.best_pose_error - self.min_delta_error):
            self.best_pose_error = rolling_pose_error
            improved = True
            if improvement_type:
                improvement_type = 'both'
            else:
                improvement_type = 'pose_error'

        # Update improvement tracking
        if improved:
            self.episodes_since_improvement = 0
            self.best_checkpoint_episode = episode

            # Save checkpoint if models provided
            if models is not None and self.save_dir:
                checkpoint_path = self._save_checkpoint(
                    episode=episode,
                    models=models,
                    critic=critic,
                    rolling_success=rolling_success,
                    rolling_pose_error=rolling_pose_error,
                    is_best=True
                )
                self.best_checkpoint_path = checkpoint_path
                self.checkpoints_saved += 1

                self.logger.info(
                    f"✓ New best performance at episode {episode} "
                    f"(improved {improvement_type}): "
                    f"success={rolling_success:.3f}, pose_error={rolling_pose_error:.4f}"
                )
        else:
            self.episodes_since_improvement += 1

        # Determine if we should stop
        should_stop = False
        stop_reason = None

        # Only consider stopping if:
        # 1. Curriculum is complete
        # 2. We have enough data in the window
        # 3. No improvement for 'patience' episodes
        if curriculum_complete and len(self.success_history) >= self.window_size:
            if self.episodes_since_improvement >= self.patience:
                should_stop = True
                stop_reason = f'no_improvement_for_{self.patience}_episodes'

                self.logger.info(
                    f"Early stopping triggered at episode {episode}:\n"
                    f"  - No improvement for {self.patience} episodes\n"
                    f"  - Best success rate: {self.best_success_rate:.3f}\n"
                    f"  - Best pose error: {self.best_pose_error:.4f}\n"
                    f"  - Best checkpoint: episode {self.best_checkpoint_episode}"
                )

        # Prepare return info
        return {
            'should_stop': should_stop,
            'reason': stop_reason,
            'improved': improved,
            'improvement_type': improvement_type,
            'rolling_success_rate': rolling_success,
            'rolling_pose_error': rolling_pose_error,
            'rolling_position_error': rolling_position_error,
            'rolling_orientation_error': rolling_orientation_error,
            'episodes_since_improvement': self.episodes_since_improvement,
            'best_success_rate': self.best_success_rate,
            'best_pose_error': self.best_pose_error,
            'best_checkpoint_episode': self.best_checkpoint_episode,
            'curriculum_complete': curriculum_complete
        }

    def _save_checkpoint(
        self,
        episode: int,
        models: Dict[int, torch.nn.Module],
        critic: torch.nn.Module,
        rolling_success: float,
        rolling_pose_error: float,
        is_best: bool = False
    ) -> str:
        """
        Save model checkpoint with metadata.

        Args:
            episode: Current episode number
            models: Dict of actor models
            critic: Critic model
            rolling_success: Current rolling success rate
            rolling_pose_error: Current rolling pose error
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_ep{episode}_best.pt" if is_best else f"checkpoint_ep{episode}.pt"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

        # Prepare checkpoint data
        checkpoint = {
            'episode': episode,
            'models': {agent_idx: model.state_dict() for agent_idx, model in models.items()},
            'critic': critic.state_dict() if critic else None,
            'metrics': {
                'rolling_success_rate': rolling_success,
                'rolling_pose_error': rolling_pose_error,
                'best_success_rate': self.best_success_rate,
                'best_pose_error': self.best_pose_error,
                'episodes_since_improvement': self.episodes_since_improvement
            },
            'tracker_state': {
                'success_history': list(self.success_history),
                'pose_error_history': list(self.pose_error_history),
                'position_error_history': list(self.position_error_history),
                'orientation_error_history': list(self.orientation_error_history)
            }
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        return checkpoint_path

    def load_best_checkpoint(
        self,
        models: Dict[int, torch.nn.Module],
        critic: torch.nn.Module = None
    ) -> bool:
        """
        Load the best checkpoint into the provided models.

        Args:
            models: Dict of actor models to load into
            critic: Critic model to load into

        Returns:
            True if successful, False otherwise
        """
        if not self.best_checkpoint_path or not os.path.exists(self.best_checkpoint_path):
            self.logger.warning("No best checkpoint found to load")
            return False

        try:
            checkpoint = torch.load(self.best_checkpoint_path)

            # Load actor models
            for agent_idx, model in models.items():
                if agent_idx in checkpoint['models']:
                    model.load_state_dict(checkpoint['models'][agent_idx])
                    self.logger.info(f"Loaded best model for agent {agent_idx}")

            # Load critic if provided
            if critic and checkpoint.get('critic'):
                critic.load_state_dict(checkpoint['critic'])
                self.logger.info("Loaded best critic model")

            self.logger.info(
                f"Restored checkpoint from episode {checkpoint['episode']} with "
                f"success rate {checkpoint['metrics']['rolling_success_rate']:.3f}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return False

    def get_statistics(self) -> Dict[str, any]:
        """Get current statistics."""
        return {
            'total_episodes': self.total_episodes,
            'best_success_rate': self.best_success_rate,
            'best_pose_error': self.best_pose_error,
            'best_checkpoint_episode': self.best_checkpoint_episode,
            'episodes_since_improvement': self.episodes_since_improvement,
            'checkpoints_saved': self.checkpoints_saved,
            'window_size': self.window_size,
            'patience': self.patience,
            'current_window_size': len(self.success_history)
        }

    def reset(self):
        """Reset tracker state (for new training run)."""
        self.success_history.clear()
        self.pose_error_history.clear()
        self.position_error_history.clear()
        self.orientation_error_history.clear()

        self.best_success_rate = 0.0
        self.best_pose_error = float('inf')
        self.best_checkpoint_episode = 0
        self.episodes_since_improvement = 0

        self.best_checkpoint_path = None
        self.last_checkpoint_path = None

        self.total_episodes = 0
        self.checkpoints_saved = 0

        self.logger.info("Performance tracker reset")
