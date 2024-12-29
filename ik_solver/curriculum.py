from collections import deque
import numpy as np
import logging

class CurriculumManager:
    """
    A sophisticated curriculum manager that handles per-agent difficulty adaptation
    based on performance history and success rates. It uses rolling windows for
    more accurate performance tracking and implements proportional difficulty adjustments.
    """
    def __init__(
        self,
        initial_difficulty=1.0,
        max_difficulty=5.0,
        min_difficulty=0.5,
        success_threshold=0.7,
        window_size=100,
        difficulty_increment=0.2,
        decay_rate=0.95,
        num_agents=6
    ):
        """
        Initialize the curriculum manager with performance tracking for multiple agents.
        
        Args:
            initial_difficulty: Starting difficulty level for all agents
            max_difficulty: Upper bound for difficulty
            min_difficulty: Lower bound for difficulty
            success_threshold: Required success rate to increase difficulty
            window_size: Number of episodes to consider for success rate calculation
            difficulty_increment: Base increment for difficulty increases
            decay_rate: Base rate for difficulty decreases
            num_agents: Number of agents being managed
        """
        # Core parameters
        self.max_difficulty = max_difficulty
        self.min_difficulty = min_difficulty
        self.success_threshold = success_threshold
        self.difficulty_increment = difficulty_increment
        self.decay_rate = decay_rate
        self.window_size = window_size
        self.num_agents = num_agents

        # Initialize per-agent tracking
        self.difficulties = [initial_difficulty] * num_agents
        self.success_histories = [deque(maxlen=window_size) for _ in range(num_agents)]
        self.success_rates = [0.5] * num_agents  # Start at neutral value
        
        # Performance monitoring
        self.performance_stats = {
            'difficulty_changes': [[] for _ in range(num_agents)],
            'success_rate_history': [[] for _ in range(num_agents)]
        }

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('CurriculumManager')

    def log_agent_success(self, agent_idx: int, success: bool):
        """
        Record an agent's success/failure and update its success rate using
        a rolling window approach.
        
        Args:
            agent_idx: Index of the agent
            success: Whether the agent succeeded (True) or failed (False)
        """
        if not 0 <= agent_idx < self.num_agents:
            self.logger.error(f"Invalid agent index: {agent_idx}")
            return

        # Update success history
        self.success_histories[agent_idx].append(float(success))
        
        # Calculate new success rate based on recent history
        if self.success_histories[agent_idx]:
            self.success_rates[agent_idx] = np.mean(self.success_histories[agent_idx])
            
        # Track performance
        self.performance_stats['success_rate_history'][agent_idx].append(
            self.success_rates[agent_idx]
        )
        
        self.logger.info(
            f"Agent {agent_idx} - Success: {success}, "
            f"New success rate: {self.success_rates[agent_idx]:.3f}"
        )

    def update_agent_difficulty(self, agent_idx: int):
        """
        Update an agent's difficulty based on its recent performance.
        Uses proportional adjustments based on success/failure margins.
        
        Args:
            agent_idx: Index of the agent to update
        """
        if not 0 <= agent_idx < self.num_agents:
            self.logger.error(f"Invalid agent index: {agent_idx}")
            return

        agent_success_rate = self.success_rates[agent_idx]
        current_difficulty = self.difficulties[agent_idx]
        
        # Store previous difficulty for change tracking
        previous_difficulty = current_difficulty

        # Calculate performance margins
        success_margin = agent_success_rate - self.success_threshold
        
        # Adjust difficulty based on performance
        if success_margin > 0:
            # Increase difficulty proportionally to success margin
            increase_factor = 1.0 + (self.difficulty_increment * success_margin)
            new_difficulty = min(
                current_difficulty * increase_factor,
                self.max_difficulty
            )
        else:
            # Decrease difficulty proportionally to failure margin
            decrease_factor = self.decay_rate ** (1 + abs(success_margin))
            new_difficulty = max(
                current_difficulty * decrease_factor,
                self.min_difficulty
            )

        # Update difficulty and track change
        self.difficulties[agent_idx] = new_difficulty
        difficulty_change = new_difficulty - previous_difficulty
        self.performance_stats['difficulty_changes'][agent_idx].append(difficulty_change)

        # Log the update
        self.logger.info(
            f"Agent {agent_idx} - "
            f"Success rate: {agent_success_rate:.3f}, "
            f"Previous difficulty: {previous_difficulty:.3f}, "
            f"New difficulty: {new_difficulty:.3f}, "
            f"Change: {difficulty_change:+.3f}"
        )

    def get_agent_difficulty(self, agent_idx: int) -> float:
        """
        Get the current difficulty level for a specific agent.
        
        Args:
            agent_idx: Index of the agent
            
        Returns:
            Current difficulty level for the agent
        """
        if not 0 <= agent_idx < self.num_agents:
            self.logger.error(f"Invalid agent index: {agent_idx}")
            return self.min_difficulty
        return self.difficulties[agent_idx]

    def get_agent_stats(self, agent_idx: int) -> dict:
        """
        Get detailed statistics for a specific agent's performance.
        
        Args:
            agent_idx: Index of the agent
            
        Returns:
            Dictionary containing agent statistics
        """
        if not 0 <= agent_idx < self.num_agents:
            self.logger.error(f"Invalid agent index: {agent_idx}")
            return {}

        recent_changes = self.performance_stats['difficulty_changes'][agent_idx][-10:]
        recent_success_rates = self.performance_stats['success_rate_history'][agent_idx][-10:]

        return {
            'current_difficulty': self.difficulties[agent_idx],
            'current_success_rate': self.success_rates[agent_idx],
            'recent_difficulty_changes': recent_changes,
            'recent_success_rates': recent_success_rates,
            'success_history_length': len(self.success_histories[agent_idx]),
            'avg_recent_success_rate': np.mean(recent_success_rates) if recent_success_rates else 0.0
        }

    def reset_agent(self, agent_idx: int, initial_difficulty: float = None):
        """
        Reset an agent's curriculum state to initial values.
        
        Args:
            agent_idx: Index of the agent to reset
            initial_difficulty: Optional starting difficulty (defaults to min_difficulty)
        """
        if not 0 <= agent_idx < self.num_agents:
            self.logger.error(f"Invalid agent index: {agent_idx}")
            return

        initial_difficulty = initial_difficulty if initial_difficulty is not None else self.min_difficulty
        self.difficulties[agent_idx] = initial_difficulty
        self.success_histories[agent_idx].clear()
        self.success_rates[agent_idx] = 0.5
        
        self.logger.info(f"Reset agent {agent_idx} to initial difficulty: {initial_difficulty}")

    def reset_all(self, initial_difficulty: float = None):
        """
        Reset all agents to their initial state.
        
        Args:
            initial_difficulty: Optional starting difficulty for all agents
        """
        for agent_idx in range(self.num_agents):
            self.reset_agent(agent_idx, initial_difficulty)