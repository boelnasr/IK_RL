from collections import deque

class CurriculumManager:
    def __init__(self, 
                 initial_difficulty=1.0, 
                 max_difficulty=5.0, 
                 min_difficulty=0.5,
                 success_threshold=0.7, 
                 window_size=100, 
                 difficulty_increment=0.5,
                 decay_rate=0.9,
                 num_agents=6):
        """
        Initializes the CurriculumManager.

        Args:
            initial_difficulty (float): Starting difficulty level.
            max_difficulty (float): Maximum difficulty level.
            min_difficulty (float): Minimum difficulty level.
            success_threshold (float): Success rate required to move to the next stage.
            window_size (int): Number of episodes over which to calculate success rate.
            difficulty_increment (float): Increment step for increasing difficulty (global).
            decay_rate (float): Rate at which difficulty decreases after failure (0 < decay_rate < 1).
            num_agents (int): Number of agents for per-agent difficulty.
        """
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.min_difficulty = min_difficulty
        self.success_threshold = success_threshold
        self.difficulty_increment = difficulty_increment
        self.decay_rate = decay_rate
        self.success_history = deque(maxlen=window_size)
        self.window_size = window_size

        # Per-agent difficulties and success rates
        self.num_agents = num_agents
        self.difficulties = [initial_difficulty] * num_agents
        self.success_rates = [0.0] * num_agents

    def log_success(self, success: bool):
        """
        Logs the success of an episode globally.

        Args:
            success (bool): Whether the episode was successful.
        """
        self.success_history.append(success)

    def calculate_success_rate(self) -> float:
        """
        Calculates the success rate over the history window.

        Returns:
            float: Success rate (0.0 to 1.0).
        """
        return sum(self.success_history) / len(self.success_history) if self.success_history else 0.0

    def should_increase_difficulty(self) -> bool:
        """
        Determines if the global difficulty should be increased based on recent performance.

        Returns:
            bool: True if the difficulty should be increased, False otherwise.
        """
        if len(self.success_history) == self.window_size:
            success_rate = self.calculate_success_rate()
            return success_rate >= self.success_threshold
        return False

    def should_decrease_difficulty(self) -> bool:
        """
        Determines if the global difficulty should be decreased due to poor performance.

        Returns:
            bool: True if the difficulty should be decreased, False otherwise.
        """
        if len(self.success_history) == self.window_size:
            success_rate = self.calculate_success_rate()
            return success_rate < (self.success_threshold / 2)
        return False

    def update_difficulty(self):
        """
        Dynamically adjusts the global difficulty level based on agent performance.
        """
        if self.should_increase_difficulty():
            new_difficulty = min(self.current_difficulty + self.difficulty_increment, self.max_difficulty)
            if new_difficulty != self.current_difficulty:
                self.current_difficulty = new_difficulty
                self.success_history.clear()  # Reset history after advancing difficulty
                print(f"Global difficulty increased to {self.current_difficulty:.2f}")

        elif self.should_decrease_difficulty():
            new_difficulty = max(self.current_difficulty * self.decay_rate, self.min_difficulty)
            if new_difficulty != self.current_difficulty:
                self.current_difficulty = new_difficulty
                self.success_history.clear()  # Reset history after difficulty decrease
                print(f"Global difficulty decreased to {self.current_difficulty:.2f}")

    def get_current_difficulty(self) -> float:
        """
        Gets the current global difficulty level.

        Returns:
            float: The current global difficulty level.
        """
        return self.current_difficulty

    def reset(self):
        """
        Resets the difficulty and history.
        """
        self.current_difficulty = self.min_difficulty
        self.success_history.clear()
        print("CurriculumManager reset to initial difficulty.")

    def get_agent_difficulty(self, agent_idx: int) -> float:
        """
        Gets the difficulty level for a specific agent.

        Args:
            agent_idx (int): Index of the agent.

        Returns:
            float: The difficulty level for the given agent.
        """
        return self.difficulties[agent_idx]

    def log_agent_success(self, agent_idx: int, success: bool):
        """
        Logs the success for a specific agent, updating its success rate.

        Args:
            agent_idx (int): Index of the agent.
            success (bool): True if the agent succeeded in the episode, False otherwise.
        """
        # Update the per-agent exponential moving average of success
        self.success_rates[agent_idx] = 0.9 * self.success_rates[agent_idx] + 0.1 * int(success)

    def update_agent_difficulty(self, agent_idx: int):
        """
        Updates the difficulty level for a specific agent based on its success rate.

        Logic:
          - If success rate > self.success_threshold, increase agent difficulty slightly.
          - If success rate < self.success_threshold/2, decrease agent difficulty slightly using decay.
          - Otherwise, leave it unchanged.
        """
        agent_success_rate = self.success_rates[agent_idx]
        agent_difficulty = self.difficulties[agent_idx]

        if agent_success_rate > self.success_threshold:
            # Increase difficulty
            new_difficulty = min(agent_difficulty + 0.1, self.max_difficulty)
            if new_difficulty != agent_difficulty:
                self.difficulties[agent_idx] = new_difficulty
                print(f"Agent {agent_idx} difficulty increased to {new_difficulty:.2f}")

        elif agent_success_rate < (self.success_threshold / 2.0):
            # Decrease difficulty
            new_difficulty = max(agent_difficulty * self.decay_rate, self.min_difficulty)
            if new_difficulty != agent_difficulty:
                self.difficulties[agent_idx] = new_difficulty
                print(f"Agent {agent_idx} difficulty decreased to {new_difficulty:.2f}")
