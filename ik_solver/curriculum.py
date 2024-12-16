from collections import deque



class CurriculumManager:
    def __init__(self, initial_difficulty=1.0, max_difficulty=5.0, success_threshold=0.7, window_size=100):
        """
        Initializes the CurriculumManager.
        Args:
            initial_difficulty (float): Starting difficulty level.
            max_difficulty (float): Maximum difficulty level.
            success_threshold (float): Success rate required to move to the next stage.
            window_size (int): Number of episodes over which to calculate success rate.
        """
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.success_threshold = success_threshold
        self.success_history = deque(maxlen=window_size)
        self.window_size = window_size

    def log_success(self, success):
        """
        Logs the success of an episode.
        Args:
            success (bool): Whether the episode was successful.
        """
        self.success_history.append(success)

    def should_increase_difficulty(self):
        """
        Determines if the difficulty should be increased based on recent performance.
        Returns:
            bool: True if the difficulty should be increased, False otherwise.
        """
        if len(self.success_history) == self.window_size:
            success_rate = sum(self.success_history) / len(self.success_history)
            if success_rate >= self.success_threshold:
                return True
        return False

    def update_difficulty(self):
        """
        Updates the difficulty level if the performance threshold is met.
        """
        if self.should_increase_difficulty() and self.current_difficulty < self.max_difficulty:
            self.current_difficulty += 1
            self.success_history.clear()  # Reset history after advancing difficulty
            print(f"Difficulty increased to {self.current_difficulty}")

    def get_current_difficulty(self):
        """
        Gets the current difficulty level.
        Returns:
            float: The current difficulty level.
        """
        return self.current_difficulty




