import numpy as np

        
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        Initializes the Prioritized Experience Replay Buffer.

        Args:
            capacity (int): Maximum number of experiences to store.
            alpha (float): Controls the level of prioritization. alpha=0 gives uniform sampling.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, experience, priority=1.0):
        """
        Adds a new experience to the buffer with a given priority.

        Args:
            experience: The experience to store (tuple).
            priority (float): Priority for sampling this experience.
        """
        max_priority = self.priorities.max() if self.buffer else priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Samples a batch of experiences based on priority.

        Args:
            batch_size (int): Number of experiences to sample.
            beta (float): Bias correction factor for sampling probability.

        Returns:
            batch (list): Sampled experiences.
            weights (list): Importance-sampling weights for each experience.
            indices (list): Indices of sampled experiences in the buffer.
        """
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        # Compute the probability of each experience
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample experiences according to the computed probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[idx] for idx in indices]

        # Compute importance-sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        return batch, weights, indices

    def update_priorities(self, indices, priorities):
        """
        Updates the priorities of sampled experiences based on TD error.

        Args:
            indices (list): Indices of experiences to update.
            priorities (list): New priority values for the experiences.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
