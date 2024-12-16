import numpy as np
from collections import namedtuple
import torch

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'info'])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Enhanced Prioritized Experience Replay Buffer.
        
        Args:
            capacity (int): Maximum buffer size
            alpha (float): Priority exponent, controls how much prioritization is used
            beta_start (float): Initial beta value for importance sampling
            beta_frames (int): Number of frames over which to anneal beta to 1.0
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # Used for beta annealing
        
        # Main buffer storage
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        
        # Statistics tracking
        self.max_priority_ever = 1.0
        self.min_priority_ever = 1.0
        
        # Episode tracking
        self.episode_boundaries = []  # Store episode start indices
        self.current_episode = []  # Temporary storage for current episode
        
        # Priority statistics
        self.priority_stats = {
            'mean': [],
            'std': [],
            'max': [],
            'min': []
        }

    def _get_current_beta(self):
        """Calculate current beta value based on annealing"""
        beta = self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames)
        self.frame = min(self.frame + 1, self.beta_frames)
        return min(1.0, beta)

    def add(self, experience, priority=None):
        """
        Add experience to buffer with enhanced priority handling.
        
        Args:
            experience: Experience namedtuple
            priority (float): Optional priority value. If None, max priority is used
        """
        if priority is None:
            priority = self.max_priority_ever
        
        # Update priority statistics
        self.max_priority_ever = max(self.max_priority_ever, priority)
        self.min_priority_ever = min(self.min_priority_ever, priority)
        
        # Add to buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        
        # Update priority statistics
        if len(self.buffer) > 0:
            self.priority_stats['mean'].append(np.mean(self.priorities[:len(self.buffer)]))
            self.priority_stats['std'].append(np.std(self.priorities[:len(self.buffer)]))
            self.priority_stats['max'].append(np.max(self.priorities[:len(self.buffer)]))
            self.priority_stats['min'].append(np.min(self.priorities[:len(self.buffer)]))

    def sample(self, batch_size, device=None):
        """
        Enhanced sampling with device support and better numerical stability.
        
        Args:
            batch_size (int): Size of batch to sample
            device: torch device to send tensors to
        """
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        # Compute sampling probabilities with numerical stability
        probabilities = (priorities + 1e-7) ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices and compute importance sampling weights
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        beta = self._get_current_beta()
        
        # Compute weights with numerical stability
        weights = (len(self.buffer) * probabilities[indices] + 1e-10) ** (-beta)
        weights /= weights.max()  # Normalize weights

        # Prepare batch
        batch = [self.buffer[idx] for idx in indices]
        
        # Convert to torch tensors if device is specified
        if device is not None:
            weights = torch.FloatTensor(weights).to(device)
            batch = [Experience(
                torch.FloatTensor(exp.state).to(device),
                torch.FloatTensor(exp.action).to(device),
                torch.FloatTensor([exp.reward]).to(device),
                torch.FloatTensor(exp.next_state).to(device),
                torch.FloatTensor([float(exp.done)]).to(device),
                exp.info
            ) for exp in batch]

        return batch, weights, indices

    def update_priorities(self, indices, priorities):
        """
        Update priorities with clipping and stability checks.
        
        Args:
            indices (list): Indices to update
            priorities (list): New priority values
        """
        priorities = np.clip(priorities, 1e-8, None)  # Prevent zero priorities
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority_ever = max(self.max_priority_ever, priority)
            self.min_priority_ever = min(self.min_priority_ever, priority)

    def get_priority_stats(self):
        """Get statistics about priorities"""
        return {
            'current_mean': np.mean(self.priorities[:len(self.buffer)]),
            'current_std': np.std(self.priorities[:len(self.buffer)]),
            'max_ever': self.max_priority_ever,
            'min_ever': self.min_priority_ever,
            'priority_history': self.priority_stats
        }

    def add_episode(self, episode_experiences, priorities=None):
        """
        Add a complete episode to the buffer.
        
        Args:
            episode_experiences (list): List of experiences from one episode
            priorities (list): Optional list of priorities
        """
        episode_start = self.pos
        
        for i, exp in enumerate(episode_experiences):
            priority = priorities[i] if priorities is not None else None
            self.add(exp, priority)
            
        self.episode_boundaries.append((episode_start, self.pos))

    def clear(self):
        """Clear the buffer"""
        self.buffer = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0
        self.episode_boundaries = []
        self.current_episode = []
        
    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)

    def is_full(self):
        """Check if buffer is full"""
        return len(self.buffer) == self.capacity