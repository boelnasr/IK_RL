# -*- coding: utf-8 -*-
#exploration.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# Initialize device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if CUDA is available and print device name
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Using CPU.")
class CountBasedExploration:
    def __init__(self):
        self.state_counts = defaultdict(int)

    def get_intrinsic_reward(self, state_key):
        self.state_counts[state_key] += 1
        return 1 / (self.state_counts[state_key] ** 0.5)

class CuriosityModule(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, device='cpu'):
        super(CuriosityModule, self).__init__()
        self.device = device
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        ).to(self.device)

    def forward(self, state, action, next_state):
        # Ensure inputs are on the correct device
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)

        concatenated_input = torch.cat([state, action], dim=1)
        predicted_next_state = self.forward_model(concatenated_input)
        intrinsic_reward = F.mse_loss(predicted_next_state, next_state, reduction='none').mean(dim=-1)
        return intrinsic_reward, predicted_next_state

class ExplorationModule:
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.device = device
        # Count-based exploration setup
        self.count_based_exploration = CountBasedExploration()
        # Curiosity-driven exploration setup
        self.curiosity_module = CuriosityModule(state_dim, action_dim, device=self.device)
        # Weighting factors
        self.count_based_weight = 1
        self.curiosity_weight = 1

    def get_count_based_reward(self, state):
        # Ensure state is detached and moved to CPU
        state_key = tuple(state.detach().cpu().numpy().flatten())
        return self.count_based_exploration.get_intrinsic_reward(state_key)

    def get_curiosity_based_reward(self, state, action, next_state):
        intrinsic_reward, _ = self.curiosity_module(state, action, next_state)
        return intrinsic_reward

    def get_combined_intrinsic_reward(self, state, action, next_state):
        # Ensure inputs are on the correct device
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)

        # Get count-based intrinsic reward
        count_reward = self.get_count_based_reward(state)

        # Get curiosity-based intrinsic reward
        curiosity_reward = self.get_curiosity_based_reward(state, action, next_state)

        # Combine the two rewards
        total_intrinsic_reward = (self.count_based_weight * count_reward) + \
                                 (self.curiosity_weight * curiosity_reward)

        return total_intrinsic_reward