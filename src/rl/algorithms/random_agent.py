"""
This module contains a simple random agent that randomly selects one burning tile
as an action.
"""

import torch
import numpy as np
import torch.nn.functional as F

from src.rl.config import DEVICE, AGENT


class RandomAgent:
    
    def __init__(self, width, height, agent_seed=None):
        self.width = width
        self.height = height
        
        # Set random seed
        self.agent_seed = agent_seed if agent_seed is not None else AGENT["agent_seed"]
        self.rng = np.random.default_rng(seed=self.agent_seed)
        
        # Create neighbor counting kernel (3x3 with center=0)
        neighbor_kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32)
        neighbor_kernel[0, 0, 1, 1] = 0
        self.neighbor_kernel = neighbor_kernel.to(DEVICE)
    
    def act(self, state, exploration=True):
        """
        Choose an action by randomly selecting one burning tile that has alive neighbors.
        If no such tiles exist, select any burning tile.
        
        Args:
            state: Current state tensor
            exploration: Used only for compatibility with the act() function of the DQN agent

        Returns:
            A tuple (x, y) representing the action coordinates
        """
        
        # State tensor format from environment: [width, height, channels]
        alive_channel = state[:, :, 2]
        burning_channel = state[:, :, 3]
        
        # Check which cells are burning
        is_burning = burning_channel > 0.5
        
        # Count alive neighbors for each cell using convolution
        alive_neighbor_count = F.conv2d(
            alive_channel.unsqueeze(0).unsqueeze(0),
            self.neighbor_kernel,
            padding=1
        ).squeeze()
        
        # Check if any neighbor is alive
        has_alive_neighbor = alive_neighbor_count > 0
        
        # Valid action if burning AND has alive neighbor
        valid_actions = is_burning & has_alive_neighbor
        
        # Fallback: if no valid actions, all burning cells are considered valid
        has_any_valid = valid_actions.flatten().any()
        final_actions = torch.where(has_any_valid, valid_actions, is_burning)
        
        # Get indices of final valid actions
        action_indices = torch.where(final_actions.flatten())[0]
        
        # Randomly select from valid actions
        action_index = action_indices[self.rng.integers(0, len(action_indices))].item()
        
        # Convert flat index to (x, y) coordinates
        x, y = divmod(action_index, self.height)
        return x, y
