"""
This module contains a heuristic agent that by default selects the burning tile on the firefront
that has the maximum total density in its surrounding neighbors within a given radius.
The module also includes to option to select a burning tile from a set of valid actions
provided via an action set mask and the option to select this tile using a softmax rule on the total density.
Weighs neighbors by the number of paths to reach them, computed using dynamic programming.
"""

import torch
import numpy as np
import torch.nn.functional as F

from src.rl.config import DEVICE, AGENT


class HeuristicAgent:
    
    def __init__(self, width, height, radius=1, agent_seed=None, greedy=True, **kwargs):
        self.width = width
        self.height = height
        self.radius = radius
        
        # Set random seed
        self.agent_seed = agent_seed if agent_seed is not None else AGENT["agent_seed"]
        self.rng = np.random.default_rng(seed=self.agent_seed)
        
        # Create path-counting kernel
        self.neighbor_kernel = self._create_exact_path_kernel(radius).to(DEVICE)
        
        # Calculate padding needed for convolution
        self.padding = radius

        # Create a simple kernel for checking for alive neighbors (radius 1)
        simple_kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32)
        simple_kernel[0, 0, 1, 1] = 0  # Set center to 0
        self.simple_kernel = simple_kernel.to(DEVICE)

        # Greedy action selection or softmax
        self.greedy = greedy
    
    def _create_exact_path_kernel(self, max_depth):
        """       
        For each cell at position (i,j) relative to center, the weight equals the total
        number of paths of length 1, 2, ..., max_depth that can reach (i,j).
        """
        kernel_size = 2 * max_depth + 1
        center = max_depth
        
        # Create DP table: dp[length][dx+center][dy+center] = number of paths of 'length' to (dx,dy)
        dp = torch.zeros(max_depth + 1, kernel_size, kernel_size, dtype=torch.float32)
        
        # Base case: 0 steps can only reach (0,0)
        dp[0, center, center] = 1.0

        # Neighbor offsets
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # Fill DP table bottom-up
        for length in range(1, max_depth + 1):
            for dx in range(-max_depth, max_depth + 1):
                for dy in range(-max_depth, max_depth + 1):
                    i, j = dx + center, dy + center
                    
                    # Skip if out of bounds
                    if i < 0 or i >= kernel_size or j < 0 or j >= kernel_size:
                        continue
                    
                    # Count paths by summing from all possible previous positions
                    total_paths = 0.0
                    for prev_dx, prev_dy in directions:
                        prev_x, prev_y = dx - prev_dx, dy - prev_dy
                        prev_i, prev_j = prev_x + center, prev_y + center
                        
                        # Check bounds for previous position
                        if (0 <= prev_i < kernel_size and 0 <= prev_j < kernel_size):
                            total_paths += dp[length - 1, prev_i, prev_j]
                    
                    dp[length, i, j] = total_paths
        
        # Sum across all path lengths to get final kernel weights
        kernel = torch.zeros(1, 1, kernel_size, kernel_size, dtype=torch.float32)
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == center and j == center:
                    continue  # Skip center cell
                
                # Sum paths of all lengths from 1 to max_depth
                total_weight = dp[1:, i, j].sum()
                kernel[0, 0, i, j] = total_weight
        
        return kernel
    
    def act(self, state, exploration=True, return_q_values=False, action_set_mask=None):
        """
        Choose an action by selecting the burning tile with maximum path-weighted surrounding density
        within the specified radius.
        
        Args:
            state: Current state tensor
            exploration: Used only for compatibility with the act() function of the DQN agent
            return_q_values: If True, returns the Q-values grid, i.e., the heuristic scores for each action.
            action_set_mask: Optional mask to filter valid actions given the action set
        Returns:
            If return_q_values is False: A tuple (x, y) representing the action coordinates
            If return_q_values is True: A tuple ((x, y), q_values_grid) where q_values_grid is a 2D numpy array
        """
        
        # State tensor format from environment: [width, height, channels]
        density_channel = state[:, :, 0]
        alive_channel = state[:, :, 2]
        burning_channel = state[:, :, 3]
        
        # Check which cells are burning
        is_burning = burning_channel > 0.5
        
        # Count immediate alive neighbors
        alive_neighbor_count = F.conv2d(
            alive_channel.unsqueeze(0).unsqueeze(0),
            self.simple_kernel,
            padding=1
        ).squeeze()
        
        # Check if any neighbor is alive
        has_alive_neighbor = alive_neighbor_count > 0
        
        # Valid actions: burning tiles with alive neighbors
        valid_actions = is_burning & has_alive_neighbor
        
        # Fallback: if no valid actions, all burning cells are considered valid
        if not valid_actions.any():
            valid_actions = is_burning
        
        # Apply the action set mask if provided
        if action_set_mask is not None:
            action_set_mask = torch.tensor(action_set_mask, device=DEVICE, dtype=torch.bool)
            valid_actions = valid_actions & action_set_mask

        # Calculate path-weighted density sum using exact path counts
        alive_density_channel = density_channel * alive_channel  # Only care about density of alive cells
        path_weighted_density_sum = F.conv2d(
            alive_density_channel.unsqueeze(0).unsqueeze(0),
            self.neighbor_kernel,
            padding=self.padding
        ).squeeze()
        
        # Mask out invalid actions by setting their path-weighted density sum to -infinity
        masked_density_sum = torch.where(
            valid_actions, 
            path_weighted_density_sum, 
            torch.tensor(-torch.inf, device=DEVICE)
        )
        
        if not self.greedy:
            # Softmax action selection among valid actions
            softmax_probs = torch.nn.functional.softmax(masked_density_sum.flatten(), dim=0).cpu().numpy()
            
            # Sample an action based on the softmax probabilities
            selected_idx = self.rng.choice(len(softmax_probs), 1, p=softmax_probs)[0]

        else:
            # Find all cells with maximum density and break ties
            max_density = torch.max(masked_density_sum)
            best_actions = (masked_density_sum == max_density) & valid_actions
            best_indices = torch.where(best_actions.flatten())[0]
            if len(best_indices) > 1:
                selected_idx = best_indices[self.rng.integers(0, len(best_indices))].item()
            else:
                selected_idx = best_indices[0].item()
            
        # Convert flat index to (x, y) coordinates
        x, y = divmod(selected_idx, self.height)

        if return_q_values:
            # Return the selected action coordinates and the q_values grid
            q_values_grid = masked_density_sum.cpu().numpy()
            return (x, y), q_values_grid
        else:
            # Return the selected action coordinates
            return x, y
        

    def reset(self, seed=None):
        """Reset the agent's random number generator with a new seed."""
        if seed is not None:
            self.agent_seed = seed
        self.rng = np.random.default_rng(seed=self.agent_seed)
    
    