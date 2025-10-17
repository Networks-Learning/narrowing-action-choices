from src.config import BURN_CYCLE, DEVICE
from src.utils import Status
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch


DEBUG = False


class Grid:
    def __init__(self, grid_config):
        """Initialize the map based on the grid_config"""
        # set up the map dimensions
        self.width = grid_config.width
        self.height = grid_config.height

        # create the state tensor with 5 channels
        # channels: DENSITY (channel 0), TIME TO BURN (channel 1), ALIVE (channel 2), BURNING (channel 3), BURNT (channel 4)
        self.state = torch.zeros(size=(self.width, self.height, 5), dtype=torch.float32, device=DEVICE)
        # initialize the density channel
        self.state[:,:,0] = torch.tensor(grid_config.densities, dtype=torch.float32, device=DEVICE)
        
        # number of alive cells
        self.cells_alive = self.width * self.height

        # number of cells that are on fire 
        self.cells_burning = 0

        # initialize the fire
        self.state[:,:,2] = 1  # all cells are alive at the start
        for i, j in grid_config.fire_coords:
            self.state[i, j, 3] = 1  # set the fire cells to burning
            self.state[i, j, 2] = 0  # set the fire cells to not alive
            self.cells_alive -= 1
            self.cells_burning += 1
            self.state[i, j, 1] = 1.0  # set the normalized time to burn to 1.0

        # initialize the spread random generator
        self.spread_generator = torch.Generator(device=DEVICE)
        self.spread_generator.manual_seed(grid_config.spread_seed)

        self.initial_env = deepcopy(self)
    
    def _burning_clock_update(self):
        # Find all burning cells
        burning_mask = self.state[:, :, 3] == 1
        
        # Decrement normalized time to burn for all burning cells (1/BURN_CYCLE per step)
        self.state[:, :, 1] = torch.where(burning_mask, 
                                         self.state[:, :, 1] - (1.0 / BURN_CYCLE), 
                                         self.state[:, :, 1])
        
        # Find cells that just finished burning
        finished_burning_mask = burning_mask & (self.state[:, :, 1] <= 0)
        
        if finished_burning_mask.any():
            # Update cells that finished burning from BURNING to BURNT status
            self.state[:, :, 3] = torch.where(finished_burning_mask, 0, self.state[:, :, 3])  # BURNING -> 0
            self.state[:, :, 4] = torch.where(finished_burning_mask, 1, self.state[:, :, 4])  # BURNT -> 1
            self.state[:, :, 1] = torch.where(finished_burning_mask, 0, self.state[:, :, 1])  # Reset time to 0
    
    def _spread_fire(self):
        """Spread fire using fully vectorized tensor operations"""
        alive_mask = self.state[:, :, 2] == 1
        burning_mask = self.state[:, :, 3] == 1
        
        if not burning_mask.any() or not alive_mask.any():
            return
        
        # Count burning neighbors using simple convolution
        kernel = torch.ones(3, 3, device=self.state.device)
        kernel[1, 1] = 0.0  # Don't count center cell
        burning_neighbor_count = torch.nn.functional.conv2d(
            burning_mask.float().unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=1
        ).squeeze()
        
        # Only consider alive cells that have burning neighbors
        candidates = alive_mask & (burning_neighbor_count > 0)
        
        if not candidates.any():
            return
        
        # For each candidate cell, draw n Bernoulli trials where n = number of burning neighbors
        # This is equivalent to: 1 - (1-p)^n where p is the cell's density
        cell_densities = self.state[:, :, 0]
        n_trials = burning_neighbor_count.int()
        prob_no_fire = torch.pow(1.0 - cell_densities, n_trials.float())
        prob_catch_fire = 1.0 - prob_no_fire
        
        random_samples = torch.rand(cell_densities.shape, generator=self.spread_generator, device=cell_densities.device)
        catches_fire = candidates & (random_samples < prob_catch_fire)
        
        # Apply new fires
        if catches_fire.any():
            self.state[:, :, 2] = torch.where(catches_fire, 0, self.state[:, :, 2])  # ALIVE -> 0
            self.state[:, :, 3] = torch.where(catches_fire, 1, self.state[:, :, 3])  # BURNING -> 1
            self.state[:, :, 1] = torch.where(catches_fire, 1.0, self.state[:, :, 1])  # Set normalized time to burn to 1.0

    def step(self, action_x, action_y):
        """One step transition given the coordinates of the selected cell"""
        
        # apply water to the selected cell (set from burning to burnt)
        self.state[action_x, action_y, 3] = 0  # BURNING -> 0
        self.state[action_x, action_y, 4] = 1  # BURNT -> 1
        self.state[action_x, action_y, 1] = 0  # time_to_burn -> 0

        # update the burning clock and burning status of the remaining cells
        self._burning_clock_update()

        # determine if the game ends --- if there are any burning cells left
        game_ends = self.state[:, :, 3].sum() == 0

        if not game_ends:
            # spread the fire if game has not ended 
            self._spread_fire()

        # update the counters
        self.cells_alive = int((self.state[:, :, 2] == 1).sum())
        self.cells_burning = int((self.state[:, :, 3] == 1).sum())

        return game_ends

    def get_grid_state_tensor(self):
        """Return the state tensor for RL training"""
        return self.state
    
    def get_grid_coarse_state_tensor(self):
        """Return the coarse state tensor"""
        # create the coarse state of the environment
        state_coarse = self.state.clone()
        state_coarse[:,:,0][self.state[:,:,0] < 0.3] = 0.2
        state_coarse[:,:,0][(self.state[:,:,0] >= 0.3) & (self.state[:,:,0] < 0.6)] = 0.45
        state_coarse[:,:,0][self.state[:,:,0] >= 0.6] = 0.75
        return state_coarse