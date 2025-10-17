import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from enum import Enum 
from FyeldGenerator import generate_field
import os
import pandas as pd
from collections import defaultdict
from copy import deepcopy
import pickle
from src.config import GridConfig, ROOT_DIR, FIELDS_PATH
#from environment import Grid
# from scipy.ndimage import gaussian_filter
import random


DEBUG = True

class Status(Enum):
    ALIVE = 1
    BURNING = 2
    BURNT = 3

def gen_field(shape, n, field_rng):
    """Generate a field of cells with random densities.
    
    Args:
        shape: Tuple specifying the field dimensions (height, width)
        n: Parameter controlling field density distribution
        field_rng: Random number generator to use
    """
    distr = lambda shape : field_rng.normal(0, 2, shape) + 1j * field_rng.normal(0, 2, shape)
    
    pk = lambda n : (lambda k : np.power(k, -n))
    
    return generate_field(distr, pk(n), shape)

def plot_field(shape, n=3, seed=42):
    """Plot the field of cells."""
    field = gen_field(shape, n, np.random.default_rng(seed))
    # normalize the field in (0,1)
    field = np.abs(field)
    field = (field - np.min(field)) / (np.max(field) - np.min(field))
    
    # discretize the field by rounding to the nearest 0.1
    field = np.round(field, 1)

    # remove extreme values
    field = np.clip(field, 0.1, 0.9)

    ax = plt.imshow(field, cmap='summer_r')
    # reverse the y-axis
    plt.gca().invert_yaxis()
    plt.colorbar()
    fig_field_dir = f'figures/{FIELDS_PATH}/{shape[0]}x{shape[0]}'
    if not os.path.exists(fig_field_dir):
        os.makedirs(fig_field_dir)
    plt.savefig(f'{fig_field_dir}/n_{n}_seed_{seed}.pdf', bbox_inches='tight')
    plt.clf()
    field_dir = f'outputs/{FIELDS_PATH}/{shape[0]}x{shape[0]}'
    if not os.path.exists(field_dir):
        os.makedirs(field_dir)
    np.savetxt(f'{field_dir}/n_{n}_seed_{seed}.txt', field, fmt='%.1f')


def create_fields(shape=(10, 10), init_seed=92894878398, n_maps=5):
    """Create density field maps controlled by a seed.
    
    Args:
        shape: Tuple specifying the field dimensions (height, width)
        init_seed: Master seed that controls map generation
        n_maps: Number of unique maps to generate
    """
    # Generate deterministic seeds derived from the init_seed
    field_rng = np.random.default_rng(init_seed)
    seeds = field_rng.integers(10000000, 99999999, n_maps)
    
    for _, seed in enumerate(seeds):
        plot_field(shape, n=10, seed=seed)
    
def create_fire(width, height, seed=42, n_patterns=4):
    """Create initial fire spread patterns controlled by a seed. (Used for training the DQN)
    
    Args:
        width: Grid width
        height: Grid height
        seed: Random seed to control fire pattern generation
        n_patterns: Number of unique fire patterns to generate
    """
    # Initialize with deterministic seed
    fire_rng = np.random.default_rng(seed)
    
    initial_spread_coords = []
    spread_dir = f'outputs/spreads/{width}x{height}'
    if not os.path.exists(spread_dir):
        os.makedirs(spread_dir)
    
    # Generate n_patterns different fire starting patterns
    for k in range(n_patterns):
        # Generate random fire locations with different patterns
        # For predictability, we'll use a structured approach with randomness
        
        # Get random fire origin point that's not too close to edges
        fire_i = fire_rng.integers(1, height-3)  
        fire_j = fire_rng.integers(1, width-3)
        
        initial_spread_coords.append([])
        # Create a small cluster of burning cells
        box_spread_neighbors = [(fire_i, fire_j), (fire_i, fire_j+1), 
                               (fire_i+1, fire_j), (fire_i+1, fire_j+1)]        
        
        for i, j in box_spread_neighbors:
            if i >= 0 and i < height and j >= 0 and j < width:
                initial_spread_coords[-1].append((i,j))
                
        np.savetxt(f'{spread_dir}/initial_spread_coords_{k+1}.txt', 
                  initial_spread_coords[-1], fmt='%d')


def neighbors(i, j):
    """Get the coordinates of the neighbors of a cell."""
    return [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j-1), (i, j+1), (i+1, j-1), (i+1, j),  (i+1, j+1)]

def isin_firefront(cells, i, j):
    """Check if a cell is in the firefront.

    Parameters
    ----------
    cells: list of lists
            grid of cells
    i: int
            Row index of the cell
    j: int
            Column index of the cell

    Returns
    -------
    is_firefront: boolean
            True if the cell is in the firefront, False otherwise
    """
    # check if any of the neighbors of the cell is alive
    nbs = neighbors(i, j)
    for n_i, n_j in nbs:
        if n_i >= 0 and n_i < len(cells) and n_j >= 0 and n_j < len(cells[0]):
            if cells[n_i][n_j].status == Status.ALIVE:
                return True
           
    return False

def get_state_dict(grid, epsilon, action_set_mask):
    """Convert the grid state and action set mask to a dictionary format."""
    game_state = {}
    for i in range(grid.width):
        for j in range(grid.height):
            is_in_set = bool(action_set_mask[i, j])
            game_state[str((i, j))] = \
            {
                "density": float(grid.cells[i][j].density), 
                "status": grid.cells[i][j].status.name.lower(),
                "time_to_burn": int(grid.cells[i][j].time_to_burn),
                "inSet": is_in_set,
                "addBorder": epsilon < 1.
            }
    return game_state