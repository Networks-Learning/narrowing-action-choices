"""
Configuration for DQN RL pipeline.
"""

import os
import torch

# Paths and device
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory structure
FIELDS_PATH = "fields"                      # Directory for field density maps
SPREADS_PATH = "spreads"                    # Directory for fire spread patterns
INSTANCES_PATH = "instances"                # Directory for train/test instance splits
AGENTS_PATH = "agents"                      # Directory for saving trained agents
RESULTS_PATH = "results"                    # Directory for evaluation results
LOGS_PATH = "logs"                          # Directory for training logs

# Environment parameters
WIDTH = 10                                  # Default grid width
HEIGHT = 10                                 # Default grid height
BURN_CYCLE = 3                              # Maximum time to burn

# Instance generation
GENERATION = {
    "generation_seed": 18740254,            # Seed for instance generation
    "train_per_category": 400,              # Number of training instances per difficulty category
    "test_per_category": 40,                # Number of testing instances per difficulty category
    "eval_spread_seeds": 10,                # Number of spread seeds for difficulty evaluation
    "max_attempts": 100,                    # Maximum generation attempts to find balanced sets
    "samples_per_attempt": 200,             # Number of random maps/fire patterns per generation attempt
    "difficulty_categories": {              # Difficulty categories defined by cells saved by greedy agent
        "very_hard": {"min": 0.0, "max": 0.20},    # 0-20% cells saved
        "hard": {"min": 0.20, "max": 0.40},        # 20-40% cells saved  
        "medium": {"min": 0.40, "max": 0.60},      # 40-60% cells saved
        "easy": {"min": 0.60, "max": 0.80},        # 60-80% cells saved
        "very_easy": {"min": 0.80, "max": 1.00}    # 80-100% cells saved
    }
}

# DQN Agent parameters
AGENT = {
    "gamma": 0.99,                          # Discount factor
    "epsilon": 1.0,                         # Initial exploration rate
    "epsilon_min": 0.1,                     # Minimum exploration rate
    "epsilon_decay": 300000,                 # Epsilon decay rate
    "batch_size": 128,                      # Batch size
    "memory_size": 200000,                  # Replay memory size
    "lr": 0.00001,                           # Learning rate
    "channels": 5,                          # Number of state channels
    "agent_seed": 817686876,                # Seed for agent's internal randomness
    "polyak_tau": 0.001,                     # Polyak averaging parameter for soft target updates
    "grad_clip": 10.0,                      # Gradient clipping max norm to prevent exploding gradients
}

# Training parameters
TRAINING = {
    "total_episodes": 150000,                # Total number of episodes to train
    "log_freq": 500,                         # Episodes between logging
    "eval_freq": 5000,                      # Episodes between test evaluations (0 to disable)
    "coarse_view": False,                   # Use coarse-grained state representation
    "stats_window": 5000,                   # Window size for computing average statistics
    "training_seed": 42,                    # Seed for training instance selection
    "spread_seed": 123456789,               # Seed for environment fire spread
    "eval_visualize": True,                 # Generate visualizations during training evaluation
    "eval_visualize_limit": 1,              # Maximum episodes to visualize during training evaluation
}

# Evaluation parameters
EVALUATION = {
    "visualize": False,                     # Generate visualizations
    "visualize_limit": 5,                   # Maximum episodes to visualize
    "multi_run_count": 10,                  # Number of runs per test instance with different seeds
}

# Neural network architecture parameters
NETWORK = {
    "conv_filters": [32, 32, 64, 64],           # Convolutional filter sizes
    "conv_kernel_size": [3, 3, 5, 7],          # Kernel sizes for convolutional layers: int to use the same kernel size for all layers
    "use_fc_layers": False,                 # Whether to use additional hidden fully connected layers
    "hidden_fc": [512, 256],                # Hidden layer sizes: int for single layer (e.g., 256) or list for multiple layers (e.g., [256, 128])
}

def get_output_paths(width, height):
    """Generate output paths based on grid dimensions."""
    dim = f"{width}x{height}"
    return {
        "fields": os.path.join("outputs", FIELDS_PATH, dim),
        "spreads": os.path.join("outputs", SPREADS_PATH, dim),
        "instances": os.path.join("outputs", INSTANCES_PATH, dim),
        "agents": os.path.join("outputs", AGENTS_PATH, dim, "dqn"),
        "results": os.path.join("outputs", RESULTS_PATH, dim, "dqn"),
        "logs": os.path.join("outputs", LOGS_PATH, dim, "dqn"),
    }

def ensure_output_dirs(width, height):
    """Ensure all output directories exist."""
    paths = get_output_paths(width, height)
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths
