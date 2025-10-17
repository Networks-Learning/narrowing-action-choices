import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from src.utils import create_fields, create_fire
from src.rl.config import ROOT_DIR, FIELDS_PATH, SPREADS_PATH, INSTANCES_PATH
from src.rl.config import GENERATION, WIDTH, HEIGHT, ensure_output_dirs, DEVICE
from src.rl.algorithms.heuristic_agent import HeuristicAgent
from src.tensor_environment import Grid
from src.config import GridConfig


class InstanceGenerator:
    """Generator for train/test instances across difficulty categories."""
    
    def __init__(self, width, height, eval_spread_seeds=3, generation_seed=42, max_attempts=50, samples_per_attempt=1000):
        """
        Initialize the balanced instance generator.
        
        Args:
            width: Width of the grid
            height: Height of the grid
            eval_spread_seeds: Number of spread seeds to use for evaluation
            generation_seed: Base seed for generation
            max_attempts: Maximum generation attempts to find balanced sets
            samples_per_attempt: Number of random instances to generate per attempt
        """
        self.width = width
        self.height = height
        self.eval_spread_seeds = eval_spread_seeds
        self.generation_seed = generation_seed
        self.max_attempts = max_attempts
        self.samples_per_attempt = samples_per_attempt
        
        # Create heuristic (greedy) agent with radius=7 for evaluation
        self.greedy_agent = HeuristicAgent(width, height, radius=7, agent_seed=42)
        
        # Get output paths
        self.paths = ensure_output_dirs(width, height)
        self.fields_path = self.paths["fields"]
        self.spread_path = self.paths["spreads"]
        self.instances_path = self.paths["instances"]
        
        print(f"Initialized BalancedInstanceGenerator ({width}x{height})")
        print(f"Evaluation: {eval_spread_seeds} spread seeds per instance")
        print(f"Max generation attempts: {max_attempts}")
        print(f"Samples per attempt: {samples_per_attempt}")
    
    def _clean_directories(self):
        """Clean existing instance files."""
        for dir_path in [self.fields_path, self.spread_path]:
            if os.path.exists(dir_path):
                for file_name in os.listdir(dir_path):
                    if file_name.endswith('.txt'):
                        os.remove(os.path.join(dir_path, file_name))
    
    def _generate_raw_instances(self, seed, attempt_suffix=""):
        """
        Generate raw instances (maps and fire patterns) using samples_per_attempt.
        
        Args:
            seed: Random seed for generation
            attempt_suffix: Suffix to add to file names to avoid conflicts across attempts
            
        Returns:
            List of instance dictionaries
        """
        
        # Generate maps and patterns
        n_maps = self.samples_per_attempt
        n_patterns = self.samples_per_attempt
        print(f"  Generating {n_maps} maps and {n_patterns} fire patterns...")
        create_fields(shape=(self.width, self.height), init_seed=seed, n_maps=n_maps)
        create_fire(self.width, self.height, seed=seed, n_patterns=n_patterns)
        
        # Collect files
        map_files = sorted([f for f in os.listdir(self.fields_path) if f.endswith('.txt') and f.startswith('n_10_seed_')])
        spread_files = sorted([f for f in os.listdir(self.spread_path) if f.endswith('.txt') and f.startswith('initial_spread_coords_')])
        
        # Rename files with attempt suffix to avoid conflicts
        if attempt_suffix:
            for i, map_file in enumerate(map_files):
                old_path = os.path.join(self.fields_path, map_file)
                new_name = f"attempt{attempt_suffix}_{map_file}"
                new_path = os.path.join(self.fields_path, new_name)
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    map_files[i] = new_name
            
            for i, spread_file in enumerate(spread_files):
                old_path = os.path.join(self.spread_path, spread_file)
                new_name = f"attempt{attempt_suffix}_{spread_file}"
                new_path = os.path.join(self.spread_path, new_name)
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    spread_files[i] = new_name
        
        print(f"  Generated {len(map_files)} map files and {len(spread_files)} spread files")
        
        if len(map_files) == 0 or len(spread_files) == 0:
            raise RuntimeError(f"Failed to generate files: {len(map_files)} maps, {len(spread_files)} spreads")
        
        # Create paired instances (map + fire pattern)
        min_files = min(len(map_files), len(spread_files))
        instances = []
        for i in range(min_files):
            instances.append({
                "map_file": map_files[i],
                "spread_file": spread_files[i],
                "map_idx": i,
                "spread_idx": i
            })
        
        print(f"  Created {len(instances)} paired instances (map-fire pattern pairs)")
        return instances
    
    def _evaluate_instance_difficulty(self, instance):
        """
        Evaluate the difficulty of a single instance using the heuristic agent.

        Args:
            instance: Instance dictionary with map_file and spread_file
            
        Returns:
            Dictionary with evaluation results
        """
        # Load density map and fire pattern
        density_path = os.path.join(self.fields_path, instance["map_file"])
        spread_path = os.path.join(self.spread_path, instance["spread_file"])
        
        if not os.path.exists(density_path):
            raise FileNotFoundError(f"Density file not found: {density_path}")
        if not os.path.exists(spread_path):
            raise FileNotFoundError(f"Spread file not found: {spread_path}")
        
        density = np.loadtxt(density_path, dtype=float)
        
        # Load fire coordinates
        fire_coords_raw = np.loadtxt(spread_path, dtype=int)
        if fire_coords_raw.ndim == 1 and len(fire_coords_raw) == 2:
            # Single coordinate pair
            fire_coords = [tuple(fire_coords_raw)]
        elif fire_coords_raw.ndim == 2 and fire_coords_raw.shape[1] == 2:
            # Multiple coordinate pairs
            fire_coords = [tuple(coord) for coord in fire_coords_raw]
        else:
            raise ValueError(f"Invalid fire coordinates format in {spread_path}: shape {fire_coords_raw.shape}")
        
        # Create a 2D array to represent the fire pattern for GridConfig
        spread = np.zeros((self.width, self.height), dtype=int)
        for i, j in fire_coords:
            if 0 <= i < self.width and 0 <= j < self.height:
                spread[i, j] = 1
        
        cells_saved_percentages = []
        
        # Evaluate across multiple spread seeds
        for spread_seed_offset in range(self.eval_spread_seeds):
            spread_seed = self.generation_seed + 1000 + spread_seed_offset
            
            # Create environment
            grid_config = GridConfig(self.width, self.height, density, fire_coords, spread_seed)
            env = Grid(grid_config)
            
            initial_cells = env.cells_alive
            
            # Run episode with heuristic (greedy) agent
            max_steps = 1000  # Just in case of buggy infinite loops
            steps = 0
            while steps < max_steps:
                # Get state
                state = env.get_grid_state_tensor()
                
                # Check if game is over (no burning cells)
                if (state[:, :, 3] == 0).all():
                    break

                # Get action from heuristic (greedy) agent
                action = self.greedy_agent.act(state, exploration=False)
                
                done = env.step(action[0], action[1])
                steps += 1
                
                if done:
                    break
            
            # Calculate cells saved percentage
            cells_saved = env.cells_alive
            cells_saved_pct = cells_saved / initial_cells if initial_cells > 0 else 0.0
            cells_saved_percentages.append(cells_saved_pct)
        
        # Average across spread seeds
        avg_cells_saved_pct = np.mean(cells_saved_percentages)
        
        # Classify difficulty using pre-defined difficulty categories
        difficulty_categories = GENERATION["difficulty_categories"]
        difficulty = None
        
        for category_name, thresholds in difficulty_categories.items():
            if thresholds["min"] <= avg_cells_saved_pct < thresholds["max"]:
                difficulty = category_name
                break
        
        # Handle edge case where avg_cells_saved_pct equals 1.0
        if difficulty is None and avg_cells_saved_pct >= 1.0:
            # Find the category with max threshold of 1.0
            for category_name, thresholds in difficulty_categories.items():
                if thresholds["max"] >= 1.0:
                    difficulty = category_name
                    break
        
        return {
            "difficulty": difficulty,
            "avg_cells_saved_pct": avg_cells_saved_pct,
            "cells_saved_percentages": cells_saved_percentages
        }
    
    def _evaluate_instances(self, instances):
        """
        Evaluate all instances and categorize by difficulty.
        
        Args:
            instances: List of instance dictionaries
            
        Returns:
            Dictionary mapping difficulty to list of instances
        """
        # Initialize categories from config
        difficulty_categories = {category: [] for category in GENERATION["difficulty_categories"].keys()}
        
        print(f"Evaluating {len(instances)} instances...")
        for instance in tqdm(instances, desc="Evaluating instances"):
            eval_result = self._evaluate_instance_difficulty(instance)
            difficulty = eval_result["difficulty"]
            
            # Add evaluation metadata to instance
            instance.update(eval_result)
            difficulty_categories[difficulty].append(instance)
    
        return difficulty_categories
    
    def _select_instances(self, difficulty_categories, target_per_category):
        """
        Select instances from difficulty categories.
        
        Args:
            difficulty_categories: Dict mapping difficulty to instance lists
            target_per_category: Target number of instances per category
            
        Returns:
            Tuple of (selected_instances, remaining_categories) where:
            - selected_instances: List of selected instances if sufficient, None otherwise
            - remaining_categories: Dict with instances that weren't selected
        """
        selected_instances = []
        
        # Get category names from config
        categories = list(GENERATION["difficulty_categories"].keys())
        remaining_categories = {category: [] for category in categories}
        
        # Check if we have enough in each category
        sufficient = True
        for category in categories:
            available = difficulty_categories[category]
            if len(available) < target_per_category:
                print(f"Insufficient {category} instances: {len(available)} < {target_per_category}")
                sufficient = False
        
        if not sufficient:
            # Return the instances we found
            return None, difficulty_categories
        
        # We have enough in each category, select balanced set
        for category in categories:
            available = difficulty_categories[category].copy()
            np.random.shuffle(available)
            selected = available[:target_per_category]
            remaining = available[target_per_category:]
            
            selected_instances.extend(selected)
            remaining_categories[category] = remaining
        
        return selected_instances, remaining_categories
    
    def _rename_and_save_instances(self, instances, prefix):
        """
        Rename instance files with prefix and update references.
        
        Args:
            instances: List of instance dictionaries
            prefix: Prefix for file names (e.g., 'train_', 'test_')
        """
        for instance in instances:
            # Rename map file
            old_map_path = os.path.join(self.fields_path, instance["map_file"])
            new_map_name = f"{prefix}{instance['map_file']}"
            new_map_path = os.path.join(self.fields_path, new_map_name)
            if os.path.exists(old_map_path):
                os.rename(old_map_path, new_map_path)
                instance["map_file"] = new_map_name
            
            # Rename spread file
            old_spread_path = os.path.join(self.spread_path, instance["spread_file"])
            new_spread_name = f"{prefix}{instance['spread_file']}"
            new_spread_path = os.path.join(self.spread_path, new_spread_name)
            if os.path.exists(old_spread_path):
                os.rename(old_spread_path, new_spread_path)
                instance["spread_file"] = new_spread_name
    
    def generate_instances(self, train_per_category, test_per_category):
        """
        Generate train and test instances.
        
        Args:
            train_per_category: Number of training instances per difficulty category
            test_per_category: Number of testing instances per difficulty category
            
        Returns:
            Dictionary with train and test instance information
        """
        print("Starting instance generation...")
        
        # Clean directories once at the very start
        self._clean_directories()
        
        train_instances = None
        test_instances = None
        
        # Generate training instances
        print(f"\nGenerating training instances ({train_per_category} per category)...")
        train_seed = self.generation_seed
        categories = list(GENERATION["difficulty_categories"].keys())
        accumulated_train_categories = {category: [] for category in categories}
        
        for attempt in range(self.max_attempts):
            print(f"Training attempt {attempt + 1}/{self.max_attempts}")
            
            # Show current status
            for category in categories:
                current_count = len(accumulated_train_categories[category])
                print(f"  Current {category} instances: {current_count}/{train_per_category}")
            
            raw_instances = self._generate_raw_instances(train_seed + attempt, f"_train_{attempt}")
            difficulty_categories = self._evaluate_instances(raw_instances)
            
            # Accumulate new instances with previous instances
            for difficulty in categories:
                accumulated_train_categories[difficulty].extend(difficulty_categories[difficulty])
            
            for category, instances in accumulated_train_categories.items():
                print(f"  Total {category}: {len(instances)} instances")
            
            # Try to select set from accumulated instances
            train_instances, remaining_categories = self._select_instances(accumulated_train_categories, train_per_category)
            
            if train_instances is not None:
                print(f"Successfully found training set on attempt {attempt + 1}")
                break
            else:
                print(f"Not enough instances yet, continuing to next attempt...")
                continue
        
        if train_instances is None:
            raise RuntimeError(f"Could not generate training set after {self.max_attempts} attempts")
        
        # Add instance IDs and rename files
        num_categories = len(GENERATION["difficulty_categories"])
        for i, instance in enumerate(train_instances):
            instance["instance_id"] = f"train_{instance['difficulty']}{i // num_categories + 1}"
        
        self._rename_and_save_instances(train_instances, "train_")
        
        # Generate testing instances
        print(f"\nGenerating testing instances ({test_per_category} per category)...")
        test_seed = self.generation_seed + 1000
        accumulated_test_categories = {category: [] for category in categories}
        
        for attempt in range(self.max_attempts):
            print(f"Testing attempt {attempt + 1}/{self.max_attempts}")
            
            # Show current status
            for category in categories:
                current_count = len(accumulated_test_categories[category])
                print(f"  Current {category} instances: {current_count}/{test_per_category}")
            
            # Generate new instances for testing
            raw_instances = self._generate_raw_instances(test_seed + attempt, f"_test_{attempt}")
            difficulty_categories = self._evaluate_instances(raw_instances)
            
            # Accumulate new instances with previous instances
            for difficulty in categories:
                accumulated_test_categories[difficulty].extend(difficulty_categories[difficulty])
            
            for category, instances in accumulated_test_categories.items():
                print(f"  Total {category}: {len(instances)} instances")
            
            # Try to select set from accumulated instances
            test_instances, remaining_categories = self._select_instances(accumulated_test_categories, test_per_category)
            
            if test_instances is not None:
                print(f"Successfully found testing set on attempt {attempt + 1}")
                break
            else:
                print(f"Not enough instances yet, continuing to next attempt...")
                continue
        
        if test_instances is None:
            raise RuntimeError(f"Could not generate testing set after {self.max_attempts} attempts")
        
        # Add instance IDs and rename files
        for i, instance in enumerate(test_instances):
            instance["instance_id"] = f"test_{instance['difficulty']}{i // num_categories + 1}"
        
        self._rename_and_save_instances(test_instances, "test_")
        
        # Prepare final data
        instance_data = {
            "grid_size": f"{self.width}x{self.height}",
            "generation_seed": self.generation_seed,
            "eval_spread_seeds": self.eval_spread_seeds,
            "train_per_category": train_per_category,
            "test_per_category": test_per_category,
            "train_instances": train_instances,
            "test_instances": test_instances,
            "train_difficulty_distribution": self._get_difficulty_distribution(train_instances),
            "test_difficulty_distribution": self._get_difficulty_distribution(test_instances)
        }
        
        # Save instances to JSON file
        instance_file = os.path.join(self.instances_path, "instances.json")
        with open(instance_file, 'w') as f:
            json.dump(instance_data, f, indent=2)
        
        print(f"\\nInstance generation complete!")
        print(f"Training instances: {len(train_instances)} ({train_per_category} per category)")
        print(f"Testing instances: {len(test_instances)} ({test_per_category} per category)")
        print(f"Instance data saved to {instance_file}")
        
        return instance_data
    
    def _get_difficulty_distribution(self, instances):
        """Get difficulty distribution statistics."""
        # Initialize distribution with all categories from config
        categories = list(GENERATION["difficulty_categories"].keys())
        distribution = {category: 0 for category in categories}
        
        for instance in instances:
            distribution[instance["difficulty"]] += 1
        return distribution


def generate_instances(width, height, train_per_category=None, test_per_category=None, 
                      eval_spread_seeds=None, generation_seed=None, max_attempts=None,
                      samples_per_attempt=None):
    """
    Generate instances of the game.
    
    Args:
        width: Width of the grid
        height: Height of the grid
        train_per_category: Number of training instances per difficulty category
        test_per_category: Number of testing instances per difficulty category
        eval_spread_seeds: Number of spread seeds for evaluation
        generation_seed: Base seed for generation
        max_attempts: Maximum generation attempts
        samples_per_attempt: Number of random instances to generate per attempt
        
    Returns:
        Dictionary with train and test instance information
    """
    # Use config defaults if not specified
    train_per_category = train_per_category if train_per_category is not None else GENERATION["train_per_category"]
    test_per_category = test_per_category if test_per_category is not None else GENERATION["test_per_category"]
    eval_spread_seeds = eval_spread_seeds if eval_spread_seeds is not None else GENERATION["eval_spread_seeds"]
    max_attempts = max_attempts if max_attempts is not None else GENERATION["max_attempts"]
    samples_per_attempt = samples_per_attempt if samples_per_attempt is not None else GENERATION["samples_per_attempt"]
    generation_seed = generation_seed if generation_seed is not None else GENERATION["generation_seed"]
    
    generator = InstanceGenerator(
        width=width,
        height=height,
        eval_spread_seeds=eval_spread_seeds,
        generation_seed=generation_seed,
        max_attempts=max_attempts,
        samples_per_attempt=samples_per_attempt
    )
    
    return generator.generate_instances(train_per_category, test_per_category)


def main(args):
    """Main function to run the instance generator."""
    return generate_instances(
        args.width, 
        args.height, 
        args.train_per_category,
        args.test_per_category,
        args.eval_spread_seeds,
        args.generation_seed,
        args.max_attempts,
        args.samples_per_attempt
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate game instances for DQN RL")
    
    parser.add_argument("--width", type=int, default=WIDTH, help="Width of the grid")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Height of the grid")
    parser.add_argument("--train_per_category", type=int, default=GENERATION["train_per_category"], 
                      help="Number of training instances per difficulty category")
    parser.add_argument("--test_per_category", type=int, default=GENERATION["test_per_category"], 
                      help="Number of testing instances per difficulty category")
    parser.add_argument("--eval_spread_seeds", type=int, default=GENERATION["eval_spread_seeds"], 
                      help="Number of spread seeds for evaluation")
    parser.add_argument("--generation_seed", type=int, default=GENERATION["generation_seed"], 
                      help="Seed for instance generation")
    parser.add_argument("--max_attempts", type=int, default=GENERATION["max_attempts"], 
                      help="Maximum generation attempts")
    parser.add_argument("--samples_per_attempt", type=int, default=GENERATION["samples_per_attempt"], 
                      help="Number of random instances to generate per attempt")
    
    args = parser.parse_args()
    main(args)
