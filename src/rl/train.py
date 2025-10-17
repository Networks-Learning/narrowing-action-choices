"""
Double DQN Training Script

This script handles training of the Double DQN agent using only the instances in the "train" set.
It loads the instances from the JSON file created by the generate_instances.py script.
The parameters used for training are set in config.py.

Usage:
    python -m src.rl.train
"""

import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import OrderedDict

from src.config import GridConfig
from src.tensor_environment import Grid
from src.rl.algorithms.dqn_agent import DQNAgent
from src.rl.algorithms.random_agent import RandomAgent
from src.rl.algorithms.heuristic_agent import HeuristicAgent
from src.rl.evaluate import evaluate_agent_on_instances
from src.rl.config import (
    DEVICE, WIDTH, HEIGHT, FIELDS_PATH, 
    AGENT, TRAINING, EVALUATION, GENERATION, ensure_output_dirs
)

def calculate_per_difficulty_stats(eval_results, test_instances):
    """
    Calculate average cells saved per difficulty level.
    
    Args:
        eval_results: Evaluation results dictionary
        test_instances: List of test instances with difficulty information
        
    Returns:
        Dictionary mapping difficulty levels to average cells saved percentages
    """
    # Create a mapping from instance_id to difficulty
    instance_to_difficulty = {}
    for instance in test_instances:
        instance_to_difficulty[instance["instance_id"]] = instance["difficulty"]
    
    # Group results by difficulty
    difficulty_results = {}
    for result in eval_results['instance_results']:
        instance_id = result['instance_id']
        if instance_id in instance_to_difficulty:
            difficulty = instance_to_difficulty[instance_id]
            if difficulty not in difficulty_results:
                difficulty_results[difficulty] = []
            # Use the averaged cells_saved_pct across multiple runs per instance
            difficulty_results[difficulty].append(result['cells_saved_pct'])
    
    # Calculate averages for each difficulty
    difficulty_averages = {}
    for difficulty, cells_saved_list in difficulty_results.items():
        if cells_saved_list:  # Only if we have results for this difficulty
            difficulty_averages[difficulty] = np.mean(cells_saved_list)
    
    return difficulty_averages

def train_dqn_on_instances(args):
    """
    Train a Double DQN agent on the instances in the train set.
    """
    
    # Get output paths and ensure directories exist
    paths = ensure_output_dirs(args.width, args.height)
    fields_path = paths["fields"]
    spread_path = paths["spreads"]
    instances_path = paths["instances"]
    models_path = paths["agents"]
    logs_path = paths["logs"]
    
    # Load instance data
    instance_file = f"{instances_path}/instances.json"
    if not os.path.exists(instance_file):
        raise FileNotFoundError(f"Instance file not found: {instance_file}. Run generate_instances.py first.")
    
    with open(instance_file, 'r') as f:
        instance_data = json.load(f)
    
    train_instances = instance_data["train_instances"]
    test_instances = instance_data["test_instances"]
    
    print(f"Training on {len(train_instances)} instances with grid size {args.width}x{args.height}")
    print(f"Parameters: gamma={args.gamma}, batch_size={args.batch_size}, lr={args.lr}, epsilon_decay={args.epsilon_decay}, polyak_tau={args.polyak_tau}")
    print(f"Training for {args.total_episodes} total episodes")
    
    # Create NumPy generator for training instance shuffling
    instance_rng = np.random.default_rng(args.training_seed)
    
    shuffled_instances = train_instances.copy()
    instance_rng.shuffle(shuffled_instances)
    instance_index = 0
    iteration_count = 0
    
    # Set global PyTorch seed for reproducible network weight initialization
    torch.manual_seed(args.training_seed)
    
    dqn_agent = DQNAgent(
        args.width, 
        args.height, 
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        channels=args.channels,
        agent_seed=args.agent_seed,
        lr=args.lr,
        polyak_tau=args.polyak_tau
    )
    
    # Load maps and spreads for all training instances (using ordered dicts for reproducibility)
    train_maps = OrderedDict()
    train_spreads = OrderedDict()
    for instance in train_instances:
        map_file = instance["map_file"]
        if map_file not in train_maps:
            train_maps[map_file] = np.loadtxt(f'{fields_path}/{map_file}', dtype=float)
        
        spread_file = instance["spread_file"]
        if spread_file not in train_spreads:
            train_spreads[spread_file] = np.loadtxt(f'{spread_path}/{spread_file}', dtype=int)
    
    # Load maps and spreads for all test instances
    test_maps = OrderedDict()
    test_spreads = OrderedDict()
    for instance in test_instances:
        map_file = instance["map_file"]
        if map_file not in test_maps:
            test_maps[map_file] = np.loadtxt(f'{fields_path}/{map_file}', dtype=float)
        
        spread_file = instance["spread_file"]
        if spread_file not in test_spreads:
            test_spreads[spread_file] = np.loadtxt(f'{spread_path}/{spread_file}', dtype=int)
    
    # Training statistics
    reward_history = []
    loss_history = []
    td_error_history = []
    cells_saved_history = []
    q_value_history = []
    total_timesteps = 0
    
    # Test set evaluation tracking
    dqn_test_scores = []
    
    # Checkpoint saving parameters
    checkpoint_interval = 10000  # Save model every 10k episodes
    last_checkpoint_episode = 0

    # Set up simple logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{logs_path}/training_{timestamp}.log"
    
    def log_print(message):
        """Print to both console and log file"""
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    # Log initial training parameters
    log_print(f"Training on {len(train_instances)} instances with grid size {args.width}x{args.height}")
    log_print(f"Parameters: gamma={args.gamma}, batch_size={args.batch_size}, lr={args.lr}, epsilon_decay={args.epsilon_decay}, polyak_tau={args.polyak_tau}")
    log_print(f"Training for {args.total_episodes} total episodes")
    log_print(f"Log file: {log_file}")
    log_print("")
    
    # Training loop
    log_print(f"Starting training for {args.total_episodes} episodes")
    
    for episode in range(1, args.total_episodes + 1):
        # Select next instance in shuffled order
        instance = shuffled_instances[instance_index]
        instance_index += 1
        
        # If we've used all instances, reshuffle and start over
        if instance_index >= len(shuffled_instances):
            instance_rng.shuffle(shuffled_instances)
            instance_index = 0
            iteration_count += 1
            log_print(f"Completed full iteration through all {len(train_instances)} training instances. Reshuffling for iteration #{iteration_count + 1}.")

        # Get instance data
        map_file = instance["map_file"]
        spread_file = instance["spread_file"]
        density = train_maps[map_file]
        spread = train_spreads[spread_file]
        
        # Create environment for this instance
        # Add iteration_count to spread_seed to ensure different fire patterns for same instance across iterations
        current_spread_seed = args.spread_seed + iteration_count
        grid_config = GridConfig(args.width, args.height, density, spread, current_spread_seed)
        env = Grid(grid_config)
        
            
        # Statistics for this episode
        total_reward = 0
        timestep = 0
        initial_cells = env.cells_alive
        cells_saved = 0
        
        while True:
            # Get current state
            current_state = env.get_grid_state_tensor()
            
            # Select action
            action = dqn_agent.act(current_state)
            action_x, action_y = action[0], action[1]
            action_idx = action[0] * args.width + action[1]  # Convert 2D action to 1D index
            
            # Phase 1: Start experience with current state and action
            dqn_agent.replay_buffer.start_experience(current_state, action_idx)
            
            cells_alive_before = env.cells_alive
            
            # Execute action in environment
            game_ends = env.step(action_x, action_y)
            
            cells_alive_after = env.cells_alive
            cells_lost = cells_alive_before - cells_alive_after
            
            # Reward design
            reward = -cells_lost    # Immediate reward: minus the number of cells that caught fire

            # Terminal reward based on total number of burnt cells
            if game_ends:
                cells_saved = env.cells_alive
                cells_burnt = initial_cells - cells_saved
                # Add terminal reward: minus the number of cells that did not make it
                reward -= cells_burnt
            
            # Phase 2: Complete experience with reward and next state
            if not game_ends:
                next_state = env.get_grid_state_tensor()  # Get state after action was applied
                dqn_agent.replay_buffer.complete_experience(reward, next_state)
            else:
                dqn_agent.replay_buffer.complete_experience(reward, None)  # Terminal state
            
            # Accumulate reward
            total_reward += dqn_agent.gamma ** timestep * reward
            
            # Learn from experience
            replay_result = dqn_agent.replay()
            if replay_result is not None:
                loss, _, _, td_error, q_stats = replay_result
                loss_history.append(loss)
                td_error_history.append(td_error)
                q_value_history.append(q_stats)
                
            # Update counters
            timestep += 1
            total_timesteps += 1
            
            # End episode if game is over
            if game_ends:
                break
                
        # Store episode statistics
        reward_history.append(total_reward)
        cells_saved_history.append(cells_saved / initial_cells)
        
        # Save checkpoint every 10k episodes
        if episode - last_checkpoint_episode >= checkpoint_interval:
            checkpoint_path = f"{models_path}/double_dqn_checkpoint_{episode}episodes.pth"
            dqn_agent.save_model(checkpoint_path)
            log_print(f"Checkpoint saved at episode {episode}: {checkpoint_path}")
            last_checkpoint_episode = episode
        
        # Calculate training metrics for logging using a rolling average
        avg_reward = np.mean(reward_history[-args.stats_window:]) 
        avg_loss = np.mean(loss_history[-args.stats_window:]) if loss_history else 0.0
        
        # Calculate TD error statistics
        if td_error_history:
            recent_td_errors = td_error_history[-args.stats_window:]
            avg_td_error = np.mean(recent_td_errors)
            median_td_error = np.median(recent_td_errors)
            p90_td_error = np.percentile(recent_td_errors, 90)
            p99_td_error = np.percentile(recent_td_errors, 99)
        else:
            avg_td_error = median_td_error = p90_td_error = p99_td_error = 0.0
            
        avg_cells_saved = np.mean(cells_saved_history[-args.stats_window:])
        current_epsilon = dqn_agent.get_current_epsilon()
        
        # Calculate Q-value statistics
        if q_value_history:
            recent_q_stats = q_value_history[-args.stats_window:]
            if recent_q_stats:
                avg_q_mean = np.mean([stat['mean'] for stat in recent_q_stats])
                avg_q_std = np.mean([stat['std'] for stat in recent_q_stats])
                min_q_value = np.min([stat['min'] for stat in recent_q_stats])
                max_q_value = np.max([stat['max'] for stat in recent_q_stats])
            else:
                avg_q_mean = avg_q_std = min_q_value = max_q_value = 0.0
        else:
            avg_q_mean = avg_q_std = min_q_value = max_q_value = 0.0
        
        # Progress logging
        if episode % args.log_freq == 0:
            log_print(f"Episode {episode}/{args.total_episodes}")
            log_print(f"  Avg Reward: {avg_reward:.4f}")
            log_print(f"  Avg Cells Saved: {avg_cells_saved:.2%}")
            log_print(f"  Avg Loss: {avg_loss:.6f}")
            log_print(f"  TD Error - Mean: {avg_td_error:.6f}, Median: {median_td_error:.6f}")
            log_print(f"  TD Error - 90th: {p90_td_error:.6f}, 99th: {p99_td_error:.6f}")
            log_print(f"  Q-Values - Mean: {avg_q_mean:.3f}, Std: {avg_q_std:.3f}")
            log_print(f"  Q-Values - Range: [{min_q_value:.3f}, {max_q_value:.3f}]")
            log_print(f"  Epsilon: {current_epsilon:.4f}")
            log_print(f"  Episode Length: {timestep} steps")
            log_print("")
        
        # Evaluate on test set at specified frequency
        if args.eval_freq > 0 and episode % args.eval_freq == 0:
            log_print(f"Evaluating on test set at episode {episode}...")
            log_print(f"Test set size: {len(test_instances)} episodes x {args.eval_multi_run_count} runs = {len(test_instances) * args.eval_multi_run_count} total simulations")
            
            # Temporarily disable exploration for DQN evaluation
            original_epsilon = dqn_agent.epsilon
            dqn_agent.epsilon = 0
            
            try:
                # Evaluate DQN agent with visualization for first episode
                dqn_eval_results = evaluate_agent_on_instances(
                    agent=dqn_agent,
                    test_instances=test_instances,
                    test_maps=test_maps,
                    test_spreads=test_spreads,
                    fields_path=fields_path,
                    spread_path=spread_path,
                    results_path=paths["results"],
                    width=args.width,
                    height=args.height,
                    gamma=args.gamma,
                    spread_seed=args.spread_seed,
                    verbose=False,
                    visualize=args.eval_visualize,
                    visualize_limit=args.eval_visualize_limit,
                    agent_type="dqn",  # Specify agent type for file naming
                    multi_run_count=args.eval_multi_run_count
                )
                
                # Create and evaluate random agent
                random_agent = RandomAgent(args.width, args.height, agent_seed=args.agent_seed)
                random_eval_results = evaluate_agent_on_instances(
                    agent=random_agent,
                    test_instances=test_instances,
                    test_maps=test_maps,
                    test_spreads=test_spreads,
                    fields_path=fields_path,
                    spread_path=spread_path,
                    results_path=paths["results"],
                    width=args.width,
                    height=args.height,
                    gamma=args.gamma,
                    spread_seed=args.spread_seed,
                    verbose=False,
                    visualize=False,  # No visualization for Random agent
                    visualize_limit=0,  # No visualization for Random agent
                    agent_type="random",  # Specify agent type for file naming
                    multi_run_count=args.eval_multi_run_count
                )
                
                # Create and evaluate greedy agent for comparison with visualization for first episode
                greedy_agent = HeuristicAgent(args.width, args.height, radius=1, agent_seed=args.agent_seed)
                greedy_eval_results = evaluate_agent_on_instances(
                    agent=greedy_agent,
                    test_instances=test_instances,
                    test_maps=test_maps,
                    test_spreads=test_spreads,
                    fields_path=fields_path,
                    spread_path=spread_path,
                    results_path=paths["results"],
                    width=args.width,
                    height=args.height,
                    gamma=args.gamma,
                    spread_seed=args.spread_seed,
                    verbose=False,
                    visualize=False,  # No visualization for Greedy agent
                    visualize_limit=0,  # No visualization for Greedy agent
                    agent_type="greedy",  # Specify agent type for file naming
                    multi_run_count=args.eval_multi_run_count
                )
                
                # Calculate per-difficulty statistics
                dqn_per_difficulty = calculate_per_difficulty_stats(dqn_eval_results, test_instances)
                random_per_difficulty = calculate_per_difficulty_stats(random_eval_results, test_instances)
                greedy_per_difficulty = calculate_per_difficulty_stats(greedy_eval_results, test_instances)
                                
                # Log evaluation results for all three agents
                log_print(f"Test Evaluation Results:")
                log_print(f"  DQN Agent:")
                log_print(f"    Avg Reward: {dqn_eval_results['average_reward']:.4f} ± {dqn_eval_results['std_reward']:.4f}")
                log_print(f"    Avg Cells Saved: {dqn_eval_results['average_cells_saved_pct']:.2f}% ± {dqn_eval_results['std_cells_saved_pct']:.2f}%")
                
                # Add per-difficulty breakdown for DQN
                log_print(f"    Per-difficulty averages:")
                for difficulty in sorted(dqn_per_difficulty.keys()):
                    log_print(f"      {difficulty}: {dqn_per_difficulty[difficulty]:.2f}%")
                
                log_print(f"    10th Percentile Cells Saved: {np.percentile([r['cells_saved_pct'] for r in dqn_eval_results['instance_results']], 10):.2f}%")
                log_print(f"    Avg Steps: {dqn_eval_results['average_steps']:.2f} ± {dqn_eval_results['std_steps']:.2f}")
                log_print(f"  Random Agent:")
                log_print(f"    Avg Reward: {random_eval_results['average_reward']:.4f} ± {random_eval_results['std_reward']:.4f}")
                log_print(f"    Avg Cells Saved: {random_eval_results['average_cells_saved_pct']:.2f}% ± {random_eval_results['std_cells_saved_pct']:.2f}%")
                
                # Add per-difficulty breakdown for Random
                log_print(f"    Per-difficulty averages:")
                for difficulty in sorted(random_per_difficulty.keys()):
                    log_print(f"      {difficulty}: {random_per_difficulty[difficulty]:.2f}%")
                
                log_print(f"    10th Percentile Cells Saved: {np.percentile([r['cells_saved_pct'] for r in random_eval_results['instance_results']], 10):.2f}%")
                log_print(f"    Avg Steps: {random_eval_results['average_steps']:.2f} ± {random_eval_results['std_steps']:.2f}")
                log_print(f"  Greedy Agent:")
                log_print(f"    Avg Reward: {greedy_eval_results['average_reward']:.4f} ± {greedy_eval_results['std_reward']:.4f}")
                log_print(f"    Avg Cells Saved: {greedy_eval_results['average_cells_saved_pct']:.2f}% ± {greedy_eval_results['std_cells_saved_pct']:.2f}%")
                
                # Add per-difficulty breakdown for Greedy
                log_print(f"    Per-difficulty averages:")
                for difficulty in sorted(greedy_per_difficulty.keys()):
                    log_print(f"      {difficulty}: {greedy_per_difficulty[difficulty]:.2f}%")
                
                log_print(f"    10th Percentile Cells Saved: {np.percentile([r['cells_saved_pct'] for r in greedy_eval_results['instance_results']], 10):.2f}%")
                log_print(f"    Avg Steps: {greedy_eval_results['average_steps']:.2f} ± {greedy_eval_results['std_steps']:.2f}")
                
                # Save DQN test scores for plotting
                test_eval_entry = {
                    "episode": episode,
                    "dqn_test_score": dqn_eval_results['average_cells_saved_pct']
                }
                dqn_test_scores.append(test_eval_entry)
                
                # Create scatter plot comparing DQN vs Greedy performance on individual test instances
                try:
                    # Extract mean performance and standard deviations for each instance
                    dqn_cells_saved = [result['cells_saved_pct'] for result in dqn_eval_results['instance_results']]
                    dqn_cells_saved_std = [result['cells_saved_pct_std'] for result in dqn_eval_results['instance_results']]
                    greedy_cells_saved = [result['cells_saved_pct'] for result in greedy_eval_results['instance_results']]
                    greedy_cells_saved_std = [result['cells_saved_pct_std'] for result in greedy_eval_results['instance_results']]
                    
                    # Create the scatter plot with error bars
                    plt.figure(figsize=(8, 8))
                    plt.errorbar(greedy_cells_saved, dqn_cells_saved, 
                                xerr=greedy_cells_saved_std, yerr=dqn_cells_saved_std,
                                fmt='o', alpha=0.7, markersize=6, capsize=3, capthick=1,
                                color='gray', markerfacecolor='gray', markeredgecolor='gray',
                                ecolor='lightgray')
                    
                    # Add diagonal line (y=x) for reference
                    min_val = min(min(greedy_cells_saved), min(dqn_cells_saved))
                    max_val = max(max(greedy_cells_saved), max(dqn_cells_saved))
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2, label=None)
                    
                    plt.xlabel('Greedy agent - Tiles saved (%) ± Std', fontsize=12)
                    plt.ylabel('DQN agent - Tiles saved (%) ± Std', fontsize=12)
                    
                    # Save the plot
                    comparison_plot_path = f"{paths['results']}/dqn_vs_greedy_episode_{episode}.png"
                    plt.tight_layout()
                    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    log_print(f"  Performance comparison plot saved: {comparison_plot_path}")
                    
                except Exception as plot_error:
                    log_print(f"  Failed to create comparison plot: {plot_error}")
                
                log_print("")
                
            except Exception as e:
                log_print(f"Evaluation failed: {e}")
                log_print("")
            finally:
                # Restore original epsilon
                dqn_agent.epsilon = original_epsilon

    # Save final model
    final_model_path = f"{models_path}/double_dqn_final_{args.total_episodes}episodes.pth"
    dqn_agent.save_model(final_model_path)
    log_print(f"Final model saved to {final_model_path}")
    
    # Save DQN test scores over time for plotting
    if dqn_test_scores:
        test_scores_file = f"{logs_path}/dqn_test_scores_{timestamp}.json"
        with open(test_scores_file, 'w') as f:
            json.dump(dqn_test_scores, f, indent=2)
        log_print(f"DQN test scores saved to {test_scores_file}")
    
    log_print("Training completed!")
    
    return dqn_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Double DQN agent")
    
    # Environment parameters
    parser.add_argument("--width", type=int, default=WIDTH, help="Width of the grid")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Height of the grid")
    parser.add_argument("--fields_path", type=str, default=FIELDS_PATH, help="Path to field density maps")
    
    # Agent parameters
    parser.add_argument("--gamma", type=float, default=AGENT["gamma"], help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=AGENT["epsilon"], help="Initial exploration rate")
    parser.add_argument("--epsilon_min", type=float, default=AGENT["epsilon_min"], help="Minimum exploration rate")
    parser.add_argument("--epsilon_decay", type=float, default=AGENT["epsilon_decay"], help="Epsilon decay rate (higher means slower decay)")
    parser.add_argument("--batch_size", type=int, default=AGENT["batch_size"], help="Batch size for experience replay")
    parser.add_argument("--memory_size", type=int, default=AGENT["memory_size"], help="Maximum size of replay buffer")
    parser.add_argument("--lr", type=float, default=AGENT["lr"], help="Learning rate")
    parser.add_argument("--channels", type=int, default=AGENT["channels"], help="Number of state channels")
    parser.add_argument("--agent_seed", type=int, default=AGENT["agent_seed"], help="Seed for agent's internal randomness")
    parser.add_argument("--training_seed", type=int, default=TRAINING["training_seed"], help="Seed for training instance selection")
    parser.add_argument("--spread_seed", type=int, default=TRAINING["spread_seed"], help="Seed for environment fire spread")
    parser.add_argument("--polyak_tau", type=float, default=AGENT["polyak_tau"], help="Polyak averaging parameter for soft target updates")
    
    # Training parameters
    parser.add_argument("--total_episodes", type=int, default=TRAINING["total_episodes"], help="Total number of episodes to train")
    parser.add_argument("--coarse_view", action="store_true", 
                        default=TRAINING["coarse_view"], help="Use coarse-grained state representation")
    parser.add_argument("--log_freq", type=int, default=TRAINING["log_freq"], help="How often to log progress (in episodes)")
    parser.add_argument("--eval_freq", type=int, default=TRAINING["eval_freq"], help="How often to evaluate on test set (in episodes, 0 to disable)")
    parser.add_argument("--stats_window", type=int, default=TRAINING["stats_window"], help="Window size for rolling statistics")
    
    # Evaluation parameters
    parser.add_argument("--eval_visualize", action="store_true", default=TRAINING["eval_visualize"], help="Enable visualization during evaluation")
    parser.add_argument("--eval_visualize_limit", type=int, default=TRAINING["eval_visualize_limit"], help="Limit number of visualizations during evaluation")
    parser.add_argument("--eval_multi_run_count", type=int, default=EVALUATION["multi_run_count"], help="Number of runs per instance during evaluation")
    
    args = parser.parse_args()
    train_dqn_on_instances(args)
