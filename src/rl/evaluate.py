"""
Agent Evaluation Script

This script evaluates agents on the test set and saves results for analysis.
By default, it runs comprehensive evaluation of all agents (DQN, Random, Generalized Greedy variants).

Usage:
    # Comprehensive evaluation (default) - evaluates all agents, saves results to JSON/CSV
    python -m src.rl.evaluate --model_path outputs/agents/10x10/dqn/double_dqn_final_60000episodes.pth
    
    # Single agent evaluation with visualization (for DQN debugging)
    python -m src.rl.evaluate --single-agent --agent_type dqn \
        --model_path outputs/agents/10x10/dqn/double_dqn_final_60000episodes.pth \
        --visualize --visualize_limit 5
"""

import os
import json
import argparse
import torch
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from datetime import datetime

from src.config import GridConfig
from src.tensor_environment import Grid
from src.rl.algorithms.dqn_agent import DQNAgent
from src.rl.algorithms.random_agent import RandomAgent
from src.rl.algorithms.heuristic_agent import HeuristicAgent
from src.rl.config import (
    DEVICE, WIDTH, HEIGHT, FIELDS_PATH, EVALUATION, AGENT, TRAINING,
    ensure_output_dirs
)

def _load_test_data(fields_path, spread_path, instances_path):
    """
    Load test instances, maps, and spreads.
    """
    # Load instance data
    instance_file = f"{instances_path}/instances.json"
    if not os.path.exists(instance_file):
        raise FileNotFoundError(f"Instance file not found: {instance_file}. Run generate_instances.py first.")
    
    with open(instance_file, 'r') as f:
        instance_data = json.load(f)
    
    test_instances = instance_data["test_instances"]
    
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
    
    return test_instances, test_maps, test_spreads

def _create_agent(agent_type, width, height, gamma, agent_seed, channels, model_path=None):
    """
    Create an agent based on type and parameters.
    """
    if agent_type == 'dqn':
        if model_path is None:
            raise ValueError("model_path is required for DQN agent")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        agent = DQNAgent(width, height, gamma=gamma, channels=channels, agent_seed=agent_seed)
        agent.load_model(model_path)
        agent.epsilon = 0  # No exploration during evaluation
        return agent
    elif agent_type == 'random':
        return RandomAgent(width, height, agent_seed=agent_seed)
    elif agent_type == 'greedy':
        return HeuristicAgent(width, height, radius=1, agent_seed=agent_seed)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Supported types: 'dqn', 'random', 'greedy'")

def _run_agent_evaluation(agent, agent_name, test_instances, test_maps, test_spreads, 
                         width, height, gamma, spread_seed, multi_run_count=10, verbose=True):
    """
    Core evaluation function for a single agent on test instances.
    
    Args:
        agent: The agent to evaluate
        agent_name: Name of the agent for logging
        test_instances: List of test instances
        test_maps: Dictionary of test maps
        test_spreads: Dictionary of test spreads
        width: Grid width
        height: Grid height
        gamma: Discount factor
        spread_seed: Base spread seed
        multi_run_count: Number of runs per instance
        verbose: Whether to print progress
        
    Returns:
        Dictionary with evaluation results
    """
    if verbose:
        print(f"Evaluating {agent_name}...")
    
    results = []
    total_rewards = []
    total_steps = []
    total_cells_saved_pct = []
    
    for i, instance in enumerate(test_instances):
        # Get instance data
        map_file = instance["map_file"]
        spread_file = instance["spread_file"]
        instance_id = instance["instance_id"]

        density = test_maps[map_file]
        spread = test_spreads[spread_file]
        
        instance_rewards = []
        instance_steps = []
        instance_cells_saved_pct = []
        instance_initial_cells = None
        
        # Run multiple simulations for this instance with different seeds
        for run_idx in range(multi_run_count):

            current_spread_seed = spread_seed + run_idx
            grid_config = GridConfig(width, height, density, spread, current_spread_seed)
            env = Grid(grid_config)
            
            # Statistics for this run
            steps = 0
            initial_cells = env.cells_alive
            episode_reward = 0
            
            # Store initial_cells from first run
            if instance_initial_cells is None:
                instance_initial_cells = initial_cells
            
            # Run episode
            while True:
                
                # Get state and action
                state = env.get_grid_state_tensor()
                action = agent.act(state, exploration=False)
                
                cells_alive_before = env.cells_alive
                
                # Execute action
                done = env.step(action[0], action[1])
                
                # Compute reward
                cells_alive_after = env.cells_alive
                cells_lost = cells_alive_before - cells_alive_after
                
                reward = -cells_lost
                
                # Terminal reward based on burnt cells
                if done:
                    cells_saved = env.cells_alive
                    cells_burnt = initial_cells - cells_saved
                    reward -= cells_burnt
                
                # Apply discounting
                episode_reward += (gamma ** steps) * reward
                steps += 1
                
                if done:
                    break
            
            # Calculate cells saved percentage for this run
            cells_saved = env.cells_alive
            cells_saved_pct = cells_saved / initial_cells * 100
            
            # Store results
            instance_rewards.append(episode_reward)
            instance_steps.append(steps)
            instance_cells_saved_pct.append(cells_saved_pct)
        
        # Calculate average statistics for this instance across all runs
        avg_reward = np.mean(instance_rewards)
        avg_steps = np.mean(instance_steps)
        avg_cells_saved_pct = np.mean(instance_cells_saved_pct)
        std_reward = np.std(instance_rewards)
        std_steps = np.std(instance_steps)
        std_cells_saved_pct = np.std(instance_cells_saved_pct)
        
        if verbose and i % 10 == 0:
            print(f"  Instance {i+1}/{len(test_instances)} [{instance_id}]: "
                  f"Avg Cells saved={avg_cells_saved_pct:.1f}% (±{std_cells_saved_pct:.1f}%)")
        
        # Record results
        instance_result = {
            "instance_id": instance_id,
            "reward": avg_reward,
            "reward_std": std_reward,
            "steps": avg_steps,
            "steps_std": std_steps,
            "initial_cells": int(instance_initial_cells),
            "cells_saved_pct": avg_cells_saved_pct,
            "cells_saved_pct_std": std_cells_saved_pct,
            "multi_run_count": multi_run_count,
            "difficulty": instance.get("difficulty", "unknown")
        }
        results.append(instance_result)
        
        total_rewards.append(avg_reward)
        total_steps.append(avg_steps)
        total_cells_saved_pct.append(avg_cells_saved_pct)
    
    # Calculate summary statistics
    avg_reward = float(np.mean(total_rewards))
    std_reward = float(np.std(total_rewards))
    avg_steps = float(np.mean(total_steps))
    std_steps = float(np.std(total_steps))
    avg_cells_saved_pct = float(np.mean(total_cells_saved_pct))
    std_cells_saved_pct = float(np.std(total_cells_saved_pct))
    
    if verbose:
        print(f"  {agent_name} Results:")
        print(f"    Avg Reward: {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"    Avg Steps: {avg_steps:.2f} ± {std_steps:.2f}")
        print(f"    Avg Cells saved: {avg_cells_saved_pct:.2f}% ± {std_cells_saved_pct:.2f}%")
        print()
    
    return {
        "agent_name": agent_name,
        "num_instances": len(test_instances),
        "multi_run_count": multi_run_count,
        "total_simulations": len(test_instances) * multi_run_count,
        "average_reward": avg_reward,
        "std_reward": std_reward,
        "average_steps": avg_steps,
        "std_steps": std_steps,
        "average_cells_saved_pct": avg_cells_saved_pct,
        "std_cells_saved_pct": std_cells_saved_pct,
        "instance_results": results
    }

def evaluate_single_agent(agent, agent_name, test_instances, test_maps, test_spreads, 
                         width, height, gamma, spread_seed, multi_run_count=10, verbose=True):
    """
    Evaluate a single agent on all test instances (used by comprehensive mode).
    """
    return _run_agent_evaluation(agent, agent_name, test_instances, test_maps, test_spreads,
                                width, height, gamma, spread_seed, multi_run_count, verbose)

def evaluate_all_agents(args):
    """
    Comprehensive evaluation of multiple agents
    """
    
    # Get output paths and ensure directories exist
    paths = ensure_output_dirs(args.width, args.height)
    fields_path = paths["fields"]
    spread_path = paths["spreads"]
    instances_path = paths["instances"]
    
    dim = f"{args.width}x{args.height}"
    results_path = os.path.join("outputs", "results", dim, "testing")
    os.makedirs(results_path, exist_ok=True)
    
    # Load test data
    test_instances, test_maps, test_spreads = _load_test_data(fields_path, spread_path, instances_path)
    
    print(f"Evaluating agents on {len(test_instances)} test instances with grid size {args.width}x{args.height}")
    print(f"Each instance will be run {args.multi_run_count} times with different seeds")
    print(f"Total simulations per agent: {len(test_instances) * args.multi_run_count}")
    print()
    
    # Set seed for reproducibility
    torch.manual_seed(args.training_seed)
    
    # Initialize all agents
    all_results = {}
    
    # 1. DQN Agent (if model path provided)
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading DQN agent from: {args.model_path}")
        dqn_agent = DQNAgent(
            args.width, args.height, 
            gamma=args.gamma, 
            channels=args.channels, 
            agent_seed=args.agent_seed
        )
        dqn_agent.load_model(args.model_path)
        dqn_agent.epsilon = 0  # No exploration during evaluation
        
        dqn_results = evaluate_single_agent(
            dqn_agent, "DQN", test_instances, test_maps, test_spreads,
            args.width, args.height, args.gamma, args.spread_seed, args.multi_run_count
        )
        all_results["DQN"] = dqn_results
    else:
        if args.model_path:
            print(f"Warning: DQN model not found at {args.model_path}, skipping DQN evaluation")
        else:
            print("No DQN model path provided, skipping DQN evaluation")
    
    # 2. Random Agent
    random_agent = RandomAgent(args.width, args.height, agent_seed=args.agent_seed)
    random_results = evaluate_single_agent(
        random_agent, "Random", test_instances, test_maps, test_spreads,
        args.width, args.height, args.gamma, args.spread_seed, args.multi_run_count
    )
    all_results["Random"] = random_results
    
    # 3-9. Generalized Greedy Agents (radius 1-7)
    for radius in range(1, 8):
        agent_name = f"Gen Greedy (r={radius})"
        gen_greedy_agent = HeuristicAgent(
            args.width, args.height, radius=radius, agent_seed=args.agent_seed
        )
        gen_greedy_results = evaluate_single_agent(
            gen_greedy_agent, agent_name, test_instances, test_maps, test_spreads,
            args.width, args.height, args.gamma, args.spread_seed, args.multi_run_count
        )
        all_results[agent_name] = gen_greedy_results
    
    # Create timestamp for this evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for agent_name, results in all_results.items():
        print(f"{agent_name:15s}: {results['average_cells_saved_pct']:.2f}% ± {results['std_cells_saved_pct']:.2f}%")
    print()
    
    # Save detailed results to JSON
    results_file = f"{results_path}/all_agents_evaluation_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    return all_results

def evaluate_agent_on_instances(args=None, agent=None, test_instances=None, test_maps=None, test_spreads=None, 
                               fields_path=None, spread_path=None, results_path=None, 
                               width=None, height=None, gamma=None, spread_seed=None,
                               verbose=True, visualize=False, visualize_limit=0, agent_type=None, multi_run_count=None):
    """
    Evaluate an agent on the instances in the test set.
    
    This function can be called in two ways:
    1. From command line with args (standalone evaluation)
    2. From training script with direct parameters (integrated evaluation)
    
    Args:
        args: Command line arguments (for standalone use)
        agent: Pre-loaded agent (for integrated use)
        test_instances: List of test instances (for integrated use)
        test_maps: OrderedDict of test maps (for integrated use)
        test_spreads: OrderedDict of test spreads (for integrated use)
        fields_path: Path to fields directory (for integrated use)
        spread_path: Path to spreads directory (for integrated use)
        results_path: Path to results directory (for integrated use)
        width: Grid width (for integrated use)
        height: Grid height (for integrated use)
        gamma: Discount factor (for integrated use)
        spread_seed: Environment spread seed (for integrated use)
        verbose: Whether to print detailed progress (default True)
        visualize: Whether to generate visualizations (for integrated use)
        visualize_limit: Maximum number of episodes to visualize (for integrated use)
        agent_type: Type of agent for visualization naming (for integrated use)
        multi_run_count: Number of runs per test instance with different seeds (for integrated use)
        
    Returns:
        Dictionary with evaluation summary
    """
    
    if args is not None:
        # Standalone usage, extract parameters from args
        paths = ensure_output_dirs(args.width, args.height)
        fields_path = paths["fields"]
        spread_path = paths["spreads"]
        instances_path = paths["instances"]
        results_path = paths["results"]
        
        # Load test data
        test_instances, test_maps, test_spreads = _load_test_data(fields_path, spread_path, instances_path)
        
        if verbose:
            print(f"Evaluating on {len(test_instances)} test instances with grid size {args.width}x{args.height}")
            print(f"Agent type: {args.agent_type}")
            if args.agent_type == 'dqn':
                print(f"Using model: {args.model_path}")
            print(f"Test set size: {len(test_instances)} episodes x {multi_run_count} runs = {len(test_instances) * multi_run_count} total simulations")
        
        # Set global PyTorch seed for reproducible evaluation (same as training)
        training_seed = args.training_seed
        torch.manual_seed(training_seed)
        
        # Create agent for evaluation
        agent = _create_agent(args.agent_type, args.width, args.height, args.gamma, args.agent_seed, args.channels, args.model_path)
        
        # Extract other parameters from args
        width = args.width
        height = args.height
        gamma = args.gamma
        spread_seed = args.spread_seed
        visualize = args.visualize
        visualize_limit = args.visualize_limit
        multi_run_count = args.multi_run_count
        
    else:
        # Integrated usage, use provided parameters
        if any(param is None for param in [agent, test_instances, width, height, gamma, spread_seed]):
            raise ValueError("For integrated usage, agent, test_instances, width, height, gamma, and spread_seed must be provided")
        # Use visualization parameters passed to function
        # Set default multi_run_count if not provided
        if multi_run_count is None:
            multi_run_count = EVALUATION["multi_run_count"]
    
    if verbose and args is None:
        print(f"Evaluating on {len(test_instances)} test instances...")
        print(f"Test set size: {len(test_instances)} episodes x {multi_run_count} runs = {len(test_instances) * multi_run_count} total simulations")
    
    # Create a timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine agent name for logging
    agent_name = args.agent_type if args is not None else (agent_type if agent_type is not None else "Agent")
    
    # Run evaluation
    eval_results = _run_agent_evaluation(
        agent, agent_name, test_instances, test_maps, test_spreads,
        width, height, gamma, spread_seed, multi_run_count, verbose
    )
    
    # Extract results
    results = eval_results["instance_results"]
    
    # Handle visualization for first few instances if requested (single agent mode only)
    if visualize and results_path is not None and len(results) > 0:
        # For visualization, we need to re-run the first few instances to capture state history
        # This is only needed in single agent mode with visualization
        vis_agent_type = args.agent_type if args is not None else (agent_type if agent_type is not None else "unknown")
        
        for i in range(min(visualize_limit, len(test_instances))):
            instance = test_instances[i]
            instance_result = results[i]
            
            # Re-run first instance to get visualization data
            map_file = instance["map_file"]
            spread_file = instance["spread_file"]
            density = test_maps[map_file]
            spread = test_spreads[spread_file]
            
            # Create environment for visualization (using first run seed)
            current_spread_seed = spread_seed
            grid_config = GridConfig(width, height, density, spread, current_spread_seed)
            env = Grid(grid_config)
            
            # Run episode and capture states/actions for visualization
            states_history = []
            actions_history = []
            q_values_history = []
            
            while True:
                state_tensor = env.get_grid_state_tensor()
                states_history.append(state_tensor.clone())
                
                # Try to get Q-values from DQN agent, fall back to regular action for other agents
                try:
                    action, q_values_grid = agent.act(state_tensor, exploration=False, return_q_values=True)
                    q_values_history.append(q_values_grid)
                except TypeError:
                    # Agent doesn't support return_q_values parameter (e.g., Random, Greedy agents)
                    action = agent.act(state_tensor, exploration=False)
                
                actions_history.append(action)
                
                done = env.step(action[0], action[1])
                if done:
                    break
            
            # Generate visualization
            visualize_episode(
                states_history, 
                actions_history, 
                instance_result, 
                results_path, 
                timestamp,
                vis_agent_type,
                q_values_history if q_values_history else None
            )
    
    # Print evaluation summary
    if verbose:
        print("\nEvaluation Summary:")
        print(f"Total simulations: {eval_results['total_simulations']}")
        print(f"Average Reward: {eval_results['average_reward']:.4f} ± {eval_results['std_reward']:.4f}")
        print(f"Average Steps: {eval_results['average_steps']:.2f} ± {eval_results['std_steps']:.2f}")
        print(f"Average Cells Saved: {eval_results['average_cells_saved_pct']:.2f}% ± {eval_results['std_cells_saved_pct']:.2f}%")
    
    # Create evaluation summary
    evaluation_summary = {
        "timestamp": timestamp,
        "agent_type": args.agent_type if args is not None else "unknown",
        "num_instances": len(test_instances),
        "multi_run_count": multi_run_count,
        "total_simulations": eval_results['total_simulations'],
        "average_reward": eval_results['average_reward'],
        "std_reward": eval_results['std_reward'],
        "average_steps": eval_results['average_steps'],
        "std_steps": eval_results['std_steps'],
        "average_cells_saved_pct": eval_results['average_cells_saved_pct'],
        "std_cells_saved_pct": eval_results['std_cells_saved_pct'],
        "instance_results": results
    }
    
    # Save results only in standalone mode
    if args is not None and results_path is not None:
        agent_type = args.agent_type
        results_file = f"{results_path}/evaluation_results_{agent_type}_{timestamp}.json"
        evaluation_summary["agent_type"] = agent_type
        if agent_type == 'dqn':
            evaluation_summary["model_path"] = args.model_path
        
        with open(results_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        if verbose:
            print(f"Results saved to {results_file}")
    
    return evaluation_summary

def visualize_episode(states_history, actions_history, instance_result, results_path, timestamp, agent_type, q_values_history=None):
    """
    Create a simple grid visualization showing:
    - Background: Green shades for density
    - Letters: A (Alive), B (Burning), D (Burnt)
    - Action highlight: Red border for selected tile
    - Q-values: Displayed as numbers in each cell (if provided)
    
    Args:
        states_history: List of state tensors for each step
        actions_history: List of actions taken at each step
        instance_result: Dictionary with instance info
        results_path: Path to save visualizations
        timestamp: Timestamp for this evaluation
        agent_type: Type of agent (for naming)
        q_values_history: Optional list of Q-value grids for each step
    """
    instance_id = instance_result["instance_id"]
    steps = instance_result["steps"]
    reward = instance_result["reward"]
    cells_saved_pct = instance_result["cells_saved_pct"]
    
    # Create directory for visualizations
    visualizations_path = f"{results_path}/visualizations_{timestamp}"
    os.makedirs(visualizations_path, exist_ok=True)
    
    # Get density map from first state (fixed throughout episode)
    first_state = states_history[0].cpu().numpy()
    density_channel = first_state[:, :, 0]  # DENSITY channel
    density_normalized = np.clip(density_channel, 0, 1)
    
    # Create a visualization for each step
    for step, (state_tensor, action) in enumerate(zip(states_history, actions_history)):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Convert tensor to numpy
        state_np = state_tensor.cpu().numpy()
        
        # Extract channels
        alive_channel = state_np[:, :, 2]        # ALIVE
        burning_channel = state_np[:, :, 3]      # BURNING
        burnt_channel = state_np[:, :, 4]        # BURNT
        
        height, width = density_channel.shape
        
        # Create background based on density
        background = np.zeros((height, width, 3))
        for i in range(height):
            for j in range(width):
                density_val = density_normalized[i, j]
                
                red_component = 0.8 * (1 - density_val)
                green_component = 1.0 - 0.6 * density_val
                blue_component = 0.8 * (1 - density_val)
                
                background[i, j] = [red_component, green_component, blue_component]
        
        # Display the background
        ax.imshow(background, origin='upper')
        
        # Add letters for each cell state and Q-values if available
        for i in range(height):
            for j in range(width):
                letter = ""
                if burnt_channel[i, j] > 0:
                    letter = "D"  # Burnt (Dead)
                elif burning_channel[i, j] > 0:
                    letter = "B"  # Burning
                elif alive_channel[i, j] > 0:
                    letter = "A"  # Alive
                
                # Add Q-value if available
                q_value_text = ""
                if q_values_history is not None and step < len(q_values_history):
                    q_grid = q_values_history[step]
                    if not np.isnan(q_grid[i, j]):
                        q_value_text = f"\n{q_grid[i, j]:.2f}"
                
                # Combine letter and Q-value
                display_text = letter + q_value_text
                
                if display_text:
                    # Set colors: Red for burning (B), White for burnt (D), Black for alive (A)
                    if letter == 'B':
                        color = 'red'
                    elif letter == 'D':
                        color = 'white'
                    else:
                        color = 'black'
                    
                    # Adjust font size based on whether we have Q-values
                    font_size = 8 if q_value_text else 12
                    
                    ax.text(j, i, display_text, 
                           ha='center', va='center', 
                           fontsize=font_size, fontweight='bold', 
                           color=color)
        
        # Highlight the action with a red border
        if step < len(actions_history):
            act_x, act_y = action
            from matplotlib.patches import Rectangle
            rect = Rectangle((act_y-0.5, act_x-0.5), 1, 1, 
                           linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        ax.set_title(f"Step {step+1}/{steps}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save the visualization
        filename = f"{visualizations_path}/{agent_type}_{instance_id}_step{step+1:03d}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Episode visualization saved: {visualizations_path}/{agent_type}_{instance_id}_step*.png ({steps} steps)")

def main(args):
    """Main function for evaluating agents."""
    if args.comprehensive:
        # Run comprehensive evaluation of all agents
        return evaluate_all_agents(args)
    else:
        # Run single agent evaluation (original behavior)
        return evaluate_agent_on_instances(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agents on the test set")
    
    # Evaluation mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--comprehensive", action="store_true", default=True,
                           help="Run comprehensive evaluation of all agents (DQN, Random, Greedy with different radii)")
    mode_group.add_argument("--single-agent", action="store_true", 
                           help="Run single agent evaluation only (used during training)")
    
    # Agent type (for single agent evaluation)
    parser.add_argument("--agent_type", type=str, choices=['dqn', 'random', 'greedy'], default='dqn', 
                       help="Type of agent to evaluate (for single agent mode)")
    
    # Environment parameters
    parser.add_argument("--width", type=int, default=WIDTH, help="Width of the grid")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Height of the grid")
    parser.add_argument("--fields_path", type=str, default=FIELDS_PATH, help="Path to field density maps")
    
    # Agent parameters
    parser.add_argument("--gamma", type=float, default=AGENT["gamma"], help="Discount factor")
    parser.add_argument("--agent_seed", type=int, default=AGENT["agent_seed"], help="Seed for agent's internal randomness")
    parser.add_argument("--channels", type=int, default=AGENT["channels"], help="Number of state channels")
    parser.add_argument("--model_path", type=str, help="Path to the trained model file (required for DQN agent)")
    parser.add_argument("--spread_seed", type=int, default=TRAINING["spread_seed"], help="Seed for environment fire spread")
    parser.add_argument("--training_seed", type=int, default=TRAINING["training_seed"], help="Seed for PyTorch reproducibility")
    
    # Evaluation parameters
    parser.add_argument("--visualize", action="store_true", default=EVALUATION["visualize"], help="Generate visualizations of episodes")
    parser.add_argument("--visualize_limit", type=int, default=EVALUATION["visualize_limit"], help="Maximum number of episodes to visualize")
    parser.add_argument("--multi_run_count", type=int, default=EVALUATION["multi_run_count"], help="Number of runs per test instance with different seeds")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.comprehensive and args.agent_type == 'dqn' and args.model_path is None:
        parser.error("--model_path is required when using DQN agent in single agent mode")
    
    main(args)
