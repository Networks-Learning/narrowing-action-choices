from src.config import GridConfig, ROOT_DIR, FIELDS_PATH, REWARDS_PATH, SEED, SPREADS_PATH, MODEL_PATH
from src.environment import Grid
from src.assistant import Assistant
from src.rl.algorithms.dqn_agent import DQNAgent
from src.rl.algorithms.heuristic_agent import HeuristicAgent
from src.rl.config import AGENT
from collections import defaultdict
import numpy as np 
import argparse
import os
import pickle
from copy import deepcopy
from tqdm import tqdm
import pandas as pd

"""Evaluate an agent playing multiple game instances with a decision support policy for a given epsilon value."""

DEBUG = False

def single_game(
        grid, 
        grid_config, 
        epsilon=1., 
        sigma=0.01, 
        set_seed=42,  
        human_seed=42,
        agent=None,
        human=None,
        coarse=False
    ):
    """Run a single episode of the agent playing a game instance with the given parameters 
    and return the rewards at each time step.
    
    Args:
        grid: The grid environment to run the episode on.
        grid_config: The configuration of the grid.
        epsilon: The epsilon value of the decision support policy.
        sigma: The sigma value for the decision support policy.
        set_seed: The seed for the action sets.
        human_seed: The seed for the simulated human (the one selecting actions) agent.
        agent: The AI agent to be used by the decision support policy.
        human: The agent we use to select actions from the action sets.
        coarse: Whether to use coarse state representation.
    Returns:
        rs: The rewards collected during the episode.
    """
    
    # create the assistant (decision support policy)
    assistant = Assistant(grid_config, epsilon=epsilon, sigma=sigma, seed=set_seed, agent=agent)
    
    # Set the simulated human agent seed
    if human is None:
        raise ValueError("Human agent must be provided")
    else:
        human.reset(seed=human_seed)
    time_step = 0
    game_ends = False
    
    rs = []
    if DEBUG: print(f"Simulation starts")
    
    while not game_ends:
        
        # compute the scores for each action
        action_scores_grid = assistant.state_action_scores(grid.new_grid, coarse=coarse)
        
        # a grid with 1 for the actions that are in the action set
        action_set_mask = assistant.action_set(action_scores_grid)
        
        if DEBUG:
            print(f"Human scores: {action_scores_grid}")
            print(f"Action set mask: {action_set_mask}")


        cells_before = deepcopy(grid.cells_alive)
        
        state = grid.new_grid.get_grid_state_tensor()
        action, human_score_grid = human.act(state, action_set_mask=action_set_mask, exploration=False, return_q_values=True)
        
        if DEBUG:
            print(f"Action: {action}")
            print(f"Human score grid: {human_score_grid}")
            print(f"State: {state}")
        
        game_ends = grid.step(action[0], action[1])
        cells_after = grid.cells_alive
        rs.append(cells_after - cells_before)
        
        time_step += 1
        if DEBUG: print(f"Time step: {time_step}")
        
    
    if DEBUG: 
        print(f"Simulation ends")
        print(f"Cells alive: {grid.cells_alive}")

    return rs
        

def single_initial_map(args, density, spread, seeds, conf, agent=None, human=None, mixed_agent=None):
    """Run a multiple game instances sharing the same initial state and 
    return the rewards.

    Args:
        args: The command line arguments.
        density: The densities of the map.
        spread: The initial spread coordinates of the fire.
        seeds: The seeds for each iteration.
        conf: The configuration of the map and spread.
        agent: The agent to use (if already created).
        human: The simulated human agent to use (if already created).
        mixed_agent: The simulated human agent using the mixture of policies.
    Returns:
        rewards: The rewards collected during the episodes.
        conf: The configuration of the map and spread.
    """
    rewards = []
    scores = []
    for seed in seeds:
        grid_config = GridConfig(args.width, args.height, density, spread, int(seed))
        # create the grid
        grid = Grid(grid_config)
        if args.human_policy == 'mix' and args.human_type == 'heuristic':
            if conf in mixed_agent:
                human = mixed_agent[conf]
            else:
                raise ValueError(f"No mixed agent found for configuration {conf}")
        rs = single_game(deepcopy(grid), grid_config, epsilon=args.arm, set_seed=seed, sigma=args.sigma, human_seed=seed, agent=agent, human=human, coarse=args.coarse)
        rewards.append(rs)
    return rewards, conf

if __name__ == "__main__":
    # read the grid config from command line arguments
    parser = argparse.ArgumentParser(description='Run the simulation.')
    parser.add_argument('--width', type=int, default=10, help='Width of the grid.')
    parser.add_argument('--height', type=int, default=10, help='Height of the grid.')
    parser.add_argument('--arm', type=float, default=1.0, help='Arm value.')
    parser.add_argument('--num_iterations', type=int, default=10, help='Number of iterations with different spread seeds per episode.')
    parser.add_argument('--sigma', type=float, default=0.01, help='Sigma value for the half normal distribution.')
    parser.add_argument('--human_policy', type=str, choices=['greedy', 'softmax', 'mix', 'random'], help='Type of the simulated heuristic human policy. Here by "human" we refer to the agent that takes the actions. ')
    parser.add_argument('--human_type', type=str, choices=['heuristic', 'dqn'], default='heuristic', help='Type of the heuristic score function used by the simulated human policy. Here by "human" we refer to the agent that takes the actions.')
    parser.add_argument('--human_param', type=int, default=1, choices=list(range(1,9)), help='Parameter (radius) for the human heuristic score function.') 
    parser.add_argument('--agent_type', type=str, choices=['heuristic', 'dqn'], help='Type of the agent used by the decision support policy.')
    parser.add_argument('--agent_param', type=int, default=1, choices=list(range(1,9)), help='Parameter (radius) for the agent heuristic score function.')
    parser.add_argument('--coarse', action='store_true', help='Use coarse state representation.')
    parser.add_argument('--games_from_file', action='store_true', help='Select game instances from file.')
    args = parser.parse_args()

    # read the map densities from file and the initial fire spreads
    fields_path = f"{ROOT_DIR}/../outputs/{FIELDS_PATH}/{args.width}x{args.height}"
    spread_pth = f"{ROOT_DIR}/../outputs/{SPREADS_PATH}/{args.width}x{args.height}"
    all_densities = [ np.loadtxt(f'{fields_path}/{field}', dtype=float) for field in sorted(os.listdir(fields_path)) ]
    all_spreads = [ np.loadtxt(f'{spread_pth}/{sp}', dtype=int) for sp in sorted(os.listdir(spread_pth)) ]
    seed = SEED
    
    # generate all random seeds with high entropy
    seeds = np.random.default_rng(seed).integers(10000000, 99999999, args.num_iterations * len(all_spreads) * len(all_densities))
    
    all_rewards = defaultdict(list)

    agent = None # policy used by the decision support system to create action sets
    human = None # policy used to select an action from action sets
     
    MIXED_AGENT = {} # dictionary to keep the mixture of heuristic policies in case human_policy is 'mix'
    # initialize the agent and human 
    if args.agent_type == 'dqn': # we use the trained dqn to construct the action sets
        agent = DQNAgent(
            args.width, args.height, 
            gamma=AGENT['gamma'], 
            channels=AGENT['channels'], 
            agent_seed=AGENT['agent_seed'],
        )
        agent.load_model(MODEL_PATH)
        agent.epsilon = 0  # No exploration during evaluation
    elif args.agent_type == 'heuristic': # we use a heuristic score to construct the action sets
        agent = HeuristicAgent(
            args.width, args.height, 
            radius=args.agent_param, 
            agent_seed=AGENT['agent_seed'],
        )
    if args.human_policy == 'greedy' and args.human_type == 'heuristic': # we use a greedy heuristic policy to select actions from the action sets
        human = HeuristicAgent(
            args.width, args.height, 
            radius=args.human_param, 
            agent_seed=AGENT['agent_seed'],
        )
    elif args.human_policy == 'greedy' and args.human_type == 'dqn': # we use greedily the trained dqn to select actions from the action sets
        human = DQNAgent(
            args.width, args.height, 
            gamma=AGENT['gamma'], 
            channels=AGENT['channels'], 
            agent_seed=AGENT['agent_seed'],
        )
        human.load_model(MODEL_PATH)
        human.epsilon = 0  # No exploration during evaluation
    elif args.human_policy == 'softmax' and args.human_type == 'heuristic': # we use a softmax heuristic policy to select actions from the action sets
        human = HeuristicAgent(
            args.width, args.height, 
            radius=args.human_param, 
            agent_seed=AGENT['agent_seed'],
            greedy=False
        )
    elif args.human_policy == 'mix' and args.human_type == 'heuristic': # we use a mixture of heuristic policies to select actions from the action sets
        # get the retrospective best mix of heuristics per map and initialization given the pilot data
        best_mix_path = f"{ROOT_DIR}/../outputs/hum_closest_mix_per_map_init.csv"
        best_mix_df = pd.read_csv(best_mix_path)
        # create a dictionary with the best heuristic agent for each (map_id, init_loc_id)
        for index, row in best_mix_df.iterrows():
            map_id, init_loc_id = eval(row['game_id'])
            heuristic_n = row['agent_param']
            is_greedy = row['policy_type'] == 'greedy'
            heuristic_agent = HeuristicAgent(width=10, height=10, radius=heuristic_n, agent_seed=42, greedy=is_greedy)
            MIXED_AGENT[(map_id, init_loc_id)] = heuristic_agent
    else:
        raise ValueError("Invalid human policy or type")
    
    if args.games_from_file:
        # read the game configurations from file
        study_data_path = f"{ROOT_DIR}/../human_study_data/study_data.csv"
        study_data = pd.read_csv(study_data_path)
        
        games_df = study_data[['mapId', 'initLocId', 'seed', 'epsilon']].drop_duplicates().reset_index(drop=True)
        all_rewards = defaultdict(list)

        for _, row in tqdm(games_df.iterrows()):
            map_id = int(row['mapId'])
            spread_id = int(row['initLocId'])
            density = all_densities[map_id]
            spread = all_spreads[spread_id]
            seeds = [int(row['seed'])]
            epsilon = row['epsilon']
            rs, conf = single_initial_map (args, density, spread, seeds, (map_id, spread_id, seeds[0], epsilon), agent=agent, human=human, mixed_agent=MIXED_AGENT)
            all_rewards[conf].append(rs)

    else:
        # collect the rewards for each episode and iteration
        seed_batch_idx = 0
        all_rewards = defaultdict(list)
        for map_id, density in tqdm(enumerate(all_densities)):
            for spread_id, spread in tqdm(enumerate(all_spreads)):
                rs, conf = single_initial_map (args, density, spread, seeds[seed_batch_idx:seed_batch_idx + args.num_iterations], (map_id, spread_id), agent=agent, human=human, mixed_agent=MIXED_AGENT)
                all_rewards[conf] = rs
                seed_batch_idx += args.num_iterations

    # save the rewards to files
    metric_name = REWARDS_PATH
    metric = all_rewards
    method = 'from_file' if args.games_from_file else f"num_seeds_{args.num_iterations}"
    metric_path = f"{ROOT_DIR}/../outputs/{metric_name}/{args.width}x{args.height}/{method}/sigma_{args.sigma}/hum_{args.human_policy}/{args.human_type}_{args.human_param}/machine_{args.agent_type}_{args.agent_param}/arm_{args.arm}"

    if not os.path.exists(metric_path):
        os.makedirs(metric_path)
    if args.coarse:
        metric_name += '_coarse'
    with open(f"{metric_path}/{metric_name}.pkl", 'wb') as f:
        pickle.dump(metric, f)
