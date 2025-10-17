from pickletools import dis
import pandas as pd
import numpy as np
import argparse
import os 
from copy import deepcopy

"""Bandit algorithms"""

def partition_interval(interval, num_parts=2):
    """Partition the given interval into num_parts equal parts."""
    start, end = interval
    step = (end - start) / num_parts
    return [(start + i * step, start + (i + 1) * step) for i in range(num_parts)]

def lipschitz_best_arm(arm_rewards, lipschitz_const=1.0, beta=1.0, n=100, eps_max=1.0, random_state=42, return_intervals=False):
    """Lipschitz best arm identification algorithm to find the best arm (epsilon value)
    Args:
        arm_rewards: DataFrame with multiple observed rewards (payoffs) for each arm (epsilon value)
        lipschitz_const: Lipschitz constant L
        beta: Exploration parameter
        n: Total budget (number of samples)
        eps_max: Maximum arm (epsilon) value
        random_state: Random state for reproducibility
        return_intervals: If True, return the intervals explored during the algorithm
    Returns:
        eps_opt: Best epsilon value found
        max_payoff: Maximum payoff achieved by the best epsilon value
        intervals (optional): DataFrame with the intervals explored during the algorithm
    """
    all_arms = arm_rewards.index.unique()
    t = 0 
    k = 1
    intervals = []
    active_intervals = partition_interval((0, eps_max))
    j = 0
    while t <= n:
        lk = 2 ** (-k)
        nk = 2 ** (k * beta)
        max_payoff = -np.inf
        eps_opt = 0.0
        empirical_payoff = {interval: 0.0 for interval in active_intervals}

        # exploration phase
        for i_low, i_high in active_intervals:
            epsilon_i = (i_low + i_high) / 2
            # search the closest arm to epsilon_i
            closest_arm = min(all_arms, key=lambda x: abs(x - epsilon_i))
            payoff_i = arm_rewards.loc[closest_arm].sample(nk, replace=True, random_state=random_state+j).mean()['reward']
            j += 1
            empirical_payoff[(i_low, i_high)] = payoff_i
            intervals.append((k, i_low, i_high, closest_arm, payoff_i))
            if payoff_i > max_payoff:
                max_payoff = payoff_i
                eps_opt = closest_arm
        len_active_intervals = deepcopy(len(active_intervals))
        # elimination phase
        new_active_intervals = []   
        for interval in active_intervals:
            if max_payoff - empirical_payoff[interval] <= (2 + lipschitz_const / 2) * lk:
                zoomed_intervals = partition_interval(interval, num_parts=2)
                new_active_intervals.extend(zoomed_intervals)
        active_intervals = new_active_intervals
        t += nk * len_active_intervals
        k += 1
        
    if return_intervals:
        return eps_opt, max_payoff, pd.DataFrame(intervals, columns=['iteration', 'interval_start', 'interval_end', 'epsilon_interval', 'empirical_payoff'])
    return eps_opt, max_payoff

def uniform_best_arm(arm_rewards, n_discrete=10, n=100, eps_max=1.0, random_state=42):
    """Uniform best arm identification algorithm to find the best arm (epsilon value)
    Args:
        arm_rewards: DataFrame with multiple observed rewards (payoffs) for each arm (epsilon value)
        n_discrete: Number of discretized arms (epsilon values) to consider
        n: Total exploration budget
        eps_max: Maximum arm (epsilon) value
        random_state: Random state for reproducibility
    Returns:
        eps_opt: Best epsilon value found
        max_payoff: Maximum payoff achieved by the best epsilon value
    """
    all_arms = arm_rewards.index.unique()
    active_intervals = partition_interval((0, eps_max), num_parts=n_discrete)
    max_payoff = -np.inf
    eps_opt = 0.0
    empirical_payoff = {interval: 0.0 for interval in active_intervals}
    j = 0
    # uniform exploration phase
    for i_low, i_high in active_intervals:
        epsilon_i = (i_low + i_high) / 2
        # search the closest arm to epsilon_i
        closest_arm = min(all_arms, key=lambda x: abs(x - epsilon_i))
        payoff_i = arm_rewards.loc[closest_arm].sample(n // n_discrete, replace=True, random_state=random_state+j).mean()['reward']
        j += 1
        empirical_payoff[(i_low, i_high)] = payoff_i
        if payoff_i > max_payoff:
            max_payoff = payoff_i
            eps_opt = closest_arm

    return eps_opt, max_payoff

def execute_bandits(cumu_reward_episode_per_arm, L=100, beta=3, gamma=0.99, reps=500):
    """Execute the bandit algorithms above and save the results to a CSV file."""
    rng = np.random.default_rng(2083202570)
    random_states = rng.integers(0, 2**32 - 1, size=reps)
    from tqdm import tqdm
    res = []
    for random_state in tqdm(random_states):
        for n in range(500, 80001, 1000):
            eps_opt, _ = lipschitz_best_arm(cumu_reward_episode_per_arm, lipschitz_const=L, beta=beta, n=n, eps_max=1.0, random_state=random_state)
            params = {
                'L': L,
                'beta': beta,
                'n': n,
                'eps_opt': eps_opt,
                'gamma': gamma,
                'random_state': random_state,
                'Alg': 'Ours'
            }
            res.append(params)
            eps_base, _ = uniform_best_arm(cumu_reward_episode_per_arm, n_discrete=100, n=n, eps_max=1.0, random_state=random_state)
            params = {
                'L': L,
                'beta': beta,
                'n': n,
                'eps_opt': eps_base,
                'gamma': gamma,
                'random_state': random_state,
                'Alg': 'Uniform'
            }
            res.append(params)

    res_df = pd.DataFrame(res)
    path = f"outputs/bandits/liba_vs_unif_results_L{L}_b{beta}_n{reps}.csv"
    if not os.path.exists('outputs/bandits/'):
        os.makedirs('outputs/bandits/')
    res_df.to_csv(path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=float, default=100, help='Lipschitz constant')
    parser.add_argument('--beta', type=int, default=3, help='Exploration parameter')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--n', type=int, default=500, help='Number of iterations')
    args = parser.parse_args()
    
    L = args.L
    beta = args.beta
    gamma = args.gamma
    n = args.n

    # read the study data
    study_data_path = 'human_study_data/study_data_complete.csv'
    study_data = pd.read_csv(study_data_path)
    
    unique_game_key_list = ['mapId', 'initLocId', 'seed', 'epsilon']
    # compute the discounted cumulative reward per game 
    study_data['discounted_reward'] = study_data[['reward', 'time_step']].apply(lambda x: int(x['reward']) * (np.round(gamma, decimals=2) ** int(x['time_step'])), axis=1)
    discounted_cumulative_rewards = study_data.groupby(unique_game_key_list).agg({'discounted_reward': 'sum'}).rename(columns={'discounted_reward': 'reward'}).reset_index()
    cumu_reward_episode_per_arm = discounted_cumulative_rewards.set_index('epsilon')

    execute_bandits(cumu_reward_episode_per_arm, L=L, beta=beta, gamma=gamma, reps=n)