import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

def get_fig_dim(width, fraction=1, aspect_ratio=None):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    aspect_ratio: float, optional
            Aspect ratio of the figure

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    if aspect_ratio is None:
        # If not specified, set the aspect ratio equal to the Golden ratio (https://en.wikipedia.org/wiki/Golden_ratio)
        aspect_ratio = (1 + 5**.5) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in / aspect_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def latexify(font_serif='Computer Modern', mathtext_font='cm', font_size=10, small_font_size=None, usetex=True):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    font_serif: string, optional
		Set the desired font family
    mathtext_font: float, optional
    	Set the desired math font family
    font_size: int, optional
    	Set the large font size
    small_font_size: int, optional
    	Set the small font size
    usetex: boolean, optional
        Use tex for strings
    """

    if small_font_size is None:
        small_font_size = font_size

    params = {
        'backend': 'ps',
        'text.latex.preamble': '\\usepackage{gensymb} \\usepackage{bm}',
            
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'font.size': font_size,
        
        # Optionally set a smaller font size for legends and tick labels
        'legend.fontsize': small_font_size,
        'legend.title_fontsize': small_font_size,
        'xtick.labelsize': small_font_size,
        'ytick.labelsize': small_font_size,
        
        'text.usetex': usetex,    
        'font.family' : 'serif',
        'font.serif' : font_serif,
        'mathtext.fontset' : mathtext_font
    }

    matplotlib.rcParams.update(params)
    plt.rcParams.update(params)


def read_pkl(path):
    """
    Reads the data from the given path and returns a list.
    """
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def mean_cumulative_reward(path, gamma=0.99, indices=None, return_top_indices=False, threshold=None, return_dict=False):
    """
        Computes the mean cumulative reward across episodes.
    Args:
        path (str): Path to the file containing the rewards.
        gamma (float): Discount factor for future rewards.
    Returns:
        mean (float): Mean cumulative reward.
        ci95 (float): 95% confidence interval for the mean.
    """
    rewards_dict = read_pkl(path)
    cum_rewards = []
    cnt = 0
    rewards = defaultdict(lambda:[])
    if indices is not None:
        for conf in indices.keys():
            for i in indices[conf]:
                rewards[conf].append(rewards_dict[conf][i])
    else:
        rewards = rewards_dict        
    # print(len(rewards))
    
    cum_rewards_per_conf = {}
    top_indices = defaultdict(lambda:[])
    for conf in rewards.keys():
        cum_rewards_per_map = []
        for seed, rs in enumerate(rewards[conf]):
            
            cum_reward = 0
            for t, r in enumerate(rs):
                # print(r)
                cum_reward += (gamma ** t) * (r) 
            if threshold is not None:
                if cum_reward >= threshold:
                    cum_rewards.append(cum_reward)
                    
                    top_indices[conf].append(seed)
                    cnt += 1
            else:
                cum_rewards.append(cum_reward)
                cum_rewards_per_map.append(cum_reward)
        cum_rewards_per_conf[conf] = (np.mean(cum_rewards_per_map), 1.96 * np.std(cum_rewards_per_map) / np.sqrt(len(cum_rewards_per_map)))
    # print(len(cum_rewards))
    if return_dict:
        return cum_rewards_per_conf
        # print(cum_rewards)
    mean = np.mean(cum_rewards)
    ci95 = 1.96 * np.std(cum_rewards) / np.sqrt(len(cum_rewards))
    if not return_top_indices:
        return mean, ci95, cum_rewards_per_conf
    else:
        return mean, ci95, top_indices, cum_rewards_per_conf
    

def mean_set_size(path):
    """
        Computes the mean set size across episodes.
    Args:
        path (str): Path to the file containing the set sizes.
    Returns:
        mean (float): Mean set size.
        ci95 (float): 95% confidence interval for the mean.
    """
    set_sizes_dict = read_pkl(path)
    # flatten the list of lists
    # print(set_sizes_dict)
    
    set_sizes = []
    for conf in set_sizes_dict.keys():
        for i in set_sizes_dict[conf]:
            set_sizes.extend(i)
            # print(conf, i)
        # set_sizes = [item for sublist in set_sizes for item in sublist]
    # replace 100 with 1
    # set_sizes = [1 if i == 100 else i for i in set_sizes]
    mean = np.mean(set_sizes)
    ci95 = 1.96 * np.std(set_sizes) / np.sqrt(len(set_sizes))
    return mean, ci95


def cumulative_reward_per_game(base_path, agent_type, agent_params, gamma=0.99):
    """Read the rewards achieved by a family of heuristic agent an return a dataframe with the cumulative rewards per game."""
    data = []
    for param in agent_params:
    
        path = f"{base_path(agent_type, param)}"
        rewards_dict = read_pkl(path)
        for conf, episodes in rewards_dict.items():
            
            for seed, ep in enumerate(episodes):
                # print(ep)
                # raise
                ep[0][0]-=4
                cumulative_reward = sum( [ (gamma ** t) * r for t, r in enumerate(ep[0])] )
                data.append({
                    'game_id': f"{(conf[0], conf[1], conf[2])}",
                    'map_id': conf[0],
                    'init_loc_id': conf[1],
                    'seed': conf[2],
                    'epsilon': conf[3],
                    'agent_type': agent_type,
                    'agent_param': param,
                    'cumulative_reward': cumulative_reward
                })
            
                
    df = pd.DataFrame(data) 
    return df 

def spread_rate_per_game(path):
    """Read the fire spread per game per time step and return a dataframe with the fire spread rate per game."""
    data = []
    fires_dict = read_pkl(path)
    for conf, episodes in fires_dict.items():

        for seed, ep in enumerate(episodes):
            # print(ep)
            # raise
            ep[0][0]+=4
            spread_rate = np.mean(ep[0])
            data.append({
                'game_id': f"{(conf[0], conf[1], conf[2])}",
                'map_id': conf[0],
                'init_loc_id': conf[1],
                'seed': conf[2],
                'epsilon': conf[3],
                'spread_rate': spread_rate
            })
        
                
    df = pd.DataFrame(data) 
    return df 
        