import numpy as np

"""Class implementing the action sets"""

DEBUG = False

class Assistant():
    """Initialize the automated assistant."""
    def __init__(self, env_config, epsilon=0.1, sigma=1, gamma=0.99, seed=42, agent=None):
        self.epsilon = epsilon
        self.sigma = sigma
        self.gamma = gamma
        self.noise_rng = np.random.default_rng(seed=seed)
        
        self.width = env_config.width
        self.height = env_config.height
        self.env_config = env_config
        
        self.agent = agent
                
    def state_action_scores(self, grid, coarse=False):
        
        # get the current grid state
        state = grid.get_grid_state_tensor()
        
        # get a coarse representation of the state 
        # 3 levels of probability 0.1-0.3 --> 0.2, 0.3-0.6 --> 0.45, 0.6-0.9 --> 0.75
        if coarse:
            state = grid.get_grid_coarse_state_tensor()

        # get the action scores from the agent
        _, q_values_grid = self.agent.act(state, exploration=False, return_q_values=True)
        
        return q_values_grid
       
    def action_set(self, q_values_grid):
        """Return the set of possible actions for the given state."""
        
        # sample the noise value
        normal_noise = self.noise_rng.normal(0, self.sigma)
        half_normal_noise = abs(normal_noise)
        
        if DEBUG:
            print(f"Assistant q_values grid before scaling: {q_values_grid}")

        # get the maximum q_values that are not nan
        q_values_grid_max = np.nan_to_num(q_values_grid, nan=-np.inf)
        if DEBUG:
            print(f"Assistant q_values grid after nan to num max: {q_values_grid_max}")
        max_q = np.max(q_values_grid_max)
        q_values_grid_min = np.nan_to_num(q_values_grid, nan=+np.inf)
        if DEBUG:
            print(f"Assistant q_values grid after nan to num min: {q_values_grid_min}")
        min_q = np.min(q_values_grid_min)
        # scale the q_value in [0, 1]
        if max_q - min_q > 0:
            q_values_grid = (q_values_grid - min_q) / (max_q - min_q)
        else:
            q_values_grid = (q_values_grid - min_q)
        
        # scaled max_q value
        q_values_grid_max = np.nan_to_num(q_values_grid, nan=-np.inf)
        max_q = np.max(q_values_grid_max)

        # threshold for the q_values
        threshold = max_q - self.epsilon - half_normal_noise
       
        assert np.any(q_values_grid >= threshold), f"{q_values_grid, threshold}"  # Ensure at least one q_values are above the threshold

        # a grid with 1 for the actions that are in the action set
        action_set_mask = q_values_grid >= threshold

        return action_set_mask