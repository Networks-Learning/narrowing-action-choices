from src.config import BURN_CYCLE
from src.utils import isin_firefront, Status, neighbors
import numpy as np
from copy import deepcopy
from src.tensor_environment import Grid as GridNew

"""Class implementing the game environment"""

DEBUG = False

class Cell:
    def __init__(self, density):
        """Initialize a cell (map tile) with the given density"""
        self.density = density
        self.time_to_burn = BURN_CYCLE
        self.status = Status.ALIVE

    def get_cell_state(self):
        return (self.density, self.time_to_burn, self.status)


class Grid:
    def __init__(self, grid_config):
        """Initialize the forest map state based on the grid_config"""
        # set up the map dimensions
        self.width = grid_config.width
        self.height = grid_config.height
        
        # create the map with the given densities
        self.cells = [[ Cell(grid_config.densities[i][j]) for j in range(self.width)] for i in range(self.height)] 

        # number of alive cells
        self.cells_alive = self.width * self.height

        # number of cells that are on fire 
        self.cells_burning = 0

        # keep track of the fire front
        self.firefront = set()

        # initialize the fire (where the initial fire starts should be indicated in the grid_config)
        for i, j in grid_config.fire_coords:
            self.cells[i][j].status = Status.BURNING
            self.cells_alive -= 1
            self.cells_burning += 1
        
        for i in range(self.height):
            for j in range(self.width):
                if self.cells[i][j].status == Status.BURNING:
                    if isin_firefront(self.cells, i, j):
                        self.firefront.add((i, j))

        # initialize the spread random generator
        self.spread_rng = np.random.default_rng(seed=grid_config.spread_seed)

        self.coarse_state = None
        self.initial_env = deepcopy(self)
        # initialize the tensor environment (required for efficiency when using the DQN)
        self.new_grid = GridNew(grid_config=grid_config)
        
    def _status_update(self, action_x, action_y):
        """Update the status of the cell (tile) given the action (apply water)"""
        if self.cells[action_x][action_y].status == Status.ALIVE:
            # water in alive cells is no op 
            pass
        elif self.cells[action_x][action_y].status == Status.BURNING:
            self.cells[action_x][action_y].status = Status.BURNT
            if self.coarse_state is not None:
                # update the coarse state
                self.coarse_state[action_x, action_y, 2] = Status.BURNT.value
            
            self.cells[action_x][action_y].time_to_burn = 0
            # self.state_tensor[action_x, action_y, 1] = 0
            if self.coarse_state is not None:
                # update the coarse state
                self.coarse_state[action_x, action_y, 1] = 0

            self.cells[action_x][action_y].density = 0
            if self.coarse_state is not None:
                # update the coarse state
                self.coarse_state[action_x, action_y, 0] = 0

            self.cells_burning -= 1
            assert self.cells_burning >= 0, self.cells_burning
            if (action_x, action_y) in self.firefront:
                # remove the cell from the fire front
                self.firefront.remove((action_x,action_y))
        else:
            pass
    
    def _burning_clock_update(self):
        """Update the burning clock and burning status if necessary for each cell (tile)"""
        for i in range(self.height): 
            for j in range(self.width):
                if self.cells[i][j].status == Status.BURNING:
                    self.cells[i][j].time_to_burn -= 1
                    if self.coarse_state is not None:
                        # update the coarse state
                        self.coarse_state[i, j, 1] = self.cells[i][j].time_to_burn
                    if self.cells[i][j].time_to_burn == 0:
                        self.cells[i][j].status = Status.BURNT 
                        if self.coarse_state is not None:
                            # update the coarse state
                            self.coarse_state[i, j, 2] = Status.BURNT.value
                        self.cells_burning -= 1
                        assert self.cells_burning >= 0, self.cells_burning
                        self.cells[i][j].density = 0
                        if self.coarse_state is not None:
                            # update the coarse state
                            self.coarse_state[i, j, 0] = 0
                        if (i,j) in self.firefront:
                            # remove the cell from the firefront
                            self.firefront.remove((i,j))
                                    
    
    def _fire_front_update(self):
        """Spread the fire and update the fire front"""
        old_firefront = self.firefront.copy()
        canditate_firefront = set()
        # reward = 0.
        for (i,j) in old_firefront:
            # flag to check if cell has to be removed from the fire front
            remove_from_firefront = True
            # spread the fire to the neighbors
            for neib_i, neib_j in neighbors(i,j):
                if neib_i >= 0 and neib_i < self.height and neib_j >= 0 and neib_j < self.width:
                    if self.cells[neib_i][neib_j].status == Status.ALIVE:
                        # draw random variable to decide if the cell catches fire
                        spread = self.spread_rng.binomial(n=1, p=self.cells[neib_i][neib_j].density)
                        if spread:
                            self.cells[neib_i][neib_j].status = Status.BURNING
                            # update the state tensor
                            self.new_grid.state[neib_i, neib_j, 2] = 0  # ALIVE -> 0
                            self.new_grid.state[neib_i, neib_j, 3] = 1  # BURNING -> 1
                            self.new_grid.state[neib_i, neib_j, 1] = 1.0  # Set normalized time to burn to 1.0

                            if self.coarse_state is not None:
                                # update the coarse state
                                self.coarse_state[neib_i, neib_j, 2] = Status.BURNING.value
                            self.cells_alive -= 1
                            self.cells_burning += 1
            
                            # check if new burning cell is in the fire front
                            if isin_firefront(self.cells, neib_i, neib_j):
                                # add neighbor to the candidate fire front
                                canditate_firefront.add((neib_i, neib_j))
            
                        else:
                            # the cell is still in the fire front
                            remove_from_firefront = False
                            
            if remove_from_firefront:
                self.firefront.remove((i,j))       
        
        # update the fire front with the candidates still in the fire front
        for (i,j) in canditate_firefront:
            if isin_firefront(self.cells, i, j):
                self.firefront.add((i,j))
              
    def step(self, action_x, action_y):
        """One step transition given the coordinates of the selected cell"""
        # update the burning clock and burning status
        self._burning_clock_update()

        # update the clock for the tensor grid
        self.new_grid._burning_clock_update()
        
        # update the status of the selected cell given the action
        self._status_update(action_x, action_y)

        # apply water to the selected cell (set from burning to burnt)
        self.new_grid.state[action_x, action_y, 3] = 0  # BURNING -> 0
        self.new_grid.state[action_x, action_y, 4] = 1  # BURNT -> 1
        self.new_grid.state[action_x, action_y, 1] = 0  # time_to_burn -> 0

        game_ends = self.cells_burning == 0 or self.cells_alive == 0 or len(self.firefront) == 0
        #  spread the fire if game has not ended 
        if game_ends:
            return True
        else:
            # spread the fire and update the fire front
            self._fire_front_update()
            game_ends = self.cells_burning == 0 or self.cells_alive == 0 or len(self.firefront) == 0
            
            return game_ends        
