import random
import numpy as np
import matplotlib.pyplot as plt

from class_reward_function import RewardBinary

from utils_worlds import GridWorldSparse40x40Mixed, GridWorld40x40Circles


class GridWorld:
    """
    GridWorld environment generated from a given list of distributions
    """

    colors = {
        "goal":(0, 0, 255),
        "agent_start":(255, 0, 0),
        "agent_current":(255, 0, 0),
        "trajectory":(255, 0, 0),
    }

    sizes = {
        "cell":13,
        "wall":2,
    }

    max_sample_failure = 1000

    def __init__(
        self,
        size,
        start_distribution,
        distributions,
        goal_type="mix 1",
        reward_function=RewardBinary(),
        is_guessing_game=False,
        ):
        """
        is_guessing_game : agent will only guess a single cell
        goal_type : 
                      "uniform N" uniformly sample a reward from all goals, N times;
                      "mix N" sample a reward from a random distribution, N times;
                      "unique N" N rewards per distribution;
                      "all" all potential goals are reward;
        """

        self.size = size

        self.potential_goal_areas = []
        for g in distributions:
            new_area = g._make_shape(np.zeros([self.size, self.size, 3]))
            self.potential_goal_areas += new_area
        self.distributions = distributions

        goal_split = goal_type.split(" ")
        self.goal_type = goal_split[0]
        self.nb_samples = int(goal_split[1]) if len(goal_split) > 1 else None

        self.reward_function = reward_function
        
        self.is_guessing_game = is_guessing_game
        self.start_distribution = start_distribution

        # Intialiazing start distribution
        self.start_distribution._make_shape(np.zeros([self.size, self.size, 3]))

        self.current_pos = None
        self.state_hist = []

        self.reset()

    def reset(self):
        """
        Resets the environment and defines new goals aswell as a new starting position
        """

        self.reward_coords = []
        if self.goal_type == "uniform":
            if self.nb_samples >= len(self.potential_goal_areas):
                raise ValueError("Failed to sample, shapes might be too small")
                
            self.reward_coords = random.sample(
                population=self.potential_goal_areas,
                k=self.nb_samples
            )

        elif self.goal_type == "mix":
            i, failed = 0, 0

            while i < self.nb_samples:
                new_goal = tuple(random.choice(self.distributions).sample())
                
                if new_goal in self.reward_coords:
                    failed += 1
                    if failed >= self.max_sample_failure:
                        raise ValueError("Failed to sample, shapes might be too small")
                    continue
                
                self.reward_coords.append(new_goal)
                i += 1
        
        elif self.goal_type == "unique":
            for distribution in self.distributions:
                if self.nb_samples >= len(distribution.potential_area):
                    raise ValueError("Failed to sample, shapes might be too small")
                
                self.reward_coords += random.sample(
                    population=distribution.potential_area,
                    k=self.nb_samples
                )

        
        elif self.goal_type == "all":
            self.reward_coords = self.potential_goal_areas.copy()
        
        self.init_pos = self.start_distribution.sample()
        self.state_hist = [tuple(self.init_pos)]

        self.current_pos = self.init_pos
        return self.init_pos

    def step(self, action):
        """
        Actions :
            0 left
            1 right
            2 up
            3 down

        Returns :
            current position, reward, done
        """

        if self.is_guessing_game is False:

            if action == 0:
                self.current_pos = [
                    self.current_pos[0],
                    max(0, self.current_pos[1] - 1)
                ]

            elif action == 1:
                self.current_pos = [
                    self.current_pos[0],
                    min(self.size - 1, self.current_pos[1] + 1)
                ]

            elif action == 2:
                self.current_pos = [
                    max(0, self.current_pos[0] - 1),
                    self.current_pos[1]
                ]

            elif action == 3:
                self.current_pos = [
                    min(self.size - 1, self.current_pos[0] + 1),
                    self.current_pos[1]
                ]
            
            done = self.current_pos in self.reward_coords

        else:
            self.current_pos = action
            done = True

        self.state_hist.append(self.current_pos)
        return self.current_pos, self.reward_function(self), done

    def _world_to_grid(self, row, col):
        """
        Returns the real coordinates of the given row and col
        """

        return (
            row * (self.sizes["wall"] + self.sizes["cell"]) + self.sizes["wall"],
            col * (self.sizes["wall"] + self.sizes["cell"]) + self.sizes["wall"]
        )
    
    def _draw_cell_in_grid(self, grid, row, col, color):
        """
        Draws a cell in the grid
        """

        start = self._world_to_grid(row, col)
        grid[
            start[0]:start[0] + self.sizes["cell"],
            start[1]:start[1] + self.sizes["cell"]
        ] = color

    def visualise_as_grid(
        self,
        visualise_traj=True,
        show=True,
        save_path=None,
        ):
        """
        Shows and/or saves the grid
        """

        # Drawing background
        grid_size = self.size * (self.sizes["cell"] + self.sizes["wall"])
        grid = 255 * np.ones([grid_size, grid_size, 3])
        
        #   Drawing walls
        for i in range(self.size+1):
            start = i * (self.sizes["cell"] + self.sizes["wall"])
            grid[start:start + self.sizes["wall"], :] = (0, 0, 0)
        
        for j in range(self.size+1):
            start = j * (self.sizes["cell"] + self.sizes["wall"])
            grid[:, start:start + self.sizes["wall"]] = (0, 0, 0)
        
        # Drawing cells' content
        #   Drawing potential goals
        for distribution in self.distributions:
            for potential in distribution.potential_area:
                self._draw_cell_in_grid(grid, *potential, distribution.area_color)
        
        #   Drawing sampled goals
        for reward in self.reward_coords:
            self._draw_cell_in_grid(grid, *reward, self.colors["goal"])
        
        #   Drawing agent
        self._draw_cell_in_grid(grid, *self.init_pos, self.colors["agent_start"])
        self._draw_cell_in_grid(grid, *self.current_pos, self.colors["agent_current"])

        #   Drawing trajectory
        if visualise_traj is True:
            for cell in self.state_hist:
                self._draw_cell_in_grid(grid, *cell, self.colors["trajectory"])
        
        # Clipping grid data to fit in 0..1 range
        grid /= 255

        if show is True:
            plt.imshow(grid)
            plt.show()

        if save_path is not None:
            plt.imsave(save_path, grid)

        return grid


if __name__ == "__main__":
    g = GridWorld(**GridWorldSparse40x40Mixed, is_guessing_game=True)    
    g.visualise_as_grid()

    g = GridWorld(**GridWorld40x40Circles, is_guessing_game=True)
    g.visualise_as_grid()
