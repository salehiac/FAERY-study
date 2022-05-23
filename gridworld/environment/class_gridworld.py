import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from itertools import product

from environment.class_reward_function import RewardBinary
from environment.utils_worlds import *


class GridWorld:
    """
    GridWorld environment generated from a given list of distributions
    """

    colors = {
        "goal":(0, 0, 255),
        "agent_start":(255, 0, 0),
        "agent_current":(255, 0, 0),
        "trajectory":(255, 180, 0),
    }

    sizes = {
        "cell":12,
        "wall":1,
        "highlight":1,
    }

    max_sample_failure = 1000

    def __init__(
        self,
        size,
        start_distribution,
        distributions,
        goal_type="weighted 5",
        reward_function=RewardBinary(),
        is_guessing_game=False,
        walls=[],
        ):
        """
        is_guessing_game : agent will only guess a single cell
        goal_type : 
                      "uniform N" uniformly sample a reward from all goals, N times;
                      "mix N" randomly choose a distribution, then sample it once, N times;
                      "concentrated N" randomly choose a distribution, then sample it N times;
                      "weigthed N" randomly choose a distribution (according to their total mass), then sample it N times;
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

        self.walls = set(walls)
        for wall in self.walls:
            for distribution in self.distributions:
                try:
                    distribution.potential_area.remove(wall)
                    distribution.size -= 1
                    self.potential_goal_areas.remove(wall)
                except ValueError:
                    pass
            
            try:
                self.start_distribution.potential_area.remove(wall)
            except ValueError:
                pass

        self.empty_cells = set(product(range(self.size), range(self.size))) - self.walls

        self.current_pos = None
        self.state_hist = []

        self.reward_coords = []

    def reset(self, change_goal=True):
        """
        Resets the environment and defines new goals aswell as a new starting position
        """

        if change_goal is True and len(self.distributions) > 0:
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
            
            elif self.goal_type == "concentrated":
                i, failed = 0, 0

                distribution = random.choice(self.distributions)
                while i < self.nb_samples:
                    new_goal = tuple(distribution.sample())

                    if new_goal in self.reward_coords:
                        failed += 1
                        if failed >= self.max_sample_failure:
                            raise ValueError("Failed to sample, shapes might be too small")
                        continue
                    
                    self.reward_coords.append(new_goal)
                    i += 1
            
            elif self.goal_type == "weighted":
                i, failed = 0, 0

                distribution = random.choices(self.distributions, weights=[d.size for d in self.distributions])[0]
                while i < self.nb_samples:
                    new_goal = tuple(distribution.sample())

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

        try:
            self.init_pos = self.start_distribution.sample()
        except IndexError:
            raise IndexError("Starting position destroyed by walls (or of area 0)")

        self.state_hist = [self.init_pos[:]] if self.is_guessing_game is False else []
        
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
            
            previous_pos = self.current_pos

            if action == 0:
                self.current_pos = tuple([
                    self.current_pos[0],
                    max(0, self.current_pos[1] - 1)
                ])

            elif action == 1:
                self.current_pos = tuple([
                    self.current_pos[0],
                    min(self.size - 1, self.current_pos[1] + 1)
                ])

            elif action == 2:
                self.current_pos = tuple([
                    max(0, self.current_pos[0] - 1),
                    self.current_pos[1]
                ])

            elif action == 3:
                self.current_pos = tuple([
                    min(self.size - 1, self.current_pos[0] + 1),
                    self.current_pos[1]
                ])
            
            # Can be faster if no min and max above, juste check on available cells
            if self.current_pos in self.walls:
                self.current_pos = previous_pos
            
            done = self.current_pos in self.reward_coords

        else:

            if action not in self.empty_cells:
                print(action)
                raise ValueError("Guess taken in a wall")

            self.current_pos = action
            done = self.current_pos in self.reward_coords

        self.state_hist.append(self.current_pos)
        return self.current_pos, self.reward_function(self), done

    def __call__(self, ag, nb_steps):
        """
        Runs the environment on a given agent
        """

        self.reset(change_goal=False)

        if hasattr(ag, "eval"):  # in case of torch agent
            ag.eval()
        
        if self.is_guessing_game is True:
            nb_steps = 1

        with torch.no_grad():
            fitness = 0
            obs = self.current_pos

            for _ in range(0, nb_steps):
                action = ag(obs)
                obs, reward, done = self.step(action)
                fitness += reward

                if done is True:
                    break
        
        ag.update_behavior(self.state_hist)
        return fitness, done, self.state_hist

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
    
    def _draw_highlight(self, grid, row, col, color):
        """
        Highlights a cell with given color
        """

        start = self._world_to_grid(row, col)

        cell_size = self.sizes["cell"]
        margin = self.sizes["highlight"]

        # Left side
        grid[
            start[0] - margin:start[0],
            start[1] - margin:start[1] + cell_size + margin
        ] = color

        # Right side
        grid[
            start[0] + cell_size:start[0] + cell_size + margin,
            start[1] - margin:start[1] + cell_size + margin
        ] = color

        # Up side
        grid[
            start[0] - margin:start[0] + cell_size + margin,
            start[1] - margin:start[1]
        ] = color

        # Down side
        grid[
            start[0] - margin:start[0] + cell_size + margin,
            start[1] + cell_size:start[1] + cell_size + margin
        ] = color

    def visualise_as_grid(
        self,

        list_state_hist=[],

        show_traj=False,
        show_start=True,
        show_end=True,

        show=False,
        save_path=None,

        highlight_goals=True
        ):
        """
        Shows and/or saves the grid, can be modified with given trajectories
        """

        if len(list_state_hist) == 0:
            list_state_hist = [self.state_hist]

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
        
        for wall in self.walls:
            self._draw_cell_in_grid(grid, *wall, (0, 0, 0))
        
        # Drawing cells' content
        #   Drawing potential goals
        for distribution in self.distributions:
            for potential in distribution.potential_area:
                self._draw_cell_in_grid(grid, *potential, distribution.area_color)
        
        #   Drawing sampled goals
        for reward in self.reward_coords:
            self._draw_cell_in_grid(grid, *reward, self.colors["goal"])
        
        #   Drawing agents
        for state_hist in list_state_hist:
            #       Drawing trajectory
            if show_traj is True:
                for cell in state_hist:
                    self._draw_cell_in_grid(grid, *cell, self.colors["trajectory"])
        
        #   Two separate loops to avoid erasing
        for state_hist in list_state_hist:  
            #       Drawing positions
            if show_start is True:
                self._draw_cell_in_grid(grid, *state_hist[0], self.colors["agent_start"])

            if show_end is True:
                self._draw_cell_in_grid(grid, *state_hist[-1], self.colors["agent_current"])
        
        # Highlighting the goals if asked, helps if goal is overwritten by agent
        if highlight_goals is True:
            for reward in self.reward_coords:
                self._draw_highlight(grid, *reward, self.colors["goal"])
 
        # Clipping grid data to fit in 0..1 range
        grid /= 255

        if show is True:
            plt.imshow(grid)
            plt.show()

        if save_path is not None:
            plt.imsave(save_path, grid)

        return grid


if __name__ == "__main__":

    GridWorldSparse40x40Mixed["start_distribution"] = UniformRectangle((0,0), 40, 40)
    g = GridWorld(**GridWorldSparse40x40Mixed, is_guessing_game=False)    

    for k in range(100):
        position, reward, done = g.step(random.randint(0,3))
        if done is True:
            print("DONE")
            break
    
    g.visualise_as_grid(
        show_traj=True,
        show=True
    )