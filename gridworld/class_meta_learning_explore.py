import networkx

try:
    from math import dist
except ImportError:
    import math
    dist = lambda x, y: math.sqrt(sum([(x[i]-y[i])**2 for i in range(len(x))]))

import numpy as np
import matplotlib.pyplot as plt

from class_meta_learning_faery import MetaLearningFAERY
from utils_explore import Capacity, WOWA


class MetaLearningExplore(MetaLearningFAERY):
    """
    Implementation of explore variant of FAERY,
        F0 is replaced with a WOWA on the distance travelled by a meta-individual's offspring
    """

    def __init__(self, *args, w, p, traj_length=3, should_show_evo_tree=False, **kwargs):
        """
        w : weights for OWA, caracterizes the imbalance
        p : weights for WOWA, localizes the imbalance
        traj_length : length of the paths to consider (zero-padded if too short)
        """

        super().__init__(*args, should_show_evo_tree=should_show_evo_tree, **kwargs)

        self.w = w
        self.p = p
        self.phi = Capacity(w)

        self.traj_length = traj_length
    
    def _get_path(self, graph, node, max_length):
        """
        Returns all paths  of length max_length starting from node
        """

        if max_length == 0 or len(list(graph.neighbors(node))) == 0:
            return [[node]]

        return [
            [node] + path
            for neighbor in graph.neighbors(node)
            for path in self._get_path(graph, neighbor, max_length-1) if node not in path
        ]

    def _get_travelled_path(self, population):
        """
        Returns the average movement along an individual's path for the whole population
        """

        # Average movement of each individual
        avg_mov = [
            [None for _ in range(self.traj_length)]
            for __ in range(len(population))
        ]

        # Graphs of all instances
        all_graphs = [
            networkx.DiGraph(instance.history.genealogy_tree).reverse()
            for instance in self.instances
        ]

        # Computing the avg travel for each individual
        max_depth = [-1 for _ in range(len(population))]
        for i in range(len(population)):
            avg_mov[i] = [0 for _ in range(self.traj_length)]
            nb_at_each_depth = [0 for k in range(self.traj_length)]

            # Iterating over all instances
            for g, graph in enumerate(all_graphs):
                all_successors = self._get_path(graph, i+1, self.traj_length)

                # Going down in depth, averaging for current instance
                for path in all_successors:
                    for k in range(len(path)-1):

                        start_behavior = self.instances[g].history.genealogy_history[path[k]].behavior
                        end_behavior = self.instances[g].history.genealogy_history[path[k+1]].behavior

                        avg_mov[i][k] += dist(start_behavior, end_behavior)
                        nb_at_each_depth[k] += 1
                    
                    max_depth[i] = max(max_depth[i], -1 if len(path) <= 1 else k)

            # Averaging over all instances
            if max_depth[i] != -1:
                for k in range(max_depth[i]):
                    avg_mov[i][k] /= nb_at_each_depth[k]

        # Computing average acceleration and average distance
        avg_dist = np.mean(avg_mov)

        avg_accel = [
            sum([
                avg_mov[j][l+1] - avg_mov[j][l]
                for j in range(len(population))
            ]) / len(population)
            for l in range(self.traj_length-1)
        ]

        # Completing the path with average acceleration if not enough depth
        for i, depth in enumerate(max_depth):
            if depth == -1:
                avg_mov[i][0] = avg_dist
                for j in range(self.traj_length-1):
                    avg_mov[i][j+1] = avg_mov[i][j] + avg_accel[j]

            elif depth < self.traj_length:
                for j in range(depth + 1, self.traj_length):
                    avg_mov[i][j] = avg_mov[i][j-1] + avg_accel[j-1]

        return avg_mov
    
    def _update_scores(self, population):
        """
        Computes and updates the individuals' scores according to FAERY Explore
        """

        # For all instances, backtrack the solvers to root and retrieve their depth
        agent_to_depth = self._get_agent_to_depth(population)
        # Computing average path for all individual
        avg_movement = self._get_travelled_path(population)
        
        # Updating the scores
        for i, agent in enumerate(population):
            nb_solved, depth = agent_to_depth[agent][0], agent_to_depth[agent][1]
            agent.fitness.values = (
                WOWA(self.w, self.p, avg_movement[i], self.phi),
                depth / nb_solved if nb_solved != 0 else -1 * self.selection_weights[1] * float("inf")
            )


if __name__ == "__main__":

    from environment.class_gridagent import GridAgentNN, GridAgentGuesser
    from class_quality_diversity import QualityDiversity

    faery = MetaLearningExplore(
        nb_instances=25,

        nb_generations_outer=70,
        population_size_outer=25, offspring_size_outer=25,

        inner_algorithm=QualityDiversity,
        nb_generations_inner=20,
        population_size_inner=25, offspring_size_inner=25,

        selection_weights=(1,1),

        w = (.05, .25, .7),
        p = (4/6, 3/12, 1/12),

        ag_type=GridAgentGuesser,

        creator_parameters={
            "individual_name":"NNIndividual",
            "fitness_name":"NNFitness",
        },

        toolbox_kwargs_inner={
            "stop_when_solution_found":True,
            "max_steps_after_found":0,
            "mutation_prob":.3,
        },

        mutation_prob=1
    )

    faery.should_show_evo_tree = False
    pop, log, hof = faery(show_history=True)

    # print(faery.inner_logbook)

    # Showing the meta-population history
    #   We have to run them on an environment first to generate their behaviors
    for ag in faery.history.genealogy_history.values():
        faery.model_environment(ag, 1)
    
    #   We don't need to show the sampled rewards
    faery.model_environment.reward_coords = []

    grid_hist = faery.model_environment.visualise_as_grid(
        list_state_hist=[[ag.behavior] for ag in faery.history.genealogy_history.values()],
        show_traj=True,
        show_start=False,
        show_end=True,
        show=False
    )

    faery.model_environment.reset()
    for ag in faery.population:
        faery.model_environment(ag, 1)
    
    faery.model_environment.reward_coords = []

    grid_final = faery.model_environment.visualise_as_grid(
        list_state_hist=[[ag.behavior] for ag in faery.population],
        show_traj=True,
        show_start=False,
        show_end=True,
        show=False
    )

    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(grid_hist)
    axs[1].imshow(grid_final)

    plt.show()
