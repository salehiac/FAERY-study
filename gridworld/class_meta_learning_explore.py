import networkx

try:
    from math import dist
except ImportError:
    import math
    dist = lambda x, y: math.sqrt(sum([(x[i]-y[i])**2 for i in range(len(x))]))

import numpy as np

from class_meta_learning_faery import MetaLearningFAERY
from utils_explore import build_capacity, WOWA


class MetaLearningExplore(MetaLearningFAERY):
    """
    Implementation of explore variant of FAERY,
        F0 is replaced with a WOWA on the distance travelled by a meta-individual's offspring
    """

    def __init__(self, *args, w, p, ndigits=18, traj_length=3, should_show_evo_tree=False, **kwargs):
        """
        w : weights for OWA, caracterizes the imbalance
        p : weights for WOWA, localizes the imbalance
        ndigits : precision for WOWA
        traj_length : length of the paths to consider (zero-padded if too short)
        """

        super().__init__(*args, should_show_evo_tree=should_show_evo_tree, **kwargs)

        self.w = w
        self.p = p
        self.phi = build_capacity(w, p, ndigits=ndigits)

        self.traj_length = traj_length

    def _get_travelled_path(self, population):
        """
        Returns the average movement along an individual's path for the whole population
        """

        all_avg_paths = [[] for _ in range(len(population))]
        all_paths = [
            [
                self.traj_length * [0] for _ in range(len(self.instances))
            ] for __ in range(len(population))
        ]

        for i, instance in enumerate(self.instances):
            graph = networkx.DiGraph(instance.history.genealogy_tree).reverse()

            for k, ind in enumerate(population):
                node = ind.history_index
                
                paths = all_paths[k]
                for d, edge in enumerate(networkx.bfs_successors(graph, node, depth_limit=self.traj_length)):
                    start, ends = edge

                    start_behavior = instance.genealogy_history[start].behavior
                    paths[i][d] = sum([
                        dist(start_behavior, instance.genealogy_history[end].behavior)
                        for end in ends
                    ]) / len(ends)

                all_avg_paths[k].append(tuple(np.mean(paths, axis=0)))