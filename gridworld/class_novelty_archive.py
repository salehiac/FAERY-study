from math import dist
from scipy.spatial import KDTree


class NoveltyArchive:
    """
    Archive used to compute novelty scores
    """

    def __init__(
        self,
        init_agents=[],
        neighbouring_size=15,
        max_size=None,
        distance_function = dist
        ):
        """
        init_agents : behaviors inside the archive at initialization
        neighbouring_size : k-th nearest neighbours to compute the novelty with
        max_size : maximum size of the archive
        distance_function : the distance function to use between two points
        """

        self.all_agents = init_agents[:]
        self.behaviors = [ag.behavior for ag in init_agents]

        self.kdTree = None
        self.dist = distance_function
        self.neigh_size = neighbouring_size

        self.max_size = max_size        
    
    def update(self, agent):
        """
        Update the archive with a new behavior
        """
        
        # MAYBE ONLY KEEP THE BEST INDIVIDUALS, NEED TO RECOMPUTE NOVELTY
        if agent not in self.all_agents:
            self.all_agents += [agent]
            self.behaviors += [agent.behavior]

            if self.max_size is not None:
                self.all_agents = self.all_agents[-self.max_size:]
                self.behaviors = self.behaviors[-self.max_size:]
                
            self.kdTree = KDTree(self.behaviors)
    
    def get_novelty(self, behavior):
        """
        Gets the novelty of a behavior compared to the archive
        """

        return [sum(self.kdTree.query(behavior, k=self.neigh_size)[0])]

    def get_size(self):
        """
        Return the size of the archive
        """

        return len(self.behaviors)
    
    def reset(self, init_agents=[]):
        """
        Resets the archive
        """

        self.all_agents = init_agents[:]
        self.behaviors = [ag.behavior for ag in init_agents]

        self.kdTree = None