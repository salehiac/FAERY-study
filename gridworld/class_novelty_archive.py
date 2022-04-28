from math import dist
from scipy.spatial import KDTree

class NoveltyArchive:
    """
    Archive used to compute novelty scores
    """

    def __init__(
        self,
        init_behaviors=[],
        neighbouring_size=15,
        max_size=None,
        distance_function = dist
        ):
        """
        init_behaviors : behaviors inside the archive at initialization
        neighbouring_size : k-th nearest neighbours to compute the novelty with
        max_size : maximum size of the archive
        distance_function : the distance function to use between two points
        """

        self.behaviors = init_behaviors[:]
        self.kdTree = None
        self.neigh_size = neighbouring_size
        self.max_size = max_size

        self.dist = distance_function
    
    def update(self, new_behavior):
        """
        Update the archive with a new behavior
        """

        self.behaviors += [new_behavior]
        if self.max_size is not None:
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