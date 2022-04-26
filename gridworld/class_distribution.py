import random

from abc import ABC, abstractmethod


class Distribution(ABC):
    """
    A group of shapes, the pixels of which can be randomly sampled as goals for GridWorld.
    Overlap is counted as one.
    """

    id = 0

    area_color = (0, 255, 0)

    def __init__(self):
        
        Distribution.id += 1
        self.id = Distribution.id

        self.objects = []
        self.potential_area = []
        self.size = 0
    
    def _make_me(self, world):
        """
        Returns potential area as list of tuples
        """

        raise NotImplementedError("Overwrite method if object doesn't contain sub-distributions")
    
    def _make_shape(self, world):
        """
        Returns potential area as list of tuple from children
        """

        self.potential_area = []
        if len(self.objects) != 0:
            for obj in self.objects:
                self.potential_area += obj._make_shape(world)
        else:
            self.potential_area = self._make_me(world)
        
        self.size = len(self.potential_area)
        return self.potential_area
    
    @abstractmethod
    def sample(self):
        """
        Returns a sampled pixel
        """

        return NotImplemented


class UniformDistribution(Distribution):
    """
    """

    def __init__(self):
        super().__init__()
    
    def sample(self):
        """
        Returns a sampled pixel (uniform probability)
        """

        return self.potential_area[random.randint(0, self.size-1)]
