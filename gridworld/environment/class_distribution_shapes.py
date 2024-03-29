import cv2
import numpy as np

from environment.class_distribution import SequentialDistribution, UniformDistribution


class UniformCircular(UniformDistribution):
    """
    A distribution of circular shape
    """

    def __init__(self, center, radius):
        super().__init__()

        self.center = center
        self.radius = radius

    def _make_me(self, world):

        world_tmp = world.copy()
        world_new = cv2.circle(
            img=world,
            center=self.center[::-1],
            radius=self.radius,
            color=self.area_color,
            thickness=-1
        )

        shape = (world_new - world_tmp).nonzero()[:2]
        return list(map(tuple, list(np.concatenate([np.array([c]).T for c in shape], axis=1))))


class UniformRing(UniformDistribution):
    """
    A distribution of ring shape
    """

    def __init__(self, center, inner_radius, outer_radius):
        super().__init__()

        self.center = center
        
        assert inner_radius < outer_radius, "Radius are inverted"

        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def _make_me(self, world):

        world_tmp = world.copy()
        world_new = cv2.circle(
            img=world,
            center=self.center[::-1],
            radius=self.outer_radius,
            color=self.area_color,
            thickness=self.outer_radius - self.inner_radius
        )

        shape = (world_new - world_tmp).nonzero()[:2]
        return list(map(tuple, list(np.concatenate([np.array([c]).T for c in shape], axis=1))))


class UniformRectangle(UniformDistribution):
    """
    A distribution of rectangular shape

    start : upper left corner
    width, height
    """
    
    def __init__(self, start, width, height):
        super().__init__()

        self.start = start
        self.width = width
        self.height = height

    def _make_me(self, world):

        world_new = world.copy()

        potential_area = []
        for i in range(min(self.height, world_new.shape[0] - self.start[0])):
            for j in range(min(self.width, world_new.shape[1] - self.start[1])):
                c_i, c_j = self.start[0] + i, self.start[1] + j
                potential_area.append((c_i,c_j))
                
        return potential_area


class UniformHorizontalStripes(UniformDistribution):
    """
    A distribution of horizontal stripes

    start : upper left corner
    width : width of each stripe
    nb_stripes : number of stripes
    space : space between stripes
    """

    def __init__(self, start, width, nb_stripes, space):
        super().__init__()

        for i in range(nb_stripes):
            self.objects.append(
                UniformRectangle(
                    (start[0] + i * (space + 1), start[1]),
                    width, 1
                )
            )


class SequentialPoints(SequentialDistribution):
    """
    A sequential distribution of points
    """

    def __init__(self, list_points):
        super().__init__()
        
        for point in list_points:
            self.objects.append(
                UniformRectangle(point, 1, 1)
            )
