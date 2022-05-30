from itertools import product

from environment.class_distribution_shapes import *

# File containing dict of a world's size and list of distributions

GridWorldSparse40x40Mixed = {
    "size":40,
    "distributions":[
        UniformHorizontalStripes((27,0), 40, 7, 1),
        UniformCircular((6, 31), 3),
        UniformCircular((13, 5), 3),
    ],
    "start_distribution":UniformHorizontalStripes((0,0), 40, 1, 0)
}

GridWorldSparse40x40MixedCut = {
    "size":40,
    "distributions":[
        UniformHorizontalStripes((27,0), 40, 7, 1),
        UniformCircular((6, 31), 3),
        UniformCircular((13, 5), 3),
    ],
    "start_distribution":UniformCircular((1,1), 0),
    "walls":list(product(range(25), range(18,21))) + \
        list(product(range(22,25), range(14)))
}

GridWorld40x40Circles = {
    "size":40,
    "distributions":[
        UniformRing((19, 19), radius, radius+1)
        for radius in range(1,31,4)
    ],
    "start_distribution":UniformCircular((19,19), 0)
}

GridWorld19x19TestWOWA = {
    "size":20,
    "distributions":[
    ],
    "start_distribution":SequentialPoints([(0,k) for k in range(20)])
}