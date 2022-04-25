from class_distribution_shapes import *

# File containing dict of a world's size and list of distributions

GridWorldSparse40x40Mixed = {
    "size":40,
    "distributions":[
        UniformHorizontalStripes((27,0), 40, 7, 1),
        UniformCircular((6, 31), 3),
        UniformCircular((13, 5), 3),
    ],
    "start_distribution":UniformCircular((1,1), 0)
}

GridWorld40x40Circles = {
    "size":40,
    "distributions":[
        UniformRing((19, 19), radius, radius+1)
        for radius in range(1,31,4)
    ],
    "start_distribution":UniformCircular((19,19), 0)
}