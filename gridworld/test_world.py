import matplotlib.pyplot as plt

from environment.class_gridworld import GridWorld
from environment.utils_worlds import GridWorld40x40Circles, GridWorldSparse40x40Mixed, GridWorldSparse40x40MixedCut


gs = [
    GridWorld(**GridWorldSparse40x40Mixed),
    GridWorld(**GridWorldSparse40x40MixedCut),
    GridWorld(**GridWorld40x40Circles),
]

fig, axs = plt.subplots(ncols=3, figsize=(24,8))

for k, g in enumerate(gs):
    axs[k].imshow(g.visualise_as_grid())

plt.show()
