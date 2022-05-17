import torch
import matplotlib.pyplot as plt

from deap.tools import mutPolynomialBounded

from environment.class_gridworld import GridWorld
from environment.utils_worlds import GridWorldSparse40x40Mixed
from environment.class_distribution_shapes import UniformRectangle
from environment.class_gridagent import GridAgentGuesser, GridAgentNN

g1 = GridWorld(**GridWorldSparse40x40Mixed, is_guessing_game=True)

GridWorldSparse40x40Mixed["start_distribution"] = UniformRectangle((18,18), 3, 3)

 
nn = GridAgentNN(
    input_dim=2, 
    output_dim=2,
    hidden_layers=3, 
    hidden_dim=20,
    use_batchnorm=False,
    non_linearity=torch.tanh, 
    output_normalizer=lambda x: torch.round((40 - 1) * abs(x)).int(),
)

fig, ax = plt.subplots(nrows=2, ncols=2)
pos = iter([(0,0), (0,1), (1,0), (1,1)])

for _ in range(4):
    mutant = mutPolynomialBounded(
        nn,
        eta=15,
        low=-1,
        up=1,
        indpb=.3
    )[0]
    
    g1.reset()
    ax[next(pos)].imshow(
        g1.visualise_as_grid(
            list_state_hist=[[mutant((19,19))] for _ in range(100)],
            show_traj=True,
            show_start=False,
            show_end=True,
            show=False
        )
    )

plt.show()