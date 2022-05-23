from environment.class_gridagent import GridAgentGuesser
from environment.class_gridworld import GridWorld
from environment.utils_worlds import GridWorld12x12TestWOWA


class GridAgentFixedMutation(GridAgentGuesser):
    """
    Agent with fixed evolvability on 3 cells
    """

    mutation_dist = [
        (4, 4, 4),
        (7, 4, 1),
        (1, 4, 7)
    ]

    def __init__(self, init_position, min_mutation_amp=1, max_mutation_amp=5, grid_size=40):
        super().__init__(init_position, min_mutation_amp, max_mutation_amp, grid_size)

        self.nb_mutant = 0
    
    def mutate(self):
        """
        Fixed evolvability on 3 cells
        """

        super().mutate(
            mutation="DOWN",
            amplitude=self.mutation_dist[self.action[1]][self.nb_mutant]
        )
        
        self.nb_mutant += 1


from class_quality_diversity import QualityDiversity

ns = QualityDiversity(
    nb_generations=1,
    population_size=3,
    offspring_size=3,

    max_steps_after_found=0,

    ag_type=GridAgentFixedMutation,
    environment=GridWorld(**GridWorld12x12TestWOWA)
)

pop, log, hof = ns(show_history=False, verbose=True)

ns.environment.visualise_as_grid(
    list_state_hist=[[behavior] for behavior in ns.archive.behaviors],
    show_traj=True,
    show_start=False,
    show_end=True,
    show=True
)