import bisect
import matplotlib.pyplot as plt

from copy import deepcopy
from class_quality_diversity import QualityDiversity
from class_meta_learning_faery import MetaLearningFAERY

from environment.class_gridagent import GridAgentGuesser
from environment.class_gridworld import GridWorld
from environment.utils_worlds import GridWorld19x19TestWOWA


class GridAgentFixedMutation(GridAgentGuesser):
    """
    Agent with fixed evolvability on 3 cells
    """

    checkpoints = [0, 6, 12]

    mutation_dist = {
        checkpoints[0]:(6, 6, 6),
        checkpoints[1]:(9, 6, 3),
        checkpoints[2]:(3, 6, 9)
    }

    def __init__(self, init_position, min_mutation_amp=1, max_mutation_amp=5, grid_size=40):
        super().__init__(init_position, min_mutation_amp, max_mutation_amp, grid_size)

        self.nb_mutant = 0
    
    def mutate(self):
        """
        Fixed evolvability on 3 cells
        """

        if self.nb_mutant >= len(list(self.mutation_dist.values())[0]):
            self.nb_mutant -= 1

        super().mutate(
            mutation="DOWN",
            amplitude=self.mutation_dist[
                self.checkpoints[max(bisect.bisect_left(self.checkpoints, self.action[1])-1, 0)]
            ][self.nb_mutant]
        )
        
        self.nb_mutant += 1

if __name__ == "__main__":

# MANUAL

    # env = GridWorld(**GridWorld19x19TestWOWA)

    # agents = [
    #     GridAgentFixedMutation(init_position=env.reset(), grid_size=env.size)
    #     for _ in range(19)
    # ]

    # mutants = []
    # for agent in agents:
    #     for _ in range(3):
    #         mutant = deepcopy(agent if _ == 0 else mutants[-1])
    #         mutant.mutate()
    #         mutants.append(mutant)
    
    # agents += mutants

    # for agent in mutants:
    #     env(agent, 1)

    # env.visualise_as_grid(
    #     list_state_hist=[[agent.behavior] for agent in agents],
    #     show_traj=True,
    #     show_start=False,
    #     show_end=True,
    #     show=True
    # )

# QD

    # ns = QualityDiversity(
    #     nb_generations=3,
    #     population_size=19,
    #     offspring_size=None,

    #     mutation_prob=1,

    #     max_steps_after_found=0,

    #     ag_type=GridAgentFixedMutation,
    #     environment=GridWorld(**GridWorld19x19TestWOWA)
    # )

    # pop, log, hof = ns(show_history=False, verbose=True)

    # ns.environment.visualise_as_grid(
    #     list_state_hist=[[behavior] for behavior in ns.archive.behaviors],
    #     show_traj=True,
    #     show_start=False,
    #     show_end=True,
    #     show=True
    # )

# EXPLORE
    pass