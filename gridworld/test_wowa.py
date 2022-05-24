import bisect
from os import environ
import matplotlib.pyplot as plt

from copy import deepcopy
from class_novelty_archive import NoveltyArchive
from class_quality_diversity import QualityDiversity
from class_meta_learning_explore import MetaLearningExplore

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

    def __init__(self, init_position=None, min_mutation_amp=1, max_mutation_amp=5, grid_size=40):
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
    

class UselessAlgo(QualityDiversity):

    def __init__(
        self,
        *args,
        archive={
            "type": NoveltyArchive,
            "parameters": {
                "neighbouring_size": 2,
                "max_size": None
            }
        },
        **kwargs):

        super().__init__(*args, selection_weights=(1, 0), **kwargs)

    def _update_fitness(self, population):

        fitnesses = self.toolbox.map(self._evaluate, population)
        
        # Updating the archive
        for ag in population:
            next(fitnesses)
            self.archive.update(ag)

        # Computing the population's novelty
        if self.archive.get_size() >= self.archive.neigh_size:
            for ag in population:
                ag.fitness.values = (
                    ag.nb_mutant,
                    0
                )
            return True
        
        for ag in [ind for ind in population if ind.fitness.valid is False]:
            ag.fitness.values = (
                0,
                0
            )

        return False

        
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
    
    faery = MetaLearningExplore(
        nb_instances=5,

        nb_generations_outer=10,
        population_size_outer=19, offspring_size_outer=25,

        inner_algorithm=UselessAlgo,
        nb_generations_inner=4,
        population_size_inner=19, offspring_size_inner=25,

        selection_weights=(1,1),

        w = (1,0,0),
        p = (1/3, 1/3, 1/3),

        ag_type=GridAgentFixedMutation,

        creator_parameters={
            "individual_name":"NNIndividual",
            "fitness_name":"NNFitness",
        },

        toolbox_kwargs_inner={
            "stop_when_solution_found":True,
            "max_steps_after_found":0,
            "mutation_prob":1,
        },

        mutation_prob=1,
        environment={
            "type":GridWorld,
            "parameters":{
                "is_guessing_game":True,
                **GridWorld19x19TestWOWA
            }
        }
    )

    faery.should_show_evo_tree = False
    pop, log, hof = faery(show_history=True)

    # Showing the meta-population history
    #   We have to run them on an environment first to generate their behaviors
    for ag in faery.history.genealogy_history.values():
        faery.model_environment(ag, 1)
    
    #   We don't need to show the sampled rewards
    faery.model_environment.reward_coords = []

    grid_hist = faery.model_environment.visualise_as_grid(
        list_state_hist=[[ag.behavior] for ag in faery.history.genealogy_history.values()],
        show_traj=True,
        show_start=False,
        show_end=True,
        show=False
    )

    faery.model_environment.reset()
    for ag in faery.population:
        faery.model_environment(ag, 1)
    
    faery.model_environment.reward_coords = []

    grid_final = faery.model_environment.visualise_as_grid(
        list_state_hist=[[ag.behavior] for ag in faery.population],
        show_traj=True,
        show_start=False,
        show_end=True,
        show=False
    )

    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(grid_hist)
    axs[1].imshow(grid_final)

    plt.show()