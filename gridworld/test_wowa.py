
import matplotlib.pyplot as plt

from class_quality_diversity import QualityDiversity
from class_meta_learning_explore import MetaLearningExplore

from environment.class_gridworld import GridWorld
from environment.utils_worlds import GridWorld19x19TestWOWA
from environment.class_gridagent import GridAgentGuesser

from utils_explore import interpolate_weights


### COMMON PARAMETERS
traj_length = 10

w_to_test = [
    ((.7, .16, .14), "uniform"),
    ((.14, .16, .7), "fast"),
    ((.9, .08, .02), "slow"),
]

p_to_test = [
    ((1,), "anywhere"),
    ((1, .75, .25, 0), "early"),
    ((0, .25, .75, 1), "late"),
]

explore_kwargs = {
    "nb_instances":5,
    "nb_generations_outer":10, "nb_generations_inner":10,
    "population_size_outer":20, "population_size_inner":20,

    "selection_weights":(1,1),

    "inner_algorithm":QualityDiversity,

    "environment":{
        "type":GridWorld,
        "parameters":{
            "is_guessing_game":True,
            **GridWorld19x19TestWOWA
        }
    },
}
###

### CHANGE THE MAP'S EVOLVABILITY
class GridAgentGuesserNewEvolve(GridAgentGuesser):
    def __init__(self, init_position=None, min_mutation_amp=1, max_mutation_amp=5, grid_size=40):
        super().__init__(init_position, min_mutation_amp, max_mutation_amp, grid_size)
    
    def _make_amplitude(self, input_position=None):

        if input_position is None:
            input_position = self.action[:]

        # HORIZONTAL
        # amplitude = input_position[0]//10 + 1
        # VERTICAL
        # amplitude = input_position[1]//10 + 1
        # DIAGONAL
        # amplitude = sum(input_position) // 10 + 1
        # BAND
        spos = sum(input_position)
        amplitude = (spos // 5 + 1) if spos <= 20 else (4 - (spos-20) // 5)

        return amplitude
###

### RUN EXAMPLES
base_size = 3
for p, plabel in p_to_test:

    fig_world, axs = plt.subplots(nrows=3, ncols=3, figsize=(3*base_size,3*base_size))

    fig_world.tight_layout()

    for i, (w, wlabel) in enumerate(w_to_test):
        
        print(p, w, end='\r')

        faery = MetaLearningExplore(

            traj_length=traj_length,
            w=interpolate_weights(w, traj_length),
            p=interpolate_weights(p, traj_length),

            # Outer mutations
            mutation_prob=1,

            ag_type=GridAgentGuesserNewEvolve,
            toolbox_kwargs_inner={
                "stop_when_solution_found":True,
                "max_steps_after_found":0,
                "mutation_prob":1,
            },

            # Useless args
            offspring_size_outer=explore_kwargs["population_size_outer"],
            offspring_size_inner=explore_kwargs["population_size_inner"],

            creator_parameters={
                "individual_name":"MetaIndividual",
                "fitness_name":"MetaFitness",
            },
            ###

            **explore_kwargs
        )

        pop, log, hof = faery(verbose=False)

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

        # axs[i][0].imshow(faery.probe_evolvability())
        axs[i][0].imshow(grid_hist)
        axs[i][1].imshow(grid_final)

        axs[i][0].set_title("History w:{} p:{}".format(wlabel, plabel))
        axs[i][1].set_title("Last population")

        faery.show_history(show=False, axs=[axs[i][2]], index_to_show=[0])

fig, ax = plt.subplots()
ax.imshow(faery.probe_evolvability())
plt.show()
