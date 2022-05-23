import matplotlib.pyplot as plt

from class_meta_learning_explore import MetaLearningExplore
from environment.class_gridagent import GridAgentGuesser
from class_quality_diversity import QualityDiversity


w_p_to_test = (
    [.5, .4, .1],
    [.5, .4, .1]
) # BALANCED

explore = MetaLearningExplore(
    nb_instances=10,

    nb_generations_outer=20,
    population_size_outer=25, offspring_size_outer=25,

    inner_algorithm=QualityDiversity,
    nb_generations_inner=20,
    population_size_inner=25, offspring_size_inner=25,

    selection_weights=(1,1),

    w=w_p_to_test[0],
    p=w_p_to_test[1],

    ag_type=GridAgentGuesser,

    creator_parameters={
        "individual_name":"NNIndividual",
        "fitness_name":"NNFitness",
    },

    toolbox_kwargs_inner={
        "stop_when_solution_found":True,
        "max_steps_after_found":0,
        "mutation_prob":1,
    },

    mutation_prob=1
)

explore.should_show_evo_tree = False
pop, log, hof = explore(show_history=True)

# Showing the meta-population history
#   We have to run them on an environment first to generate their behaviors
for ag in explore.history.genealogy_history.values():
    explore.model_environment(ag, 1)

#   We don't need to show the sampled rewards
explore.model_environment.reward_coords = []

grid_hist = explore.model_environment.visualise_as_grid(
    list_state_hist=[[ag.behavior] for ag in explore.history.genealogy_history.values()],
    show_traj=True,
    show_start=False,
    show_end=True,
    show=False
)

explore.model_environment.reset()
for ag in explore.population:
    explore.model_environment(ag, 1)

explore.model_environment.reward_coords = []

grid_final = explore.model_environment.visualise_as_grid(
    list_state_hist=[[ag.behavior] for ag in explore.population],
    show_traj=True,
    show_start=False,
    show_end=True,
    show=False
)

fig, axs = plt.subplots(ncols=2)
axs[0].imshow(grid_hist)
axs[1].imshow(grid_final)

print(log)

plt.show()