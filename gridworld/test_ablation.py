import matplotlib.pyplot as plt

from class_quality_diversity import QualityDiversity
from class_meta_learning_faery import MetaLearningFAERY
from environment.class_gridagent import GridAgentNN, GridAgentGuesser


objectives = (1,1)

# Running
faery = MetaLearningFAERY(
    nb_instances=25,

    nb_generations_outer=70,
    population_size_outer=25, offspring_size_outer=25,

    inner_algorithm=QualityDiversity,
    nb_generations_inner=20,
    population_size_inner=25, offspring_size_inner=25,

    selection_weights=objectives,

    ag_type=GridAgentGuesser,

    creator_parameters={
        "individual_name":"NNIndividual",
        "fitness_name":"NNFitness",
    },

    toolbox_kwargs_inner={
        "stop_when_solution_found":True,
        "mutation_prob":.3,
    },

    mutation_prob=1
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

fig, axs = plt.subplots(ncols=2, figsize=(16,8))
axs[0].imshow(grid_hist)
axs[1].imshow(grid_final)

plt.savefig("ablation_" + str(objectives))
