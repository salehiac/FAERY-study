from class_quality_diversity import QualityDiversity


class NoveltySearch(QualityDiversity):
    """
    Simple novelty search class
    """

    def __init__(self, *args, selection_weights=(1, 0), **kwargs):
        super().__init__(*args, selection_weights=selection_weights, **kwargs)


if __name__ == "__main__":

    from environment.class_gridagent import GridAgentNN, GridAgentGuesser

    compute_NN = True
    compute_Guesser = False

    # For NN agents
    if compute_NN is True:

        import torch
        from deap import tools

        ns = NoveltySearch(
            nb_generations=20,
            population_size=10,
            offspring_size=10,

            ag_type=GridAgentGuesser,

            creator_parameters={
                "individual_name":"NNIndividual",
                "fitness_name":"NNFitness",
             },
        )

    # For guesser agents (those in the article)
    if compute_Guesser is True:

        ns = NoveltySearch(
            nb_generations=20,
            population_size=10,
            offspring_size=10,

            ag_type=GridAgentGuesser,
        )

    pop, log, hof = ns(show_history=True)
    print(log)

    # Novelty over the all archive
    ns.environment.visualise_as_grid(
        list_state_hist=[[behavior] for behavior in ns.archive.behaviors],
        show_traj=True,
        show_start=False,
        show_end=True,
        show=True
    )
