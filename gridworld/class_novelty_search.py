from class_inner_algorithm import InnerAlgorithm
from class_novelty_archive import NoveltyArchive


class NoveltySearch(InnerAlgorithm):
    """
    Simple novelty search class
    """

    def __init__(
        self,
        
        *args,

        archive = {
            "type":NoveltyArchive,
            "parameters":{
                "neighbouring_size":15,
                "max_size":None
            }
        },

        **kwargs
        ):

        super().__init__(
            *args, **kwargs
        )

        self.archive = archive["type"](**archive["parameters"])
    
    def _update_fitness(self, population):
        """
        Evaluates a population of individuals and saves their novelty as fitness
        """

        # Weird call because of scoop
        fitnesses = self.toolbox.map(
            InnerAlgorithm._evaluate,
            [self for _ in range(len(population))],
            population
        )

        # Updating the archive
        for ag in population:
            next(fitnesses)
            self.archive.update(ag)
        
        # Computing the population's novelty
        if self.archive.get_size() >= self.archive.neigh_size:
            for ag in population:
                ag.fitness.values = self.archive.get_novelty(ag.behavior)
    
    def _compile_stats(self, population, logbook_kwargs={}):
        """
        Compile the stats on the whole archive
        """

        self.hall_of_fame.update(self.archive.all_agents)
        self.logbook.record(
            **logbook_kwargs,
            **self.statistics.compile(self.archive.all_agents)
        )


if __name__ == "__main__":

    import torch

    from class_gridworld import GridWorld
    from utils_worlds import GridWorldSparse40x40Mixed
    from class_gridagent import GridAgentNN, GridAgentGuesser

    compute_NN = False
    compute_Guesser = True

    # For NN agents
    if compute_NN is True:
        gw = GridWorld(**GridWorldSparse40x40Mixed, is_guessing_game=False)

        ns = NoveltySearch(
            environment=gw,
            nb_generations=100,
            population_size=10,
            offspring_size=10,
            meta_generation=0,

            generator={
                "function":lambda x, **kw: x(**kw),
                "parameters":{
                    "input_dim":2, 
                    "output_dim":2,
                    "hidden_layers":3, 
                    "hidden_dim":10,
                    "use_batchnorm":False,
                    "non_linearity":torch.tanh, 
                    "output_normalizer":lambda x: torch.round((gw.size - 1) * abs(x)).int(),
                    # Something is wrong in normalization
                }
            },
        )

        pop, log, hof = ns()

        gw.visualise_as_grid(
            list_state_hist=[ag.state_hist for ag in pop],
            show_start=False,
            show_end=True,
            show=True
        )

    # For guesser agents (those in the article)
    if compute_Guesser is True:
        gw = GridWorld(**GridWorldSparse40x40Mixed, is_guessing_game=True)

        ns = NoveltySearch(
            environment=gw,
            nb_generations=20,
            population_size=10,
            offspring_size=10,
            meta_generation=0,

            ag_type=GridAgentGuesser,

            generator={
                "function":lambda x, **kw: x(**kw),
                "parameters":{
                    "grid_size":gw.size
                }
            },

            mutator={
                "function": lambda x, **kw: x.mutate(),
                "parameters":{
                }
            },

            archive = {
                "type":NoveltyArchive,
                "parameters":{
                    "neighbouring_size":4,
                    "max_size":None,
                }
            },
        )

        pop, log, hof = ns()
        print(log)

        # Novelty over the all archive
        gw.visualise_as_grid(
            list_state_hist=[[behavior] for behavior in ns.archive.behaviors],
            show_traj=True,
            show_start=False,
            show_end=True,
            show=True
        )
