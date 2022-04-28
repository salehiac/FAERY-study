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
    
    def _run_inner(self, population):
        """
        Evaluates a population of individuals and saves their novelty as fitness
        """

        fitnesses = self.toolbox.map(self.toolbox.evaluate, population)

        # Updating the archive
        for ag in population:
            next(fitnesses)
            self.archive.update(ag.behavior)
        
        # Computing the population's novelty
        if self.archive.get_size() >= self.archive.neigh_size:
            for ag in population:
                ag.fitness.values = self.archive.get_novelty(ag.behavior)


if __name__ == "__main__":

    import torch

    from class_gridworld import GridWorld
    from utils_worlds import GridWorldSparse40x40Mixed

    gw = GridWorld(**GridWorldSparse40x40Mixed, is_guessing_game=True)

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
