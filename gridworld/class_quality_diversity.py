from scipy.spatial import KDTree

from class_toolbox_algorithm import ToolboxAlgorithmGridWorld
from class_novelty_archive import NoveltyArchive


class QualityDiversity(ToolboxAlgorithmGridWorld):
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

        selection_weights=(1,1),

        should_compile_stats_archive=True,

        **kwargs
        ):
        """
        archive : type and parameters of implemented archive
        should_compile_stats_archive : boolean checking if statistics should be compiled on 
                                       the whole archive, or just the current population
        """

        super().__init__(
            *args,
            selection_weights=selection_weights,
            **kwargs
        )

        self.archive = archive["type"](**archive["parameters"])
        self.env_kd_tree = KDTree(self.environment.reward_coords)

        self.should_compile_stats_archive = should_compile_stats_archive
    
    def _update_fitness(self, population):
        """
        Evaluates a population of individuals and saves their novelty and quality as fitness
        """

        fitnesses = self.toolbox.map(self._evaluate, population)
        
        # Updating the archive
        for ag in population:
            next(fitnesses)
            self.archive.update(ag)

        # Computing the population's novelty
        if self.archive.get_size() >= self.archive.neigh_size:
            for ag in population:
                ag.fitness.values = (
                    self.archive.get_novelty(ag.behavior)[0],
                    self.env_kd_tree.query(ag.behavior, k=1)[0]
                )
            return True
        return False

    def _compile_stats(self, population, logbook_kwargs={}):
        """
        Compile the stats on the whole archive
        """

        if self.should_compile_stats_archive is True:
            self.hall_of_fame.update(self.archive.all_agents)
            self.logbook.record(
                **logbook_kwargs,
                **self.statistics.compile(self.archive.all_agents)
            )
        else:
            super()._compile_stats(
                population=population,
                logbook_kwargs=logbook_kwargs
            )
    
    def reset(self):
        """
        Resets the algorithm and the archive
        """

        super().reset()
        self.archive.reset()
        

if __name__ == "__main__":

    from class_gridagent import GridAgentNN, GridAgentGuesser

    compute_NN = False
    compute_Guesser = True

    # For NN agents
    if compute_NN is True:

        import torch
        from deap import tools

        ns = QualityDiversity(
            nb_generations=20,
            population_size=10,
            offspring_size=10,

            ag_type=GridAgentNN,

            creator_parameters={
                "individual_name":"NNIndividual",
                "fitness_name":"NNFitness",
             },

            generator={
                "function":lambda x, **kw: x(**kw),
                "parameters":{
                    "input_dim":2, 
                    "output_dim":2,
                    "hidden_layers":3, 
                    "hidden_dim":10,
                    "use_batchnorm":False,
                    "non_linearity":torch.tanh, 
                    "output_normalizer":lambda x: torch.round((40 - 1) * abs(x)).int(),
                    # Something is wrong in normalization
                }
            },

            mutator={
                "function":tools.mutPolynomialBounded,
                "parameters":{
                    "eta":15,
                    "low":-1,
                    "up":1,
                    "indpb":.3,
                }
            },
        )

    # For guesser agents (those in the article)
    if compute_Guesser is True:
      
        ns = QualityDiversity(
            nb_generations=20,
            population_size=10,
            offspring_size=10,

            ag_type=GridAgentGuesser,
        )

    pop, log, hof = ns()
    print(log)

    ns.environment.visualise_as_grid(
        list_state_hist=[[behavior] for behavior in ns.archive.behaviors],
        show_traj=True,
        show_start=False,
        show_end=True,
        show=True
    )
