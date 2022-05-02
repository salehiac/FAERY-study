from class_toolbox_algorithm import ToolboxAlgorithmGridWorld
from class_gridagent import add_agent, GridAgentNN


class InnerAlgorithm(ToolboxAlgorithmGridWorld):
    """
    Inner algorithm framework for gridbox
    Changes ToolboxAlgorithm to run on an environment
    """

    def __init__(
        self,
        
        environment,
        
        nb_generations,
        population_size, offspring_size,
        
        meta_generation = None,

        ag_type=GridAgentNN,
        **toolbox_kwargs,
        ):
        """
        environment : simulation environment
        
        nb_generations : number of generations
        population_size : size of the population
        offspring_size : size of the offspring
        
        meta_generation : at which meta generation was the algorithm called (optional)

        ag_type : individuals' type
        toolbox_kwargs : kwargs for ToolboxAlgorithm metaclass
        """
        
        super().__init__(
            nb_generations=nb_generations,
            population_size=population_size,
            offspring_size=offspring_size,
            ag_type=ag_type,
            **toolbox_kwargs
        )

        self.environment = environment

        self.nb_generations = nb_generations
        self.population_size = population_size
        self.offspring_size = offspring_size

        self.meta_generation = meta_generation

    def _evaluate(self, ag):
        """
        Runs the environment on a given agent
        Returns its fitness
        """

        return self.environment(ag, nb_steps=self.nb_generations)[0]
