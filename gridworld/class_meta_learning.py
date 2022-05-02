from class_toolbox_algorithm import ToolboxAlgorithmGridWorld

from class_gridworld import GridWorld
from utils_worlds import GridWorldSparse40x40Mixed


class MetaLearning(ToolboxAlgorithmGridWorld):
    """
    Meta-learning class framework (for gridworld)
    """

    def __init__(
        self,      
        
        nb_generations_outer,
        population_size_meta, offspring_size_meta,

        inner_algorithm,
        nb_generations_inner,
        population_size_inner, offspring_size_inner,

        environment={
            "type":GridWorld,
            "parameters":{
                "is_guessing_game":True,
                **GridWorldSparse40x40Mixed
            }
        },
        nb_environments=25,

        breeder=None,
        mutator=None,
        multiprocessing=True,
        **toolbox_kwargs,
        ):
        
        super().__init__(
            nb_generations=nb_generations_outer,
            population_size=population_size_meta,
            offspring_size=offspring_size_meta,
            breeder=breeder,
            mutator=mutator,
            multiprocessing=multiprocessing,
            **toolbox_kwargs
        )

        self.environments = [
            environment["type"](**environment["parameters"])
            for _ in range(nb_environments)
        ]

        self.nb_generations_inner = nb_generations_inner
        self.population_size_inner = population_size_inner
        self.offspring_size_inner = offspring_size_inner

        self.generation = 0

    def _evaluate(self, ag):
        """
        Runs an agent on all the environments
        """

        # init population
        # transmettre paramètres
        return self.toolbox.map(
            lambda env: env(ag),
            self.environments,

        )
    
    def _update_fitness(self, population):
        """
        Updates the generation and changes the goals on all the environments (multi-tasks)
        """

        self.generation += 1

        for env in self.environments:
            env.reset(change_goal=True)
        
        return super()._update_fitness(population)
