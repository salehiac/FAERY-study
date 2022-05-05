from abc import abstractmethod

from class_toolbox_algorithm import ToolboxAlgorithmGridWorld

from environment.class_gridworld import GridWorld
from environment.utils_worlds import GridWorldSparse40x40Mixed


class MetaLearning(ToolboxAlgorithmGridWorld):
    """
    Meta-learning class framework (for gridworld)
    """

    metastep_header = "metastep_{}"
    instance_header = "instance_{}"

    def __init__(
        self,      
        
        nb_generations_outer,
        population_size_outer, offspring_size_outer,

        inner_algorithm,
        nb_generations_inner,
        population_size_inner, offspring_size_inner,
        
        nb_instances=25,
        environment={
            "type":GridWorld,
            "parameters":{
                "is_guessing_game":True,
                **GridWorldSparse40x40Mixed
            }
        },
        init_environments=None,

        toolbox_kwargs_inner={
        },

        multiprocessing=True,

        breeder=None,
        mutator=None,
        **toolbox_kwargs_outer,
        ):
        """
        params_outer : parameters for the outer algorithm
        params_inner : parameters for the inner algorithms

        nb_environment : number of different environments to run in parallel (hopefully multi-processing)
        environment : type and parameters for the environments if all identical
        init_environments :
            if not None : meta-learning will run on specified environments (bypassing previous arguments)
            useful if user wants to test adaptation on a specific array of tasks
            (ex : grasping and pushing; walking and jumping; etc...)
        """
        
        super().__init__(
            nb_generations=nb_generations_outer,
            population_size=population_size_outer,
            offspring_size=offspring_size_outer,
            breeder=breeder,
            mutator=mutator,
            multiprocessing=multiprocessing,
            **toolbox_kwargs_outer
        )

        if init_environments is None or len(init_environments):
            environments = nb_instances * [environment["type"](**environment["parameters"])]
        else:
            environments = init_environments[:]

        self.instances = [
            inner_algorithm(
                environment=environments[i],
                nb_generations=self.nb_generations_inner,
                population_size=self.population_size_inner,
                offspring_size=self.offspring_size_inner,

                ag_type=self.ag_type,
                meta_generation=self.generation
                **self.toolbox_kwargs_inner
            ) for i in range(nb_instances)
        ]

        # Each chapter is a meta_step, inside of which each chapter is an instance
        self.logbook.header = [
            self.metastep_header.format(k) for k in range(self.nb_generations)
        ]
        for chapter in self.logbook.chapters:
            self.logbook.chapters[chapter].header = [
                self.instance_header.format(k) for k in nb_instances
            ]

        self.nb_generations_inner = nb_generations_inner
        self.population_size_inner = population_size_inner
        self.offspring_size_inner = offspring_size_inner
        self.toolbox_kwargs_inner = toolbox_kwargs_inner

        self.generation = 0
    
    @abstractmethod
    def _update_fitness(self, population):
        """
        Updates the generation and runs the instances
        population : disregarded argument
        """

        # Preparing the instances
        for i, instance in enumerate(self.instances):
            instance.reset()
            # We disregard the population as parameter
            #   because the inner algorithms use the whole prior population
            instance.population = self.population

        # launching instances (parallellized if self.multiprocessing = True)
        results = self.toolbox.map(
            lambda instance: instance(verbose=False),
            self.instances
        )

        # Updating the logbook
        self.logbook.chapters[
            self.metastep_header.format(self.generation)
        ].chapters[
            self.instance_header.format(i)
        ] = instance.logbook
        self.generation += 1

        # Updates the fitness (algorithm specific)
        return NotImplemented
