from copy import deepcopy
from abc import abstractmethod

from deap import tools
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

        outer_parameters_name=("./deap_parameters/parameters.py", "parameters"),
        inner_parameters_name=("./deap_parameters/parameters.py", "parameters"),

        toolbox_kwargs_inner={
        },

        multiprocessing=False,

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
            parameters_name=outer_parameters_name,
            multiprocessing=multiprocessing,
            **toolbox_kwargs_outer
        )

        if init_environments is None or len(init_environments) == 0:
            environments = [environment["type"](**environment["parameters"]) for _ in range(nb_instances)]
        else:
            environments = init_environments[:]
        self.model_environment = deepcopy(environments[0])
        self.environment = self.model_environment

        self.instances = [
            inner_algorithm(
                environment=environments[i],
                nb_generations=nb_generations_inner,
                population_size=population_size_inner,
                offspring_size=offspring_size_inner,
                parameters_name=inner_parameters_name,
                ag_type=self.ag_type,
                **toolbox_kwargs_inner
            ) for i in range(nb_instances)
        ]

        self.instances[0].printpop = True
        
        # Each chapter is a meta_step, inside of which each chapter is an instance
        # self.inner_logbook = tools.Logbook()
        # self.inner_logbook.header = [
        #     self.metastep_header.format(k) for k in range(self.nb_generations)
        # ]
        # for chapter in self.inner_logbook.chapters:
        #     self.inner_logbook.chapters[chapter].header = [
        #         self.instance_header.format(k) for k in nb_instances
        #     ]
        #   HOTFIX : ONLY LIST OF LISTS OF LOGBOOKS
        self.inner_logbook = [[None for _ in range(nb_instances)] for __ in range(nb_generations_outer+1)]

        self.nb_generations_inner = nb_generations_inner
        self.population_size_inner = population_size_inner
        self.offspring_size_inner = offspring_size_inner
        self.toolbox_kwargs_inner = toolbox_kwargs_inner

        self.generation = 0
    
    @abstractmethod
    def _update_fitness(self, population):
        """
        Updates the generation and runs the instances
        """

        # Preparing the instances
        for i, instance in enumerate(self.instances):
            instance.reset()

        # launching instances (parallellized if self.multiprocessing = True)
        results = list(self.toolbox.map(
            lambda instance: instance(
                init_population=self.toolbox.clone(population),
                verbose=False,
                show_history=False
            ),
            self.instances
        ))
       
        # As far as the inner algorithms are concerned, the init population doesn't have any parent
        #   prevents looping in meta-learning...
        for instance in self.instances:
            for k in range(len(population)):
                instance.history.genealogy_tree[k+1] = tuple()

        # Updating the logbook
        # for i, instance in enumerate(self.instances):
        #     self.inner_logbook.chapters[
        #         self.metastep_header.format(self.generation)
        #     ].chapters[
        #         self.instance_header.format(i)
        #     ] = instance.logbook
        self.inner_logbook[self.generation] = [deepcopy(instance.logbook) for instance in self.instances]

        self.generation += 1

        # Updates the fitness (algorithm specific)
        return NotImplemented
