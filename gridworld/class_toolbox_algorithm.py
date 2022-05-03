import torch
import random
import numpy as np

from scoop import futures

from deap import tools, creator, base

from environment.class_gridworld import GridWorld, GridWorldSparse40x40Mixed
from environment.class_gridagent import GridAgentGuesser


class ToolboxAlgorithmFix(type):
    """
    Metaclass to fix deap's issue with weights values
    """

    def correct_fitness(cls, func):
        """
        Fixing deap's issue, the package multiplies by the weights then
        divides the wvalues again, prohibiting usage of infinite and zero weights
        (which are useful in inheritance or ablation studies, ex: NoveltySearch from QualityDiversity)
        """

        def wrapper(self, population, *args, **kwargs):

            if func(self, population, *args, **kwargs) is True:
                for ag in population:
                    ag.fitness.values = tuple(
                        ag.fitness.values[i] * self.selection_weights[i]
                        for i in range(len(self.selection_weights))
                    )
        
        return wrapper

    def __new__(cls, name, bases, attr):

        try:
            attr["_update_fitness"] = ToolboxAlgorithmFix.correct_fitness(cls, attr["_update_fitness"])
        except KeyError:
            pass
        return super(ToolboxAlgorithmFix, cls).__new__(cls, name, bases, attr)


class ToolboxAlgorithm(metaclass=ToolboxAlgorithmFix):
    """
    Base class for deap-based algorithms
    """

    def __init__(
        self,

        ag_type, environment,
        nb_generations,
        population_size, offspring_size,

        init_population=None,

        creator_parameters={
            "individual_name":"Individual",
            "fitness_name":"MyFitness",
        },

        selection_weights=(1,),
        cross_over_prob=.3, mutation_prob=.1,
        
        generator={
            "function":lambda x, **kw: x(**kw),
            "parameters":{
                "input_dim":2, 
                "output_dim":2,
                "hidden_layers":3, 
                "hidden_dim":10,
                "use_batchnorm":False,
                "non_linearity":torch.tanh, 
                "output_normalizer":lambda x: x,
            }
        },

        # breeder={
        #     "function":tools.cxSimulatedBinary,
        #     "parameters":{
        #         "eta":15,
        #     }
        # },
        breeder=None,

        mutator={
            "function":tools.mutPolynomialBounded,
            "parameters":{
                "eta":15,
                "low":-1,
                "up":1,
                "indpb":.3,
            }
        },

        selector={
            "function":tools.selBest,
            "parameters":{

            }
        },

        statistics={
            "parameters":{
                "key":lambda ind: ind.fitness.values
            },
            "to_register":{
                "avg": lambda x: np.mean(x, axis=0),
                "std": lambda x: np.std(x, axis=0),
                "min": lambda x: np.min(x, axis=0),
                "max": lambda x: np.max(x, axis=0),
                # "fit": lambda x: x,
            },
        },

        logbook_parameters={
        },

        hall_of_fame_parameters={
            "maxsize":1,
        },

        multiprocessing = False
        
        ):
        """

        creator_parameters : parameters for deap's creator class
                             (name should be role-specific)

        ag_type, environment : type of the created agents, environment to run them on
        selection_weights : weights to use for the fitness function (1,) for maximization
                                                                    (-1,) for minimization

        cross_over_prob, mutation_prob : cross-over and mutation probability at each step
        
        generator, breeder, mutator, selector : functions to use for corresponding deap 
            operators, define function then its parameters
        
        statistics : statistics deap will follow during the algorithm

        logbook_parameters : evolution records as chronological list of dictionnaries (optional)
        
        hall_of_fame_parameters : parameters to use for the hall-of-fame

        multiprocessing : if True uses scoop, else uses deap's default map
        """

        self.environment = environment
        self.cross_over_prob = cross_over_prob
        self.mutation_prob = mutation_prob
        self.nb_generations = nb_generations

        self.population_size = population_size
        self.offspring_size = offspring_size

        # Initialiazing deap's toolbox
        self.creator_parameters = creator_parameters

        # Careful not to have the same names for creator attributes if they don't have the same roles
        if self.creator_parameters["fitness_name"] not in creator.__dict__:
            creator.create(
                self.creator_parameters["fitness_name"],
                base.Fitness,
                weights=len(selection_weights) * (1,)
            ) # ISSUE WITH DEAP, CAN'T USE WEIGHTS AT 0, WE APPLY THEM MANUALLY INSTEAD
            self.selection_weights = selection_weights 

        if self.creator_parameters["individual_name"] not in creator.__dict__:
            creator.create(
                self.creator_parameters["individual_name"],
                ag_type,
                fitness=creator.__dict__[self.creator_parameters["fitness_name"]]
            )

        self.toolbox = base.Toolbox()

        self.toolbox.register(
            "individual",
            generator["function"],
            creator.__dict__[self.creator_parameters["individual_name"]],
            **generator["parameters"]
        )

        if breeder is not None:
            self.breeder = True
            self.toolbox.register(
                "mate",
                breeder["function"],
                **breeder["parameters"]
            )
        else:   self.breeder = None

        if mutator is not None:
            self.mutator = True
            self.toolbox.register(
                "mutate",
                mutator["function"],
                **mutator["parameters"]
            )
        else:   self.mutator = None

        self.toolbox.register("select", selector["function"], **selector["parameters"])

        # Statistics gather results
        self.statistics = tools.Statistics(**statistics["parameters"])
        for name, func in statistics["to_register"].items():
            self.statistics.register(name, func)
        
        #   Evolution history
        self.logbook_parameters = logbook_parameters
        self.logbook = tools.Logbook(**self.logbook_parameters)

        #   Remembering the best individual(s)
        self.hall_of_fame_parameters = hall_of_fame_parameters
        self.hall_of_fame = tools.HallOfFame(**self.hall_of_fame_parameters)

        # Setting up multiprocessing
        if multiprocessing is True:
            self.toolbox.register("map", futures.map)
    
        # The population
        if init_population is None or init_population == []:
            self.population = None
    
    def __del__(self):
        """
        Garbage collection
        """

        # Not pretty..?
        try:
            del creator.__dict__[self.creator_parameters["fitness_name"]]
        except KeyError:
            pass

        try:
            del creator.__dict__[self.creator_parameters["individual_name"]]
        except KeyError:
            pass
    
    def _make_offspring(self):
        """
        Makes an offspring, can be overriden
        """

        return [
            self.toolbox.clone(self.population[_])
            for _ in range(self.offspring_size)
        ]

    def _evaluate(self, ag):
        """
        Runs the environment on a given agent
        Returns its fitness
        """

        return self.environment(ag, nb_steps=self.nb_generations)[0]
    
    def _update_fitness(self, population) -> bool:
        """
        Evaluates a population of individuals and saves their fitness
        Typically where a novelty search would be implemented

        MUST RETURN TRUE IF FITNESS VALUES WERE MODIFIED
        """

        fitnesses = self.toolbox.map(self._evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        return True
    
    def _compile_stats(self, population, logbook_kwargs={}):
        """
        Saves the statistics in class's objects
        """

        self.hall_of_fame.update(population)
        self.logbook.record(
            **logbook_kwargs,
            **self.statistics.compile(population)
        )
    
    def _mate(self, i, child1, child2):
        """
        Mates two agents at index i
        """

        self.toolbox.mate(child1, child2)

        self.population[2*i] = child1
        self.population[2*i+1] = child2
    
    def _mutate(self, i, mutant):
        """
        Mutates the mutant at index i
        """

        self.toolbox.mutate(mutant)
        self.population[i] = mutant

    def __call__(self, verbose=True):
        """
        Runs the algorithm
        """

        # Initialization
        if verbose is True:
            print("Initializing population..", end="\r")

        if self.population is None:
            self.population = [
                self.toolbox.individual()
                for _ in range(self.population_size)
            ]

        self._update_fitness(self.population)
        self._compile_stats(self.population, {"gen":0})

        # Running the algorithm
        for g in range(self.nb_generations):
            if verbose is True:
                print(
                    "Running algorithm {}/{}..".format(g+1, self.nb_generations),
                    end="\r"
                )

            # Select the next generation
            self.population += self._make_offspring()
            
            # Apply crossover on the population
            if self.breeder is not None:
                tabOffspring = list(map(self.toolbox.clone, self.population))
                for i, (child1, child2) in enumerate(zip(tabOffspring[::2], tabOffspring[1::2])):
                    if random.random() < self.cross_over_prob:
                        self._mate(i, child1, child2)

                        del child1.fitness.values
                        del child2.fitness.values
            
            # Apply mutation on the population
            if self.mutator is not None:
                for i, mutant in enumerate(self.population):
                    if random.random() < self.mutation_prob:
                        self._mutate(i, mutant)

                        del mutant.fitness.values

            # Evaluate the individuals
            self._update_fitness(self.population)

            # Keeping the best individuals of the population
            self.population = self.toolbox.select(self.population, self.population_size)

            self._compile_stats(self.population, {"gen":g+1})

        if verbose is True:
            print()

        return self.population, self.logbook, self.hall_of_fame
    
    def reset(self):
        """
        Resets the algorithm
        """

        self.population = None
        self.logbook = tools.Logbook(**self.logbook_parameters)
        self.hall_of_fame = tools.HallOfFame(**self.hall_of_fame_parameters)


class ToolboxAlgorithmGridWorld(ToolboxAlgorithm):
    """
    Toolbox algorithm working on GridWorld
    """

    def __init__(
        self,

        *args,

        ag_type=GridAgentGuesser,
        environment=GridWorld(**GridWorldSparse40x40Mixed, is_guessing_game=True),

        generator={
            "function":lambda x, **kw: x(**kw),
            "parameters":{
                "grid_size":40
            }
        },

        mutator={
            "function": lambda x, **kw: x.mutate(),
            "parameters":{
            }
        },

        **kwargs
        ):
        
        super().__init__(
            *args,
            ag_type=ag_type,
            environment=environment,
            generator=generator,
            mutator=mutator,
            **kwargs
        )

    def _make_new_agent(self, ag):
        """
        Increments the current id of GridAgent and gives it to the agent
        """

        ag.id = GridAgentGuesser.id
        GridAgentGuesser.id += 1

        return ag
    
    def _mate(self, i, child1, child2):
        """
        Mates two agents together, updates their id
        """

        super()._mate(i, child1, child2)
        self._make_new_agent(child1)
        self._make_new_agent(child2)
    
    def _mutate(self, i, mutant):
        """
        Mutates the agent, updates its id
        """

        super()._mutate(i, mutant)
        self._make_new_agent(mutant)
    
    def reset(self):
        """
        Resets the algorithm and re-samples the goals of the environment
        """

        super().reset()
        self.environment.reset(change_goals=True)
    