import torch
import random
import numpy as np

from scoop import futures

from abc import ABC, abstractmethod
from deap import tools, creator, base

from class_gridagent import add_agent, GridAgentNN


class ToolboxAlgorithm(ABC):
    """
    Base class for deap-based algorithms
    """

    def __init__(
        self,

        ag_type,
        nb_generations,
        population_size, offspring_size,

        init_population=None,

        creator_parameters={
            "individual_name":"Individual",
            "fitness_name":"MyFitness",
            "should_delete":True,
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

        logbook=tools.Logbook(),

        hall_of_fame_parameters={
            "maxsize":1,
        },

        multiprocessing = False
        
        ):
        """

        creator_parameters : parameters for deap's creator class

        ag_type : type of the created agents
        selection_weights : weights to use for the fitness function (1,) for maximization
                                                                    (-1,) for minimization
        cross_over_prob, mutation_prob : cross-over and mutation probability at each step
        
        generator, breeder, mutator, selector : functions to use for corresponding deap 
            operators, define function then its parameters
        
        statistics : statistics deap will follow during the algorithm

        logbook : evolution records as chronological list of dictionnaries (optional) user
            can provide a new chapter as argument
        
        hall_of_fame_parameters : parameters to use for the hall-of-fame

        multiprocessing : if True uses scoop, else uses deap's default map
        """

        self.cross_over_prob = cross_over_prob
        self.mutation_prob = mutation_prob
        self.nb_generations = nb_generations

        self.population_size = population_size
        self.offspring_size = offspring_size

        # Initialiazing deap's toolbox
        self.creator_parameters = creator_parameters
        creator.create(
            self.creator_parameters["fitness_name"],
            base.Fitness,
            weights=selection_weights
        )
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

        #   Statistics gather results
        self.statistics = tools.Statistics(**statistics["parameters"])
        for name, func in statistics["to_register"].items():
            self.statistics.register(name, func)
        
        #   Evolution history
        self.logbook = logbook

        #   Remembering the best individual(s)
        self.hall_of_fame = tools.HallOfFame(**hall_of_fame_parameters)

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

        # Not pretty
        if self.creator_parameters["should_delete"] is True:
            del creator.__dict__[self.creator_parameters["fitness_name"]]
            del creator.__dict__[self.creator_parameters["individual_name"]]
    
    @abstractmethod
    def _make_offspring(self):
        """
        Makes an offspring (should be made from copies of individuals)
        """

        return NotImplemented

    @abstractmethod
    def _evaluate(self, ag):
        """
        Runs the environment, the meta-learning, etc...
        """

        return NotImplemented
    
    def _update_fitness(self, population):
        """
        Evaluates a population of individuals and saves their fitness
        Typically where a novelty search would be implemented
        """

        fitnesses = self.toolbox.map(self._evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
    
    def _compile_stats(self, population, logbook_kwargs={}):
        """
        Saves the statistics in class's objects
        """

        self.hall_of_fame.update(population)
        self.logbook.record(
            **logbook_kwargs,
            **self.statistics.compile(population)
        )
    
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
            offspring = self._make_offspring()
            kept_offspring = []
            
            # Apply crossover on the offspring
            if self.breeder is not None:
                tabOffspring = list(map(self.toolbox.clone, offspring))
                for child1, child2 in zip(tabOffspring[::2], tabOffspring[1::2]):
                    if random.random() < self.cross_over_prob:
                        self.toolbox.mate(child1, child2)
                        kept_offspring.append(child1)
                        kept_offspring.append(child2)
                        del child1.fitness.values
                        del child2.fitness.values
                    
            # Apply mutation on the offspring
            if self.mutator is not None:
                for mutant in offspring:
                    if random.random() < self.mutation_prob:
                        self.toolbox.mutate(mutant)
                        kept_offspring.append(mutant)
                        del mutant.fitness.values

            # Adding the offspring to the population
            self.population += kept_offspring
            del offspring

            # Evaluate the individuals
            self._update_fitness(self.population)

            # Keeping the best individuals of the population
            self.population = self.toolbox.select(self.population, self.population_size)

            self._compile_stats(self.population, {"gen":g+1})

        if verbose is True:
            print()

        return self.population, self.logbook, self.hall_of_fame


class ToolboxAlgorithmGridWorld(ToolboxAlgorithm):
    """
    Toolbox algorithm working on GridWorld (overrides _make_offspring)
    """

    def __init__(self, *args, ag_type=GridAgentNN, **kwargs):
        super().__init__(*args, ag_type=ag_type, **kwargs)

    def _make_offspring(self):
        """
        Makes an offspring (should be made from copies of individuals)
        """

        return [
            add_agent(self.toolbox.clone(ind))
            for ind in self.toolbox.select(self.population, self.offspring_size)
        ]
    