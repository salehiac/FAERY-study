import torch
import random
import numpy as np

from abc import ABC
from deap import tools, creator, base

from class_gridagent import GridAgent


class InnerAlgorithm(ABC):
    """
    Inner algorithm, runs on an environment and samples a new population at each step
    """

    def __init__(
        self, environment,
        
        nb_generations,
        population_size, offspring_size,
        cross_over_prob=.3, mutation_prob=.1,

        weights=(1,),
        ag_type=GridAgent,
        simulation_length=100,

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

        breeder={
            "function":tools.cxSimulatedBinary,
            "parameters":{
                "eta":15,
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
                "avg": np.mean,
                "std": np.std,
                "min": np.min,
                "max": np.max,
                "fit": lambda x:x[0],
            },
        },

        logbook=tools.Logbook(),
        hall_of_fame_parameters={
            "maxsize":1,
        },

        meta_generation = None
        ):
        """
        environment : simulation environment
        
        nb_generations : number of generations
        population_size : size of the population
        offspring_size : size of the offspring

        cross_over_prob, mutation_prob : cross-over and mutation probability at each step
        
        weights : weights to use for the fitness function (1,) for maximization
                                                          (-1,) for minimization
        ag_type : individuals' type

        generator, breeder, mutator, selector : functions to use for corresponding deap 
            operators, define function then its parameters
        
        statistics : statistics deap will follow during the algorithm

        logbook : evolution records as chronological list of dictionnaries (optional) user
            can provide a new chapter as argument
        
        hall_of_fame_parameters : parameters to use for the hall-of-fame

        meta_generation : at which meta generation was the algorithm called (optional)
        """
        
        self.environment = environment
        self.nb_generations = nb_generations
        self.population_size = population_size
        self.offspring_size = offspring_size

        self.cross_over_prob = cross_over_prob
        self.mutation_prob = mutation_prob

        self.weights = weights
        self.ag_type = ag_type
        self.simulation_length = simulation_length

        self.meta_generation = meta_generation

        # Initialiazing deap's toolbox
        creator.create("MyFitness", base.Fitness, weights=self.weights)
        creator.create("Individual", self.ag_type, fitness=creator.MyFitness)

        self.toolbox = base.Toolbox()

        self.toolbox.register("individual", generator["function"], creator.Individual, **generator["parameters"])
        self.toolbox.register("mate", breeder["function"], **breeder["parameters"])
        self.toolbox.register("mutate", mutator["function"], **mutator["parameters"])
        self.toolbox.register("select", selector["function"], **selector["parameters"])
        self.toolbox.register("evaluate", self._evaluate)

        #   Statistics gather results
        self.statistics = tools.Statistics(**statistics["parameters"])
        for name, func in statistics["to_register"].items():
            self.statistics.register(name, func)
        
        #   Evolution history
        self.logbook = logbook

        #   Remembering the best individual(s)
        self.hall_of_fame = tools.HallOfFame(**hall_of_fame_parameters)

        # The population
        self.population = None
    
    def __del__(self):
        """
        Garbage collection
        """

        del creator.MyFitness
        del creator.Individual

    def _evaluate(self, ag):
        """
        Runs the environment on a given agent
        Returns its fitness
        """
        
        return self.environment(ag, nb_steps=self.simulation_length)[0]
    
    def _run_inner(self, population):
        """
        Evaluates a population of individuals and saves their fitness
        Typically where a novelty search would be implemented
        """

        fitnesses = self.toolbox.map(self.toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

    def _compile_stats(self, population, generation):
        """
        Saves the statistics in class's objects
        """

        self.hall_of_fame.update(population)
        self.logbook.record(
            gen=generation,
            **self.statistics.compile(population)
        )

    def __call__(self):
        """
        Runs the algorithm
        """

        # Initialization
        self.population = [self.toolbox.individual() for _ in range(self.population_size)]

        self._run_inner(self.population)
        # self._compile_stats(self.population, 0)

        # Running the algorithm
        for g in range(self.nb_generations):

            # Select the next generation
            offspring = [
                self.toolbox.clone(ind)
                for ind in self.toolbox.select(self.population, self.offspring_size)
            ]
            
            # # Apply crossover on the offspring
            # tabOffspring = list(map(self.toolbox.clone, offspring))
            # for child1, child2 in zip(tabOffspring[::2], tabOffspring[1::2]):
            #     if random.random() < self.cross_over_prob:
            #         self.toolbox.mate(child1, child2)
            #         del child1.fitness.values
            #         del child2.fitness.values
                    
            # Apply mutation on the offspring
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals
            self._run_inner(offspring)
            
            # Adding the offspring to the population
            self.population += offspring[:]
            # Keeping the best individuals of the population
            self.population = self.toolbox.select(self.population, self.population_size)

            self._compile_stats(self.population, g)

        return self.population, self.hall_of_fame, self.logbook

