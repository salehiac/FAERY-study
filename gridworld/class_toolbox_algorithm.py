import random
import networkx
import importlib.util
import matplotlib.pyplot as plt

from scoop import futures
from deap import tools, creator, base
from networkx.drawing.nx_pydot import graphviz_layout

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

        stop_when_solution_found=True,
        max_steps_after_found=0,

        creator_parameters={
            "individual_name":"Individual",
            "fitness_name":"MyFitness",
        },

        selection_weights=(1,),
        cross_over_prob=.3, mutation_prob=.1,
        
        parameters_name=("./deap_parameters/parameters.py", "parameters"),

        multiprocessing = False
        
        ):
        """

        init_pop : the population at the start of the learning process
        stop_when_solution_found : stops the algorithm once a solution is found
        max_steps_after_found : steps to run after the first solution is found

        creator_parameters : parameters for deap's creator class
                             (name should be role-specific)

        ag_type, environment : type of the created agents, environment to run them on
        selection_weights : weights to use for the fitness function (1,) for maximization
                                                                    (-1,) for minimization

        cross_over_prob, mutation_prob : cross-over and mutation probability at each step
        
        parameters_name : tuple containing the name of the module and class that should be loaded
            for the algorithm's generator, mutator and breeder

        multiprocessing : if True uses scoop, else uses deap's default map
            /!\ work in progress
        """

        self.environment = environment
        self.ag_type = ag_type
        self.cross_over_prob = cross_over_prob
        self.mutation_prob = mutation_prob
        self.nb_generations = nb_generations

        self.population_size = population_size
        self.offspring_size = offspring_size

        self.stop_when_solution_found = stop_when_solution_found
        self.max_steps_after_found = max_steps_after_found

        # Initialiazing deap's toolbox
        self.creator_parameters = creator_parameters

        # Careful not to have the same names for creator attributes if they don't have the same roles
        if self.creator_parameters["fitness_name"] not in creator.__dict__:
            creator.create(
                self.creator_parameters["fitness_name"],
                base.Fitness,
                weights=len(selection_weights) * (1,)
            ) # ISSUE WITH DEAP, CAN'T USE WEIGHTS AT 0, WE APPLY THEM MANUALLY INSTEAD IN FIX
        self.selection_weights = selection_weights 

        if self.creator_parameters["individual_name"] not in creator.__dict__:
            creator.create(
                self.creator_parameters["individual_name"],
                ag_type,
                fitness=creator.__dict__[self.creator_parameters["fitness_name"]]
            )

        # Importing low level parameters
        spec = importlib.util.spec_from_file_location(parameters_name[1], parameters_name[0])
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        parameters = foo.ParametersGuesser()

        self.toolbox = base.Toolbox()

        self.toolbox.register(
            "individual",
            lambda  x, **kw: parameters.generator["function"](x, **kw),
            creator.__dict__[self.creator_parameters["individual_name"]],
            **parameters.generator["parameters"]
        )

        if parameters.breeder is not None:
            self.breeder = True
            self.toolbox.register(
                "mate",
                parameters.breeder["function"],
                **parameters.breeder["parameters"]
            )
        else:   self.breeder = None

        if parameters.mutator is not None:
            self.mutator = True
            self.toolbox.register(
                "mutate",
                parameters.mutator["function"],
                **parameters.mutator["parameters"]
            )
        else:   self.mutator = None

        self.toolbox.register("select", parameters.selector["function"], **parameters.selector["parameters"])

        # Statistics gather results
        self.statistics = tools.Statistics(**parameters.statistics["parameters"])
        for name, func in parameters.statistics["to_register"].items():
            self.statistics.register(name, func)
        
        #   Evolution history
        self.logbook_parameters = parameters.logbook_parameters
        self.logbook = tools.Logbook(**self.logbook_parameters)

        #   Remembering the best individual(s)
        self.hall_of_fame_parameters = parameters.hall_of_fame_parameters
        self.hall_of_fame = tools.HallOfFame(**self.hall_of_fame_parameters)

        # Setting up multiprocessing
        if multiprocessing is True:
            self.toolbox.register("map", futures.map)

        self.history = tools.History()

        self.population = None
        self.done = False
        self.solvers = set()
    
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

    def _mate(self, child1, child2):
        """Allows for special (evironment specific) operations to be performed during mating"""

        self.toolbox.mate(child1, child2)

    def _mutate(self, mutant):
        """Allows for special (evironment specific) operations to be performed during mutation"""

        self.toolbox.mutate(mutant)

    def _evaluate(self, ag):
        """
        Runs the environment on a given agent
        Returns its fitness
        """

        fitness, ag.done, state_hist = self.environment(ag, nb_steps=1)
        if ag.done is True:
            self.done = True
            self.solvers.add(ag)

        return fitness
    
    def _update_fitness(self, population) -> bool:
        """
        Evaluates a population of individuals and saves their fitness
        Typically where a novelty search would be implemented

        Computing fitness on all individuals again (not just the new one) because of the fix
        RETURN BOOLEAN IF FIX SHOULD BE APPLIED
        """

        fitnesses = map(self._evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        return True
    
    def _update_history(self, population):
        """
        Updates the evolutionnary tree with the given population
        """

        for ind in sorted(population, key=lambda x: x.id):
            self.history.update([ind])

    def _compile_stats(self, population, logbook_kwargs={}):
        """
        Saves the statistics
        """
        
        self.hall_of_fame.update(population)
        self.logbook.record(
            **logbook_kwargs,
            **self.statistics.compile(population)
        )
    
    def _transform(self, offspring):
        """
        Apply transformations to the offspring, such as breeding and mutations
        """

        # Apply crossover on the offspring
        if self.breeder is not None:
            for (child1, child2) in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cross_over_prob:
                    self._mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
        
        # Apply mutation on the offspring
        if self.mutator is not None:
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self._mutate(mutant)
                    del mutant.fitness.values

        return offspring


    def __call__(self, init_population=None, verbose=True, show_history=False):
        """
        Runs the algorithm
        """

        # Initialization
        if verbose is True:
            print("Initializing population..", end="\r")
        
        self.population = [
            self.toolbox.individual(init_position=self.environment.reset())
            for _ in range(self.population_size)
        ] if init_population is None else init_population

        #   Computing the initial fitness
        self._update_fitness(self.population)
        if len(self.history.genealogy_history) == 0:
            self._update_history(self.population)
        self._compile_stats(self.population, {"gen":0})

        # Running the algorithm
        supp_steps = 0
        for g in range(self.nb_generations):
            if verbose is True:
                print(
                    "Running algorithm {}/{}..".format(g+1, self.nb_generations),
                    end="\r"
                )

            # Select the next generation
            offspring = self._transform(list(map(
                    self.toolbox.clone,
                    self.population
            )))

            # Evaluate the offspring
            self._update_fitness(offspring)

            # Select the individuals for the next population
            #   Lots of manipulation to make sure they're selected uniquely
            #       (because of deap's toolbox.clone)
            id_to_agent = {
                ag.id:ag for ag in self.population + offspring
            }

            old_population_id = [ag.id for ag in self.population]
            self.population = self.toolbox.select(
                list(id_to_agent.values()),
                self.population_size
            )
            new_population_id = [ag.id for ag in self.population]

            self._update_history([
                id_to_agent[i]
                for i in set(new_population_id) - set(old_population_id)
            ])
            self._compile_stats(self.population, {"gen":g+1})

            if self.stop_when_solution_found and self.done:
                if verbose is True:
                    print()
                    print("Solution found at generation {}".format(g), end='\r')
                
                supp_steps += 1
                if supp_steps > self.max_steps_after_found:
                    break

        if verbose is True:
            print()

        if show_history is True:
            self.show_history()
            
        return self.population, self.logbook, self.hall_of_fame
    
    def show_history(self):
        """
        Shows the history tree generated during execution
        """

        graph = networkx.DiGraph(self.history.genealogy_tree)
        graph = graph.reverse()     # Make the graph top-down

        pos = graphviz_layout(graph, prog="dot")
        for k in range(len(self.selection_weights)):
            plt.figure()
            plt.title("Evolutionary tree, F{}".format(k))
            colors = [self.history.genealogy_history[i].fitness.values[k] for i in graph]
            networkx.draw(graph, pos, node_color=colors)
        plt.show()

    def reset(self):
        """
        Resets the algorithm
        """

        self.population = None
        self.done = False
        self.solvers = set()

        self.history = tools.History()
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

        environment=GridWorld(
            **GridWorldSparse40x40Mixed,
            is_guessing_game=True
        ),

        **kwargs
        ):

        super().__init__(
            *args,
            ag_type=ag_type,
            environment=environment,
            **kwargs
        )

    def _make_new_agent(self, ag):
        """
        Increments the current id of GridAgent and gives it to the agent
        """

        ag.id = self.ag_type.id
        self.ag_type.id += 1
  
        return ag
    
    def _mate(self, child1, child2):
        """
        Mates two agents together, updates their id
        """

        super()._mate(child1, child2)
        self._make_new_agent(child1)
        self._make_new_agent(child2)
    
    def _mutate(self, mutant):
        """
        Mutates the agent, updates its id
        """

        super()._mutate(mutant)
        self._make_new_agent(mutant)
    
    def reset(self):
        """
        Resets the algorithm and re-samples the goals of the environment
        """

        super().reset()
        self.environment.reset(change_goal=True)
    