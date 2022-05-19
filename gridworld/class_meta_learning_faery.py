import networkx
import matplotlib.pyplot as plt

from networkx.drawing.nx_pydot import graphviz_layout

from class_meta_learning import MetaLearning


class MetaLearningFAERY(MetaLearning):
    """
    Implementation of the FAERY algorithm
    """

    def __init__(self, *args, should_show_evo_tree=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.should_show_evo_tree = should_show_evo_tree
    
    def _find_roots(self, graph, node, depth=0):
        """
        Returns a list of tuples of the node's roots
        [(index, depth)]
        """

        successors = list(graph.successors(node))
        if len(successors) == 0:
            return [(node, depth)]

        parents_and_depth = []
        for parent in successors:
            parents_and_depth += self._find_roots(graph, parent, depth+1)
        
        # Taking the minimum depth for each node as several paths are possible if there are cross-overs
        unique_nodes = {}
        for pd in parents_and_depth:
            if pd[0] in unique_nodes:
                unique_nodes[pd[0]] = min(pd[1], unique_nodes[pd[0]])
            else:
                unique_nodes[pd[0]] = pd[1]

        return [(k,v) for k, v in unique_nodes.items()]

    def _update_fitness(self, population):
        """
        Updates the fitness of the population according to FAERY
        """

        # SHUFFLING -> index overflow maybe due to genealogy index?

        # Updating metadata (logbook, etc...)
        population = sorted(population, key=lambda x: x.id)

        super()._update_fitness(population)
        for ind in population:
            if hasattr(ind, "history_index") is False:
                self.history.update([ind])

        # For all instances, backtrack the solvers to root and retrieve their depth
        parents_and_depth = []
        for instance in self.instances:
            graph = networkx.DiGraph(instance.history.genealogy_tree)
            for solver in instance.solvers:
                # Nodes are not always in order in the genealogy trees..
                parents_and_depth += [
                    (
                        instance.history.genealogy_history[parent].id,
                        depth
                    ) for parent, depth in self._find_roots(graph, solver.history_index)
                ]
        
        # Computing the scores
        id_to_population = {ag.id:ag for ag in population}
        agent_to_score = {ag:[0, 0] for ag in population}

        for parent, depth in parents_and_depth:
            agent = id_to_population[parent]

            agent_to_score[agent][0] += 1
            agent_to_score[agent][1] -= depth

        # Updating the scores (couldn't do it before because deap's fitnesses are tuples)
        for parent, depth in parents_and_depth:
            agent = id_to_population[parent]

            agent.fitness.values = (
                agent_to_score[agent][0],
                agent_to_score[agent][1] / agent_to_score[agent][0]
            )
        
        for ag in [ind for ind in population if ind.fitness.valid is False]:
            ag.fitness.values = (
                0,
                -1 * self.selection_weights[1] * float("inf")
            )
        
        # Propagating the scores to the history tree
        for ind in population:
            self.history.genealogy_history[ind.history_index].fitness.values = ind.fitness.values

        if self.should_show_evo_tree is True:
            for i, ag in enumerate(population):
                print("Place in meta-pop: {}\tID: {}\tFitness: {}".format(i, ag.id, ag.fitness.values))
            self.show_evo_tree()

        return False

    def show_evo_tree(self):
        """
        Shows the evolutionary tree for the current generation
        """
        
        # Concatenating all the graphs
        all_graphs = networkx.DiGraph()
        for k, instance in enumerate(self.instances):
            graph = networkx.DiGraph(instance.history.genealogy_tree).reverse()

            relabel_mapping = dict(zip(graph.nodes, ["{}_{}".format(k, node) for node in graph.nodes]))
            graph_relabeled = networkx.relabel_nodes(graph, relabel_mapping)

            all_graphs = networkx.compose(all_graphs, graph_relabeled)

        # Adding meta-nodes as anchor for graphs
        for i in range(1, len(self.population)+1):
            for k in range(len(self.instances)):
                all_graphs.add_edge(i, "{}_{}".format(k, i))
        
        # Updating the colors if a solution was found
        all_colors = []
        for node in all_graphs.nodes:
            try:
                k, i = str(node).split('_')
                all_colors.append(
                    0.4 if self.instances[int(k)].history.genealogy_history[int(i)].done is False
                    else 1
                )
            except ValueError:
                all_colors.append(0.15)

        # Plotting the graph
        pos = graphviz_layout(all_graphs, prog="dot")
        networkx.draw(
            all_graphs,
            pos,
            labels={n:n for n in all_graphs.nodes}, with_labels=True,
            node_color=all_colors, cmap=plt.cm.OrRd
        )

        plt.show()


if __name__ == "__main__":

    from environment.class_gridagent import GridAgentNN, GridAgentGuesser
    from class_quality_diversity import QualityDiversity

    faery = MetaLearningFAERY(
        nb_instances=5,

        nb_generations_outer=10,
        population_size_outer=25, offspring_size_outer=25,

        inner_algorithm=QualityDiversity,
        nb_generations_inner=20,
        population_size_inner=25, offspring_size_inner=25,
        inner_max_steps_after_found=10,

        selection_weights=(1,1),

        ag_type=GridAgentGuesser,

        creator_parameters={
            "individual_name":"NNIndividual",
            "fitness_name":"NNFitness",
        },

        toolbox_kwargs_inner={
            "stop_when_solution_found":True,
            "mutation_prob":1,
        },

        mutation_prob=1
    )

    faery.should_show_evo_tree = False
    pop, log, hof = faery(show_history=True)

    # Showing the meta-population history
    #   We have to run them on an environment first to generate their behaviors
    for ag in faery.history.genealogy_history.values():
        faery.model_environment(ag, 1)
    
    #   We don't need to show the sampled rewards
    faery.model_environment.reward_coords = []

    grid_hist = faery.model_environment.visualise_as_grid(
        list_state_hist=[[ag.behavior] for ag in faery.history.genealogy_history.values()],
        show_traj=True,
        show_start=False,
        show_end=True,
        show=False
    )

    faery.model_environment.reset()
    for ag in faery.population:
        faery.model_environment(ag, 1)
    
    faery.model_environment.reward_coords = []

    grid_final = faery.model_environment.visualise_as_grid(
        list_state_hist=[[ag.behavior] for ag in faery.population],
        show_traj=True,
        show_start=False,
        show_end=True,
        show=False
    )

    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(grid_hist)
    axs[1].imshow(grid_final)

    plt.show()
