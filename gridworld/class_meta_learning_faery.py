import networkx
import matplotlib.pyplot as plt

from networkx.drawing.nx_pydot import graphviz_layout

from class_meta_learning import MetaLearning


class MetaLearningFAERY(MetaLearning):
    """
    Implementation of the FAERY algorithm
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
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
        
        unique_nodes = {}
        for pd in parents_and_depth:
            if pd[0] in unique_nodes:
                unique_nodes[pd[0]] = min(pd[1], unique_nodes[pd[0]])
            else:
                unique_nodes[pd[0]] = pd[1]

        return [(k,v) for k, v in unique_nodes.items()]

    def _update_fitness(self, population):
        """
        population : unused argument
        """

        # Updating metadata (logbook, etc...)
        super()._update_fitness(None)

        if len(self.history.genealogy_history) == 0:
            for ind in self.population:
                self.history.update([ind])

        # For all instances, backtrack the solvers to root and retrieve their depth
        parents_and_depth = []
        for instance in self.instances:
            graph = networkx.DiGraph(instance.history.genealogy_tree)
            for solver in instance.solvers:
                parents_and_depth += self._find_roots(graph, solver.history_index)

        # Computing the scores
        agent_to_score = {ag:[0, 0] for ag in self.population}
        index_to_agent = {ag.history_index:ag for ag in self.population}
        
        for parent, depth in parents_and_depth:
            agent = index_to_agent[parent]

            agent_to_score[agent][0] += 1
            agent_to_score[agent][1] -= depth

        # Updating the scores (couldn't do it before because deap's fitnesses are tuples)
        for parent in parents_and_depth:
            agent = index_to_agent[parent[0]]

            agent.fitness.values = (
                agent_to_score[agent][0],
                - agent_to_score[agent][1] / agent_to_score[agent][0]
            )
        
        for ag in [ind for ind in self.population if ind.fitness.valid is False]:
            ag.fitness.values = (
                -1 * self.selection_weights[0] * float("inf"),
                -1 * self.selection_weights[1] * float("inf")
            )

        for i, ag in enumerate(self.population):
            print(i, ag.fitness.values)
        self.show_tree()

        return True

    def show_tree(self):
        """
        Shows the evolutionary tree for the current generation
        """

        all_graphs = networkx.DiGraph()
        all_colors = []
        for k, instance in enumerate(self.instances):
            graph = networkx.DiGraph(instance.history.genealogy_tree).reverse()

            relabel_mapping = dict(zip(graph.nodes, ["{}_{}".format(k, node) for node in graph.nodes]))
            graph_relabeled = networkx.relabel_nodes(graph, relabel_mapping)

            all_graphs = networkx.compose(all_graphs, graph_relabeled)

        for i in range(1, len(self.population)+1):
            for k in range(len(self.instances)):
                all_graphs.add_edge(i, "{}_{}".format(k, i))
            all_colors += [0]
        
        all_colors = []
        for node in all_graphs.nodes:
            try:
                k, i = str(node).split('_')
                all_colors.append(
                    0.5 if self.instances[int(k)].history.genealogy_history[int(i)].done is False
                    else 1
                )
            except ValueError:
                all_colors.append(0)

        pos = graphviz_layout(all_graphs, prog="dot")
        networkx.draw(
            all_graphs,
            pos,
            labels={n:n for n in all_graphs.nodes}, with_labels=True,
            node_color=all_colors, cmap=plt.cm.Blues
        )

        plt.show()

if __name__ == "__main__":

    from environment.class_gridagent import GridAgentNN
    from class_quality_diversity import QualityDiversity

    faery = MetaLearningFAERY(
        nb_instances=5,

        nb_generations_outer=0,
        population_size_outer=10, offspring_size_outer=10,

        inner_algorithm=QualityDiversity,
        nb_generations_inner=20,
        population_size_inner=10, offspring_size_inner=10,

        selection_weights=(1,1),

        ag_type=GridAgentNN,

        creator_parameters={
            "individual_name":"NNIndividual",
            "fitness_name":"NNFitness",
        },

        toolbox_kwargs_inner={
            "stop_when_solution_found":True,
            "mutation_prob":.3,
        }
    )

    pop, log, hof = faery()
