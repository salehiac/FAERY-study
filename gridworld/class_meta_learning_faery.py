import networkx

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

        successors = graph.successors(node)
        if len(successors) == 0:
            return [(node, 0)]
        
        parents_and_depth = []
        for parent in successors:
            parents_and_depth.append(self._find_roots(graph, parent, depth+1))
        
        unique_nodes = {}
        for pd in parents_and_depth:
            if pd[0] in unique_nodes:
                unique_nodes[pd[0]] = min(pd[1], unique_nodes[pd[0]])
            else:
                unique_nodes[pd[0]] = pd[1]
        
        return [(k,v+depth) for k, v in parents_and_depth.items()]

    def _update_fitness(self, population):
        """
        population : unused argument
        """

        # Updating metadata (logbook, etc...)
        super()._update_fitness(None)

        # For all instances, backtrack the solvers to root and retrieve their depth
        for instance in self.instances:
            graph = networkx.DiGraph(instance.history.genealogy_tree)
            for solver in instance.solvers:
                parents_and_depth = self._find_roots(graph, solver.history_index)
        
        # Computing the scores
        agent_to_score = {ag:[0, float("inf")] for ag in self.population}
        index_to_agent = {ag.history_index:ag for ag in self.population}
        for parent, depth in parents_and_depth:
            agent = index_to_agent[parent]
            agent_to_score[agent][0] += 1
            agent_to_score[agent][1] += depth
        
        # Updating the scores (couldn't do it before because deap's fitnesses are tuples)
        for parent in parents_and_depth:
            agent = index_to_agent[parent]

            agent.fitness.values = (
                agent_to_score[agent][0],
                - agent_to_score[agent][1] / agent_to_score[agent][0]
            )
        
        return True
