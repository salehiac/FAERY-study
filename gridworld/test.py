import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.nx_pydot import graphviz_layout


def _find_roots(graph, node, depth=0):
    """
    Returns a list of tuples of the node's roots
    [(index, depth)]
    """

    successors = list(graph.successors(node))
    if len(successors) == 0:
        return [(node, depth)]

    parents_and_depth = []
    for parent in successors:
        parents_and_depth += _find_roots(graph, parent, depth+1)
    
    unique_nodes = {}
    for pd in parents_and_depth:
        if pd[0] in unique_nodes:
            unique_nodes[pd[0]] = min(pd[1], unique_nodes[pd[0]])
        else:
            unique_nodes[pd[0]] = pd[1]

    return [(k,v) for k, v in unique_nodes.items()]


graph = nx.gn_graph(10)

graph = graph
pos = graphviz_layout(graph, prog="dot")
nx.draw(graph.reverse(), pos, labels={n:n for n in graph.nodes}, with_labels=True)

print("Final :", _find_roots(graph, 8))
plt.show()