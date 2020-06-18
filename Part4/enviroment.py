from graph import generate_graph, weight_nodes, weight_edges, get_probabilities
import numpy as np

class Environment:
    def __init__(self, graph):
        self.graph = graph
        self.probabilities = get_probabilities(graph)

    def round(self, pulled_arm):
        success = np.random.binomial(1, self.probabilities)
        active_edges = [edge for i, edge in enumerate(self.graph.edges) if success[i] == 1]
        return active_edges



