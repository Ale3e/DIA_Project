import networkx as nx
import numpy as np
from graph import generate_graph, weight_nodes, weight_edges

#struttura nodi grafo
#graph.nodes() = {0: {'id': 0, 'cost': 0.6, 'status': 'susceptible'}, 1: {'id': 1, 'cost': 0.38, 'status': 'susceptible'}, 2: {'id'...

def influence_cascade(graph, seed_set):
    #todo
    return


if __name__ == "__main__":

    features = [0.10, 0.08, 0.05, 0.02]

    graph = generate_graph(100, 5, 0.05, 123)
    graph = weight_edges(graph, features)
    graph = weight_nodes(graph)

    seed_set = [0, 1, 2, 3]

    t = 0
    weighted_spread = 0.0
    triggered_nodes = []
    todo_nodes = seed_set

    print(todo_nodes)

    #activatye seed nodes
    for i in range(len(todo_nodes)):
        graph.nodes[todo_nodes[i]]['status'] = 'active'

    #IC
    #for node in todo_nodes:
    while len(todo_nodes) > 0:
        node = todo_nodes[0]
        print("At time: ")
        print(t)
        print(" from node ")
        print(node)
        print(graph.nodes[node]['id'])

        triggered_nodes.append(graph.nodes[node]['id'])

        print("influence propagates to node: ")

        for adj_node in graph.neighbors(node):

            if graph.nodes[adj_node]['status'] == 'susceptible':

                if np.random.rand() <= graph[node][adj_node]['prob']:

                    print(graph.nodes[adj_node]['id'])

                    graph.nodes[adj_node]['status'] = 'active'

                    todo_nodes.append(graph.nodes[adj_node]['id'])

        weighted_spread += graph.nodes[node]['cost']
        graph.nodes[node]['status'] = 'inactive'
        todo_nodes.remove(node)
        t += 1

    print(" Weighted spread is : ")
    print(weighted_spread)

    print("Triggered nodes are: ")
    for n in range(len(triggered_nodes)):
        print("Node: " + str(triggered_nodes[n]) + graph.nodes[triggered_nodes[n]['status']])















