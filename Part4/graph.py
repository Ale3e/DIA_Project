import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def generate_graph(n_nodes, k, probabilities, seed):
    """
    Returns a small world network with n nodes,
    starting with each node connected to its k nearest neighbors,
    and rewiring "probability"
    """

    graph = nx.newman_watts_strogatz_graph(n_nodes, k, probabilities, seed=seed)

    nx.set_node_attributes(graph, 0, 'id')
    nx.set_node_attributes(graph, 0.0, 'cost')
    nx.set_node_attributes(graph, 'susceptible', 'status')
    nx.set_edge_attributes(graph, 0.0, 'prob')

    for n in range(graph.number_of_nodes()):
        graph.nodes[n]['id'] = n

    return graph


def weight_edges(graph, features):
    """ Sets a probability to each edge based on a linear combination of 4 features
    Input: graph -- networkx Graph object
    f -- list of features probability
    """
    # if seed == None: seed_val = 0
    # else: seed_val = seed

    for edge in graph.edges():
        #np.random.seed(0)
        probability = np.random.binomial(1, 0.5, size=len(features))
        graph[edge[0]][edge[1]]['prob'] = round(sum(x * y for x, y in zip(probability, features)), 2)

    return graph


def weight_nodes(graph):
    """
    :param graph: input graph with probabilities on edges ONLY!!!
    :return: graph with cost of every node based on the probability of the edges starting from it
    """
    for node_idx in range(graph.number_of_nodes()):
        weight = []
        for node in graph.neighbors(node_idx):
            weight.append(graph[node_idx][node]['prob'])
        graph.nodes[node_idx]['cost'] = round(sum(weight) + 0.5, 2)

    return graph



if __name__ == "__main__":

    features = [0.10, 0.08, 0.05, 0.02]

    G = generate_graph(100, 5, 0.05, 123)
    G = weight_edges(G, features)
    G = weight_nodes(G)

    print(nx.info(G))

    print(G.adj[0])


    # print node attributes and of adj nodes with the influence prob
    for node_idx in range(G.number_of_nodes()):
        weight = []
        print("\nNode :")
        print(G.nodes[node_idx])
        print("Has the following adjacent nodes: ")
        for node in G.neighbors(node_idx):
            print(node)
            print(G.nodes[node])
            print(G[node_idx][node]['prob'])

    # print edges and their probability
    # n = 0
    # for edge in G.edges():
    #     print(edge)
    #     print(G[edge[0]][edge[1]]['prob'])
    #     n += 1
    #     print(n)

    # spring layout graph plot
    # pos = nx.spring_layout(G)
    # plt.figure(figsize=(12, 12))
    # nx.draw_networkx(G, pos)
    # plt.show()

    # circular layout graph plot
    # pos = nx.circular_layout(G)
    # plt.figure(figsize=(12, 12))
    # nx.draw_networkx(G, pos)
    # plt.show()
