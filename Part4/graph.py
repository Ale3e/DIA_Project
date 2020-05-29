import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def generate_graph(n_nodes, k, p, seed):
    """
    Returns a small world network with n nodes,
    starting with each node connected to its k nearest neighbors,
    and rewiring probability p
    """

    graph = nx.newman_watts_strogatz_graph(n_nodes, k, p, seed=seed)

    nx.set_node_attributes(graph, 0, 'id')
    nx.set_node_attributes(graph, 0.0, 'cost')
    nx.set_node_attributes(graph, 0, 'active')
    nx.set_node_attributes(graph, 0, 'inactive')
    nx.set_node_attributes(graph, 0, 'susceptible')
    nx.set_edge_attributes(graph, 0.0, 'prob')

    prob_matrix = np.zeros(n_nodes)

    return graph


def weight_edges(graph, f):
    """ Sets a probability to each edge based on a linear combination of 4 features
    Input: graph -- networkx Graph object
    f -- list of features probability
    """

    for edge in graph.edges():
        p = np.random.binomial(1, 0.5, size=len(f))
        graph[edge[0]][edge[1]]['prob'] = round(sum(x * y for x, y in zip(p, f)), 2)

    return graph

def weight_nodes(graph):
    #todo
    return graph


feat = [0.10, 0.08, 0.05, 0.02]
p = np.random.binomial(1, 0.5, size=len(feat))

print(p)
q = round(sum(x * y for x, y in zip(p, feat)), 2)
print(q)


G = generate_graph(1000, 5, 0.05, 123)
G_prob = weight_edges(G, feat)

print(nx.info(G))

print(G.adj[0])



# set ids, print node attributes and of adj nodes
# for g in range(G.number_of_nodes()):
#     G.nodes[g]['id'] = g
#
# for g in range(G.number_of_nodes()):
#     print("\nNode :")
#     print(G.nodes[g])
#     print("Has the following adjacent nodes: ")
#     for n in G.neighbors(g):
#         print(n)
#         print(G.nodes[n])


# print edges and their probability
n = 0
for e in G.edges():
    print(e)
    print(G[e[0]][e[1]]['prob'])
    n += 1
    print(n)

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
