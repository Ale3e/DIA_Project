import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    features = [0.10, 0.08, 0.05, 0.20]

    # def __init__(self, n_nodes, features):
    #     self.theta = np.random.dirichlet(np.ones(features), size=1)          #features = n. of probabilities that sum up to 1
    #     self.node_features = np.random.binomial(1, 0.5, size=(n_arms, dim))
    #     self.p = np.zeros(n_arms)
    #     for armIDX in range(0, n_arms):                                 #assign the prob of each arm as
    #         self.p[armIDX] = np.dot(self.theta, self.arms_features[i])
    #
    # def __init__(self, n_nodes, features):
    #     self.prob_matrix = np.random.rand(dimension, dimension)
    #     self.prob_matrix = np.around(self.prob_matrix, decimals=3)
    #     for i in range(dimension):
    #         self.prob_matrix[i, i] = 0

    def __init__(self, n_nodes, features):
        self.nodes = []
        self.edges = []
        p = np.random.binomial(1, 0.5, size=(n_nodes, n_nodes))
        self.prob_matrix = np.zeros(n_nodes)
        for node_IDX in range(0, n_nodes):  # assign the prob of each node
            self.prob_matrix[node_IDX] = np.dot(features, self.arms_features[node_IDX])

    def weight_edges(G, f):
        ''' Sets a probability to each edge based on a linear combination of 4 features
        Input: G -- networkx Graph object
        f -- list of features probability
        '''


G = nx.newman_watts_strogatz_graph(100, 5, 0.05, seed=123)  # returns a small world network with n nodes, strating with
                                                            # each node connected to its k neares neighbors,
                                                            # and rewiring probability p

print(nx.info(G))



print(G.adj[0])

nx.set_node_attributes(G, 0, 'id')
nx.set_node_attributes(G, 0.0, 'cost')

#set ids, print node attributes and of adj nodes
for g in range(G.number_of_nodes()):
    G.nodes[g]['id'] = g


for g in range(G.number_of_nodes()):
    print("\nNode :")
    print(G.nodes[g])
    print("Has the following adjacent nodes: ")
    for n in G.neighbors(g):
        print(n)
        print(G.nodes[n])


n=0
for e in G.edges():
    print(e)
    n +=1
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
