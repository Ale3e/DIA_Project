import numpy as np


class Graph:

    def __init__(self, n_nodes, features):
        self.theta = np.random.dirichlet(np.ones(features), size=1)          #features = n. of probabilities that sum up to 1
        self.node_features = np.random.binomial(1, 0.5, size=(n_arms, dim))
        self.p = np.zeros(n_arms)
        for armIDX in range(0, n_arms):                                 #assign the prob of each arm as
            self.p[armIDX] = np.dot(self.theta, self.arms_features[i])

    def __init__(self, n_nodes, features):
        self.prob_matrix = np.random.rand(dimension, dimension)
        self.prob_matrix = np.around(self.prob_matrix, decimals=3)
        for i in range(dimension):
            self.prob_matrix[i, i] = 0

### DA QUI IN POI OK ###

    def __init__(self, n_nodes, features):
        self.nodes = []
        self.edges = []
        self.prob_matrix = np.random.dirichlet(np.ones(features), size=1)


