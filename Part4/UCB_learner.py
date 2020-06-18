from graph import generate_graph, weight_nodes, weight_edges, get_probabilities
from greedy import greedy_celf
from enviroment import *
from information_cascade import *
import numpy as np
import time
import networkx as nx

class UCBLearner:
    def __init__(self, graph, budget):
        self.graph = graph
        self.empirical_mean = dict.fromkeys(self.graph.edges, 1)
        self.empirical_mean_no_bound = dict.fromkeys(self.graph.edges, 1)
        self.cumulative_reward = dict.fromkeys(self.graph.edges, 0)
        self.T = dict.fromkeys(self.graph.edges, 0)
        self.t = 0
        self.budget = budget

        for node1, node2 in self.graph.edges():
            graph[node1][node2]['prob'] = 1



    def pull_superarm(self):
        '''
        get all edges from seeds as superarm
        '''

        superarm = set()
        seeds = []
        seeds = greedy_celf(self.graph, self.budget)[1]

        for seed in seeds:
            for u, v in graph.edges(seed):
                superarm.add((u, v))

        return superarm

    def update(self, pulled_arm, reward):
        self.t += 1
        graph = self.graph
        seeds = set(u for (u, v) in pulled_arm)
        reachable_nodes = set(seeds)

        #Build a graph containing only the activated edges that are reachable from at least one seed
        live_edge_g = nx.Graph()
        live_edge_g.add_edges_from(reward)
        live_edge_g.add_nodes_from(seeds)

        for seed in seeds:
            reachable_nodes = reachable_nodes.union(nx.descendants(live_edge_g, seed))

        unreachable_nodes = set(graph.nodes).difference(reachable_nodes)
        live_edge_g.remove_nodes_from(unreachable_nodes)


        #####-----####
        for node in live_edge_g.nodes:
            for u, v in graph.edges(node):
                self.T[(u, v)] += 1
                if (u, v) in live_edge_g.edges(node):
                    self.cumulative_reward[(u, v)] += 1
                bound = np.sqrt((3*np.log(self.t))/(2 * self.T[(u, v)]))
                self.empirical_mean_no_bound[(u, v)] = min(self.cumulative_reward[(u, v)] / self.T[(u, v)] , 1)
                self.empirical_mean[(u, v)] = min(self.cumulative_reward[(u, v)] / self.T[(u, v)] + bound, 1)

    def get_estimated_probabilities(self):
        for node1, node2 in self.graph.edges():
            graph[node1][node2]['prob'] = 1
        return self.empirical_mean_no_bound
    #todo: aggiornare le probabilità degli archi nel grafo con le estimed probs


if __name__ == "__main__":
    features = [0.1, 0.08, 0.05, 0.02]

    graph = generate_graph(100, 5, 0.1, 1234)
    graph = weight_edges(graph, features)
    graph = weight_nodes(graph)

    budget = 5
    delta = 0.95
    N_simulations = 1000


    #optimal with greedy_celf#


    start_time = time.time()
    greedy = []
    greedy = greedy_celf(graph, budget, delta)
    opt_seeds = sorted(greedy[1])
    spread_cumulative = []

    for n in range(N_simulations):
        IC = information_cascade(graph, opt_seeds)[0]
        spread_cumulative.append(IC)

    opt_spread = np.mean(spread_cumulative)

    print('Time for optimal greedy simulation: {} \n'.format(time.time() - start_time))
    print('Seeds: {}'.format(sorted(opt_seeds)))
    print('Optimal spread: {} \n'.format(round(float(opt_spread), 3)))



    #UCB_Learner

    spreads = []
    env = Environment(graph)
    ucb_learner = UCBLearner(graph, 5)
    superarms = ucb_learner.pull_superarm()





    for t in range(10):
        start_time = time.time()
        seeds = set()
        pulled_arm = ucb_learner.pull_superarm()
        reward = env.round(pulled_arm)
        ucb_learner.update(pulled_arm, reward)
        estimated_seeds = greedy_celf(ucb_learner.graph, 5)
        ucb_learner.append(
            information_cascade(graph, estimated_seeds, list(ucb_learner.get_estimated_probabilities().values()), mc=100))
        print('Time for iteration {} : {}'.format(t, time.time() - start_time))

    print(spreads)
    print('Opt-spread: {}'.format(opt - spreads))
    print(np.cumsum(np.abs((opt - spreads))))
    plt.plot(np.cumsum(np.abs((opt - spreads))))
    plt.show()

    print('True probabilities: {}'.format(probabilities))

    print('Estimated probabilities: {}'.format(list(learner.get_estimated_probabilities().values())))


