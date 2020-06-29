import numpy as np
import networkx as nx
from graph import *
from greedy import *
from collections import defaultdict

if __name__ == "__main__":

    # node_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #
    # marginal_gain = dict.fromkeys(node_list, 0)
    # nodes_left_to_evaluate = set(marginal_gain.keys())
    #
    #
    # print(nodes_left_to_evaluate)
    # print(type(nodes_left_to_evaluate))
    #
    # tries = 5
    # i = 0
    #
    # while (i < tries):
    #
    #     for n in nodes_left_to_evaluate:
    #         marginal_gain[n] = np.random.rand()
    #         #print(marginal_gain[i])
    #
    #     arg_max = max(marginal_gain.values())
    #     index_max = list(marginal_gain.keys())[list(marginal_gain.values()).index(arg_max)]
    #     nodes_left_to_evaluate.remove(index_max)
    #
    #     print(arg_max)
    #     print(index_max)
    #
    #     marginal_gain.pop(index_max)
    #     print(marginal_gain)
    #     i += 1

    features = np.array([0.1, 0.08, 0.05, 0.02])
    graph = generate_graph(100, 5, 0.1, 1234)
    graph = weight_edges(graph, features)
    graph = weight_nodes(graph)
    budget = 5

    true_theta = np.random.dirichlet(np.ones(len(features)), size=1)

    # alpha = dict.fromkeys(graph.edges, 1)
    # beta = dict.fromkeys(graph.edges, 1)
    #
    # for node1, node2 in graph.edges():
    #     prob_ts = round(np.random.beta(alpha[(node1,node2)], beta[(node1,node2)]), 3)
    #     print(prob_ts)
    #     graph[node1][node2]['prob'] = prob_ts

    # superarm = set()
    # seeds = []
    # seeds = greedy_celf(graph, budget)[1]
    #
    # print('seeds are: {}'.format(seeds))
    #
    # for seed in seeds:
    #     for u, v in graph.edges():
    #         if (u == seed): superarm.add((u, v))
    #         if (v == seed) and (u, v) not in superarm: superarm.add((u, v))
    # print(superarm)
    #
    # for u,v in superarm:
    #     if graph[u][v] == None:
    #         raise Exception('({},{}) is not a valid edge of the graph')

    #round
    # success = dict.fromkeys(superarm, 0)
    # for (u, v) in success:
    #     success[(u, v)] = np.random.binomial(1, graph[u][v]['prob'])
    # print(success)
    # active_edges = [i for i in success if success[i] == 1]
    # print(active_edges)
    # print(enumerate(graph.edges))
    #print(success[0])
    # for i, edge in graph.edges:
    #     if success
    #active_edges = ([edge for i, edge in enumerate(graph.edges) if success[(i,edge)] == 1])
    # print(active_edges)