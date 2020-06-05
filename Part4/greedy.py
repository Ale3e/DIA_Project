from graph import generate_graph, weight_nodes, weight_edges
from information_cascade import information_cascade
import networkx as nx
import numpy as np
import tqdm
import time


def celf(graph, budget, delta=0.95):
    """
    Cost efficient lazy forward algorithm, by Leskovec et al. (2007)
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """

    start_time = time.time()
    seed = []
    remaining_budget = budget
    epsilon = 0.1

    return (    )


if __name__ == "__main__":

    features = [0.10, 0.08, 0.05, 0.02]

    graph = generate_graph(100, 5, 0.05, 123)
    graph = weight_edges(graph, features)
    graph = weight_nodes(graph)

    budget = 10

    start_time = time.time()
    seeds = []
    remaining_budget = budget
    epsilon = 0.1
    delta = 0.9

    # evaluate each node in the graph for marginal increase in greedy algorithm
    marginal_gain = dict.fromkeys(graph.nodes, 0)
    nodes_left_to_evaluate = set(marginal_gain.keys())

    print(nodes_left_to_evaluate)
    print(type(nodes_left_to_evaluate))
    print(graph.nodes(data='cost'))
    print(type((graph.nodes(data='cost'))))
    print(graph.nodes.data('cost'))
    var = min(graph.nodes(data='cost'))
    print(var)

    while remaining_budget >= 0 and remaining_budget >= min(graph.nodes(data='cost')) and nodes_left_to_evaluate:

        for n in nodes_left_to_evaluate:

            cost = graph.nodes[n]['cost']
            n_simulations = int((1 / (epsilon ** 2)) * np.log(len(seeds + [n]) + 1) * np.log(1 / delta))
            IC_cumulative = 0.0

            for simulation in range(n_simulations):
                node_list = []
                IC_result = information_cascade(graph, n)[0]
                IC_cumulative += IC_result

            spread = round(((IC_cumulative / n_simulations) / cost), 3)
            marginal_gain[n] = spread

        spread_max = max(marginal_gain.values())
        index_max = list(marginal_gain.keys())[list(marginal_gain.values()).index(spread_max)]
        remaining_budget -= graph.nodes[index_max]['cost']
        seeds.append(index_max)

        nodes_left_to_evaluate.remove(index_max)
        marginal_gain.pop(index_max)

    # prendi best nodo da aggiungere ai seeds (max(marginal_gain)
    # togliere nodo aggiunto ai seed dalla lista @node_left_to_evaluate
    # ripeti

    # print(min(graph.nodes(data='cost')))

    # while remaining_budget >=0 and remaining_budget >= min(graph.nodes(data='cost')):

    print(seeds)
    print(marginal_gain)
    print(round(sum(marginal_gain.values()) / len(marginal_gain), 3))
