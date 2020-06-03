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

    graph = generate_graph(10000, 5, 0.05, 123)
    graph = weight_edges(graph, features)
    graph = weight_nodes(graph)

    budget = 100

    start_time = time.time()
    seeds = []
    remaining_budget = budget
    epsilon = 0.1
    delta = 0.2

    # evaluate each node in the graph for marginal increase in greedy algorithm
    marginal_gain = dict.fromkeys(graph.nodes, 0)
    nodes_left_to_evaluate = set(marginal_gain.keys())
    set_len = len(nodes_left_to_evaluate)
    print(set_len)
    print(type(set_len))


    #while remaining_budget >=0 and remaining_budget >= min(graph.nodes(data='cost')):

    for node in tqdm.tqdm(range(set_len)):

        cost = graph.nodes[node]['cost']
        n_simulations = int((1 / (epsilon ** 2)) * np.log(len(seeds + [node]) + 1) * np.log(1 / delta))
        IC_cumulative = 0.0

        for n in range(n_simulations):
            node_list = []
            IC_result = information_cascade(graph, node)[0]
            IC_cumulative += IC_result

        spread = round(((IC_cumulative/n_simulations) / cost), 5)
        marginal_gain[node] = spread

    #prendi best nodo da aggiungere ai seeds (max(marginal_gain)
    #togliere nodo aggiunto ai seed dalla lista @node_left_to_evaluate
    #ripeti



    #print(min(graph.nodes(data='cost')))

    #while remaining_budget >=0 and remaining_budget >= min(graph.nodes(data='cost')):

    print(marginal_gain)
    print(round(sum(marginal_gain.values())/len(marginal_gain), 5))







