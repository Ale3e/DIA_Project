from graph import generate_graph, weight_nodes, weight_edges, get_probabilities
from greedy import greedy_celf
from enviroment import *
from information_cascade import *
import numpy as np
import time
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
from TS_learner import *

if __name__ == "__main__":
    features = [0.1, 0.08, 0.05, 0.02]

    graph = generate_graph(100, 5, 0.1, 1234)
    graph = weight_edges(graph, features)
    graph = weight_nodes(graph)

    budget = 7.5
    delta = 0.5

    greedy_N_simulations = 1000

    # optimal with greedy_celf#

    start_time = time.time()
    greedy = []
    greedy = greedy_celf(graph, budget, delta)
    opt_seeds = sorted(greedy[1])
    spread_cumulative = []

    for n in range(greedy_N_simulations):
        IC = information_cascade(graph, opt_seeds)[0]
        spread_cumulative.append(IC)

    opt_spread = np.mean(spread_cumulative)

    print('Time for optimal greedy simulation: {} '.format(time.time() - start_time))
    print('Seeds: {}'.format(sorted(opt_seeds)))
    print('Optimal spread: {} \n'.format(round(float(opt_spread), 3)))

    # UCB_Learner

    spreads = []
    cumulative_spreads = []
    true_probs = get_probabilities(graph)
    env = Environment(graph)
    ts_learner = TSLearner(graph, budget)

    N_simulations = 100
    T = 200

    for t in tqdm.tqdm(range(T)):
        start_time = time.time()
        super_arm = ts_learner.pull_superarm()
        reward = env.round(super_arm)
        ts_learner.update(super_arm, reward)

        estimated_seeds = greedy_celf(ts_learner.graph, budget)[1]

        for n in range(N_simulations):
            IC = information_cascade(graph, estimated_seeds)[0]
            cumulative_spreads.append(IC)
        means_spread = np.mean(cumulative_spreads)
        means_spread = round(means_spread, 3)
        spreads.append(means_spread)
        print('Spread: {}'.format(means_spread))
        print('Time for iteration {} : {}'.format(t, time.time() - start_time))

    print('Opt-spread: {}'.format(opt_spread))
    print('Spreads: {}'.format(spreads))
    regret = np.abs(opt_spread - spreads)

    ### PLOT ###
    plt.style.use('seaborn')  # pretty matplotlib plots
    plt.rcParams['figure.figsize'] = (12, 8)
    opt_spreads = []
    for t in range(T): opt_spreads.append(opt_spread)



    # print(np.cumsum(np.abs((opt_spread - spreads))))
    # plt.plot(np.cumsum(np.abs((opt_spread - spreads))))
    plt.plot(spreads, color='blue', label='TS')
    plt.plot(opt_spreads, color='red', label='opt')
    plt.xlabel('t')
    plt.ylabel('Spread')
    plt.title('TS_CMAB')
    plt.legend()
    plt.show()

    print('True probabilities: {}'.format(true_probs))
    print('Estimated probabilities: {}'.format(list(ts_learner.get_estimated_probabilities().values())))
