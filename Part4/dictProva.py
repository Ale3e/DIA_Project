import numpy as np

if __name__ == "__main__":

    node_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    marginal_gain = dict.fromkeys(node_list, 0)
    nodes_left_to_evaluate = set(marginal_gain.keys())


    print(nodes_left_to_evaluate)
    print(type(nodes_left_to_evaluate))

    tries = 5
    i = 0

    while (i < tries):

        for n in nodes_left_to_evaluate:
            marginal_gain[n] = np.random.rand()
            #print(marginal_gain[i])

        arg_max = max(marginal_gain.values())
        index_max = list(marginal_gain.keys())[list(marginal_gain.values()).index(arg_max)]
        nodes_left_to_evaluate.remove(index_max)

        print(arg_max)
        print(index_max)

        marginal_gain.pop(index_max)
        print(marginal_gain)
        i += 1