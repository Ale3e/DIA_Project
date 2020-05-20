import numpy as np


class ProbMatrix:

    features = np.random.rand(4)
    features = np.around(features, decimals=3)

    def __init__(self, dimension):

        self.prob_matrix = np.random.rand(dimension, dimension)
        self.prob_matrix = np.around(self.prob_matrix, decimals=3)
        for i in range(dimension):
            self.prob_matrix[i, i] = 0


    def print(self):
        print(self.prob_matrix)

    def save_to_csv(self):
        self.prob_matrix.tofile("prob_matrix.csv", sep=",")


prob_sn_1 = ProbMatrix(50)
prob_sn_1.print()

prob_sn_1.save_to_csv()
print(prob_sn_1.features)