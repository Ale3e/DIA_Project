import numpy as np


class Environment:
    def __init__(self, n_arms, probabilities, price):
        self.n_arms = n_arms
        self.probabilities = probabilities
        self.price = price
        self.reward = 0

    def round(self, pulled_arm):
        self.reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return self.reward

    def round_price(self, pulled_arm):
        #self.reward = abs(np.random.normal(mean, std))
        reward_price = self.price[pulled_arm] * self.reward
        return reward_price

