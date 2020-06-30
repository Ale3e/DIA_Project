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
        reward = self.reward
        reward_price = self.price[pulled_arm] * reward
        return reward_price

