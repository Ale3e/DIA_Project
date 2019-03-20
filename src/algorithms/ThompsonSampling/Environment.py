import numpy as np


class Environment():
    def __init__(self, n_arms, probabilities, price):
        self.n_arms = n_arms
        self.probabilities = probabilities
        self.price = price

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward

    def round_price(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        reward_price = self.price[pulled_arm] * self.probabilities[pulled_arm] * reward
        return reward_price

