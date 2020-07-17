import numpy as np


class PricingEnvironment:
    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward

    def round_price(self, pulled_arm):
        reward = self.reward
        reward_price = self.price[pulled_arm] * reward
        return reward_price

