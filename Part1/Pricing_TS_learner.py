from pricingEnvironment import *
import numpy as np
import matplotlib.pyplot as plt
import copy

from Part1.pricingEnvironment import PricingEnvironment


class TSLearner:

    def __init__(self, n_arms, marginal_profit=[], sliding_window=False):

        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = x = [0.0 for i in range(n_arms)]
        self.collected_rewards = np.array([])
        self.beta_parameters = np.ones((n_arms, 2))
        self.marginal_profit = marginal_profit


    def pull_arm(self):

        idx_gain = []
        for i in range(self.n_arms):
            idx_gain.append((np.random.beta(self.beta_parameters[i, 0], self.beta_parameters[i, 1]))\
                            *self.marginal_profit[i])

        max = np.max(idx_gain)
        idx_max = idx_gain.index(max)
        return idx_max

    def update(self, pulled_arm, reward):

        self.t += 1
        self.rewards_per_arm[pulled_arm]+=(reward*self.marginal_profit[pulled_arm])
        self.collected_rewards += reward*self.marginal_profit[pulled_arm]
        self.beta_parameters[pulled_arm, 0] += reward
        self.beta_parameters[pulled_arm, 1] += (1.0 - reward)



if __name__ == '__main__':

    n_arms = 4
    n_experiments = 100
    T = 365

    p = np.array([0.0363, 0.03, 0.023, 0.012])
    marginal_profit = [325, 350, 375, 400]
    prices = {0.0363: 325, 0.03: 350, 0.023: 375, 0.012: 400}
    opt = p[1]

    ts_learner = TSLearner(n_arms, marginal_profit)
    print(ts_learner.marginal_profit)

    env = PricingEnvironment(n_arms, p)

    for time in range(T):

        for customer in range(n_experiments):

            pulled_arm = ts_learner.pull_arm()
            print('Pulled arm: {} with price {}'.format(pulled_arm, ts_learner.marginal_profit[pulled_arm]))
            reward = env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)
    print('REWARD: {}'.format(np.sum(ts_learner.collected_rewards)))

