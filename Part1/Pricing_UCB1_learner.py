from pricingEnvironment import *
import numpy as np
import matplotlib.pyplot as plt
import copy


class UCB1Learner:

    def __init__(self, n_arms, marginal_profit=[], sliding_window=False):

        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = x = [0.0 for i in range(n_arms)]
        self.collected_rewards = np.array([])
        self.marginal_profit = marginal_profit
        self.empirical_means_no_bound = copy.deepcopy(marginal_profit)
        self.empirical_means = copy.deepcopy(marginal_profit)

        self.sliding_window = sliding_window
        self.n_of_samples = np.ones(n_arms)


    def pull_arm(self):


        return np.argmax(self.empirical_means)


    def update(self, pulled_arm, reward):

        self.t += 1
        self.rewards_per_arm[pulled_arm]+=(reward*self.marginal_profit[pulled_arm])
        self.collected_rewards +=(reward*self.marginal_profit[pulled_arm])
        if self.sliding_window == 0:
            self.n_of_samples[pulled_arm] += 1
        for i in range(self.n_arms):
            self.empirical_means_no_bound[i] = np.sum(self.rewards_per_arm[i]) / self.n_of_samples[i]
            self.empirical_means[i] = self.empirical_means_no_bound[i] + np.sqrt((2 * np.log(self.t+1)) / self.n_of_samples[i])





if __name__ == '__main__':

    n_arms = 4
    n_experiments = 100
    T = 365

    p = np.array([0.0363, 0.03, 0.023, 0.012])
    marginal_profit = [325, 350, 375, 400]

    ucb_learner = UCB1Learner(n_arms, marginal_profit)
    print(ucb_learner.marginal_profit)

    env = PricingEnvironment(n_arms, p)

    for time in range(T):

        for customer in range(n_experiments):

            pulled_arm = ucb_learner.pull_arm()
            reward = env.round(pulled_arm)
            print('Pulled arm: {} with price {}, reward is {}'.format(pulled_arm, ucb_learner.marginal_profit[pulled_arm], reward))
            ucb_learner.update(pulled_arm, reward)
    print('REWARD: {}'.format(np.sum(ucb_learner.collected_rewards)))

