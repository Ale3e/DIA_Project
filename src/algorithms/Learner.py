import numpy as np


class Learner:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        # -- array con risultati (0/1) divisi per arm per ogni esperimento --
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        # -- array con risultati (0/1) * rispettivo presso divisi per arm per ogni esperimento --
        self.rewards_price_per_arm = x = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])
        self.collected_rewards_price = np.array([])
        self.price = np.array(list(range(300, 500, 25)))

    def update_observations(self, pulled_arm, reward):
        # -- prima --
        # self.rewards_per_arm[pulled_arm].append(reward * self.price[pulled_arm])
        # -- dopo (ma cosa serve?) --
        self.rewards_per_arm[pulled_arm].append(reward)

        self.collected_rewards = np.append(self.collected_rewards, reward * self.price[pulled_arm])


    # def update_observations_price(self, pulled_arm, reward_price):
    #     self.rewards_price_per_arm[pulled_arm].append(reward_price)
    #     self.collected_rewards_price = np.append(self.collected_rewards_price, reward_price)
