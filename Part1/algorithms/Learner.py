import numpy as np


class Learner:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        # Array of [0,1] per arm for each experiment
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        self.rewards_per_arm_ns = x = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])
        self.collected_rewards_ns = np.array([])

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)

    def update_observations_ns(self, pulled_arm, reward):
        self.rewards_per_arm_ns[pulled_arm].append(reward)
        self.collected_rewards_ns = np.append(self.collected_rewards_ns, reward)
