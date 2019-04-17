from .TSLearner import *


class SWTSLearner(TSLearner):
    def __init__(self, n_arms, price, window_size):
        super().__init__(n_arms, price)
        self.window_size = window_size

    def update(self, pulled_arm, reward):
        self.t += 1

        cum_rew = np.sum(self.rewards_per_arm[pulled_arm][-self.window_size:])
        n_rounds_arm = len(self.rewards_per_arm[pulled_arm][-self.window_size:])

        self.beta_parameters[pulled_arm, 0] = cum_rew + 1.0
        self.beta_parameters[pulled_arm, 1] = n_rounds_arm - cum_rew + 1.0

        self.update_observations(pulled_arm, reward)
        reward = reward * self.price[pulled_arm]
        self.update_observations_ns(pulled_arm, reward)
