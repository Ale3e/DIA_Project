from algorithms.Learner import *


class TSLearner(Learner):

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        self.normal_parameters = np.zeros((n_arms, 4))
        self.values = [[] for _ in range(n_arms)]

    def initialize(self, price):
        self.normal_parameters[:, 0] = price
        self.normal_parameters[:, 2] = price

    def pull_arm(self):
        a = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        price = np.array(list(range(300, 500, 25)))
        idx = np.argmax(a * price)
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward

        self.update_observations(pulled_arm, reward)

    def pull_normal_arm(self):
        a = np.random.normal(self.normal_parameters[:, 0], self.normal_parameters[:, 1])
        idx = np.argmax(a)
        return idx

    def update_price(self, pulled_arm, reward_price):
        self.t += 1
        self.normal_parameters[pulled_arm, 2] = self.normal_parameters[pulled_arm, 2] + reward_price
        self.normal_parameters[pulled_arm, 3] = self.normal_parameters[pulled_arm, 3] + 1
        self.values[pulled_arm].append(reward_price)
        self.normal_parameters[pulled_arm, 0] = np.mean([self.values[pulled_arm]])
        self.normal_parameters[pulled_arm, 1] = np.std([self.values[pulled_arm]])

        self.update_observations_price(pulled_arm, reward_price)
