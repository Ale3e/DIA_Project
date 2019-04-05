from algorithms.Learner import *


class UCB1Learner(Learner):
    def __init__(self, counts, values, price):
        super().__init__(len(counts))
        self.counts = counts
        self.values = values
        self.price = price

    def initialize(self, pulled_arm, reward):
        # First cycle update the value of counts and values
        self.counts[pulled_arm] = self.counts[pulled_arm] + 1
        n = self.counts[pulled_arm]
        value = self.values[pulled_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[pulled_arm] = new_value

    def pull_arm(self):
        n_arms = len(self.counts)
        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)
        # Compute the upper bound value for each arm
        for arm in range(n_arms):
            bound = np.sqrt((2 * np.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bound

        idx = np.argmax(ucb_values)
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        # Update the value of counts and values
        self.counts[pulled_arm] = self.counts[pulled_arm] + 1
        n = self.counts[pulled_arm]
        value = self.values[pulled_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[pulled_arm] = new_value
        # The reward is correlate to the price of the chosen arm
        reward = reward * self.price[pulled_arm]
        self.update_observations(pulled_arm, reward)
