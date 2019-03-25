from algorithms.Learner import *


class UCB1Learner(Learner):
    def __init__(self, counts, values):
        self.counts = counts
        self.values = values

    def pull_arm(self):

        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)

        for arm in range(n_arms):
            bound = np.sqrt((2 * np.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bound

        idx = np.argmax(ucb_values)
        return idx

    def update(self, pulled_arm, reward):

        self.counts[pulled_arm] = self.counts[pulled_arm] + 1
        n = self.counts[pulled_arm]
        value = self.values[pulled_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[pulled_arm] = new_value
