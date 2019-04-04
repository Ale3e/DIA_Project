from algorithms.Learner import *


class TSLearner(Learner):

    def __init__(self, n_arms, price):
        super().__init__(n_arms)
        # Beta_parameters is the variable in which we save the parameter alpha (positive) and beta (negative)
        self.beta_parameters = np.ones((n_arms, 2))
        self.price = price

    def pull_arm(self):
        a = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
<<<<<<< HEAD
        idx = np.argmax(a * self.price)
=======
        price = np.array(list(range(300, 500, 25)))
        idx = np.argmax(a * price)
<<<<<<< Updated upstream
=======
<<<<<<< HEAD
>>>>>>> 6d5b50a6a3309856e927fa7bebedfea639cdb29a
=======
>>>>>>> 6d5b50a6a3309856e927fa7bebedfea639cdb29a
>>>>>>> Stashed changes
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        # if reward is 1, sum 1 to alpha column, if it is 0 sum 1 to beta column
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward
        # The reward is correlate to the price of the chosen arm
        reward = reward * self.price[pulled_arm]
        self.update_observations(pulled_arm, reward)
