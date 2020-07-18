from pricingEnvironment import *
import numpy as np
import matplotlib.pyplot as plt
import copy


class UCB1Learner:

    def __init__(self, n_arms, marginal_profit=[], sliding_window=False):

        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [0.0 for i in range(n_arms)]
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
        self.rewards_per_arm[pulled_arm] += (reward*self.marginal_profit[pulled_arm])
        self.collected_rewards += reward*self.marginal_profit[pulled_arm]
        if self.sliding_window == 0:
            self.n_of_samples[pulled_arm] += 1
            self.empirical_means_no_bound[pulled_arm] = self.rewards_per_arm[pulled_arm] / self.n_of_samples[pulled_arm]
            self.empirical_means[pulled_arm] = self.empirical_means_no_bound[pulled_arm] + np.sqrt((2 * np.log(self.t+1)) / self.n_of_samples[pulled_arm])





if __name__ == '__main__':

    n_arms = 4
    n_experiments = 1000
    T = 365

    p = np.array([0.363, 0.3, 0.23, 0.12])
    marginal_profit = [2.5, 5.0, 7.5, 10]
    opt_idx = 2

    ucb_learner = UCB1Learner(n_arms, marginal_profit)
    env = PricingEnvironment(n_arms, p)
    daily_rewards_UCB1 = []
    daily_rewards_TS = []

    for time in range(T):

        rewards_UCB1 = []
        rewards_TS = []

        for customer in range(n_experiments):
            pulled_arm_UCB1 = ucb_learner.pull_arm()
            reward_UCB1 = env.round(pulled_arm_UCB1)
            ucb_learner.update(pulled_arm_UCB1, reward_UCB1)
            profit_UCB1 = reward_UCB1 * marginal_profit[pulled_arm_UCB1]
            rewards_UCB1.append(profit_UCB1)


        print('Day {}, reward UCB1: {} '.format(time, np.sum(rewards_UCB1)))
        daily_rewards_UCB1.append(np.sum(rewards_UCB1))

    print('Yearly profit with UCB1: {}'.format(np.sum(daily_rewards_UCB1)))

    ### PLOT ###

    plt.style.use('seaborn')  # pretty matplotlib plots
    plt.rcParams['figure.figsize'] = (12, 8)
    opt = []
    for t in range(T): opt.append((marginal_profit[opt_idx]*p[opt_idx])*n_experiments)


    plt.plot((daily_rewards_UCB1), color='blue', label='UCB1')
    plt.plot((opt), color='red', label='opt')
    plt.xlabel('t')
    plt.ylabel('Profit')
    plt.title('Profit comparison')
    plt.legend()
    plt.show()

    regret_UCB1 = [x1 - x2 for (x1, x2) in zip(opt, daily_rewards_UCB1)]

    plt.plot(np.cumsum(regret_UCB1), color='blue', label='UCB1')
    plt.xlabel('t')
    plt.ylabel('Regret')
    plt.title('Regret Comparison')
    plt.legend()
    plt.show()



