from pricingEnvironment import *
import numpy as np
import matplotlib.pyplot as plt
import copy


class TSLearner:

    def __init__(self, n_arms, marginal_profit=[], sliding_window=False):

        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [0.0 for i in range(n_arms)]
        self.collected_rewards = np.array([])
        self.beta_parameters = np.ones((n_arms, 2))
        self.marginal_profit = marginal_profit


    def pull_arm(self):
        thetas = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        thetas = thetas * self.marginal_profit
        best_arm = np.argmax(thetas)
        return best_arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.rewards_per_arm[pulled_arm] += self.marginal_profit[pulled_arm] * reward
        self.collected_rewards += self.marginal_profit[pulled_arm] * reward
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + (1.0 - reward)



if __name__ == '__main__':

    n_arms = 4
    n_experiments = 100
    T = 365


    # p = np.array([0.363, 0.3, 0.23, 0.12])
    p = np.array([0.63, 0.4, 0.22, 0.12])
    marginal_profit = [2.5, 5.0, 7.5, 10]
    opt_idx = 2

    ts_learner = TSLearner(n_arms, marginal_profit)

    env = PricingEnvironment(n_arms, p)
    daily_rewards_TS = []

    for time in range(T):
        rewards_TS = []

        for customer in range(n_experiments):

            pulled_arm_TS = ts_learner.pull_arm()
            reward_TS = env.round(pulled_arm_TS)
            ts_learner.update(pulled_arm_TS, reward_TS)
            profit_TS = reward_TS * marginal_profit[pulled_arm_TS]
            rewards_TS.append(profit_TS)

        print('Day {}, reward TS: {} '.format(time, np.sum(rewards_TS)))
        daily_rewards_TS.append(np.sum(rewards_TS))

    print('Yearly profit with TS: {}'.format(np.sum(daily_rewards_TS)))

    ### PLOT ###

    plt.style.use('seaborn')  # pretty matplotlib plots
    plt.rcParams['figure.figsize'] = (12, 8)
    opt = []
    for t in range(T): opt.append((marginal_profit[opt_idx] * p[opt_idx]) * n_experiments)

    plt.plot((daily_rewards_TS), color='green', label='TS')
    plt.plot((opt), color='red', label='opt')
    plt.xlabel('t')
    plt.ylabel('Profit')
    plt.title('Profit comparison')
    plt.legend()
    plt.show()

    regret_TS = [np.abs(x1 - x2) for (x1, x2) in zip(opt, daily_rewards_TS)]

    plt.plot(np.cumsum(regret_TS), color='green', label='TS')
    plt.xlabel('t')
    plt.ylabel('Regret')
    plt.title('Regret Comparison')
    plt.legend()
    plt.show()


