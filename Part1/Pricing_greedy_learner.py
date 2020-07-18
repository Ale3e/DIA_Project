from pricingEnvironment import *
import numpy as np
import matplotlib.pyplot as plt

class GreedyLearner:

    def __init__(self, n_arms, marginal_profit=[]):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = x = [0.0 for i in range(n_arms)]
        self.collected_rewards = np.array([])
        self.expected_rewards = np.zeros(n_arms)
        self.marginal_profit = marginal_profit

    def pull_arm(self):
        if (self.t < self.n_arms):
            return self.t
        idxs = np.argwhere(self.expected_rewards == self.expected_rewards.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.rewards_per_arm[pulled_arm] += self.marginal_profit[pulled_arm] * reward
        self.collected_rewards += self.marginal_profit[pulled_arm] * reward
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.t - 1) + (reward*self.marginal_profit[pulled_arm])) / self.t

if __name__ == '__main__':

    n_arms = 4
    n_experiments = 10
    T = 365

    # p = np.array([0.363, 0.3, 0.23, 0.12])
    p = np.array([0.63, 0.4, 0.22, 0.12])
    marginal_profit = [2.5, 5.0, 7.5, 10]
    opt_idx = 2

    g_learner = GreedyLearner(n_arms, marginal_profit)

    env = PricingEnvironment(n_arms, p)
    daily_rewards_g = []

    for time in range(T):
        rewards_g = []

        for customer in range(n_experiments):

            pulled_arm_g = g_learner.pull_arm()
            reward_g = env.round(pulled_arm_g)
            g_learner.update(pulled_arm_g, reward_g)
            profit_g = reward_g * marginal_profit[pulled_arm_g]
            rewards_g.append(profit_g)

        print('Day {}, reward greedy: {} '.format(time, np.sum(rewards_g)))
        daily_rewards_g.append(np.sum(rewards_g))

    print('Yearly profit with greedy: {}'.format(np.sum(daily_rewards_g)))

    ### PLOT ###

    plt.style.use('seaborn')  # pretty matplotlib plots
    plt.rcParams['figure.figsize'] = (12, 8)
    opt = []
    for t in range(T): opt.append((marginal_profit[opt_idx] * p[opt_idx]) * n_experiments)

    plt.plot(daily_rewards_g, color='green', label='greedy')
    plt.plot(opt, color='red', label='opt')
    plt.xlabel('t')
    plt.ylabel('Profit')
    plt.title('Profit comparison')
    plt.legend()
    plt.show()

    regret_g = [np.abs(x1 - x2) for (x1, x2) in zip(opt, daily_rewards_g)]

    plt.plot(np.cumsum(regret_g), color='green', label='greedy')
    plt.xlabel('t')
    plt.ylabel('Regret')
    plt.title('Regret Comparison')
    plt.legend()
    plt.show()