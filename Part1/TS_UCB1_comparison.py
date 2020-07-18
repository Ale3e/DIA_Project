from Part1.Pricing_TS_learner import *
from Part1.Pricing_UCB1_learner import *

from Part1.pricingEnvironment import PricingEnvironment


if __name__ == '__main__':

    n_arms = 4

    n_experiments = 5000
    T = 365

    p = np.array([0.0363, 0.03, 0.023, 0.012])
    marginal_profit = [325, 350, 375, 400]
    opt_idx = 0


    ucb_learner = UCB1Learner(n_arms, marginal_profit)
    ts_learner = TSLearner(n_arms, marginal_profit)
    g_learner = GreedyLearner(n_arms, marginal_profit)

    env = PricingEnvironment(n_arms, p)
    daily_rewards_UCB1 = []
    daily_rewards_TS = []
    daily_rewards_g = []


    for time in range(T):

        rewards_UCB1 = []
        rewards_TS = []
        rewards_g = []

        for customer in range(n_experiments):

            pulled_arm_UCB1 = ucb_learner.pull_arm()
            reward_UCB1 = env.round(pulled_arm_UCB1)
            ucb_learner.update(pulled_arm_UCB1, reward_UCB1)
            profit_UCB1 = reward_UCB1 * marginal_profit[pulled_arm_UCB1]
            rewards_UCB1.append(profit_UCB1)

            pulled_arm_TS = ts_learner.pull_arm()
            reward_TS = env.round(pulled_arm_TS)
            ts_learner.update(pulled_arm_TS, reward_TS)
            profit_TS = reward_TS * marginal_profit[pulled_arm_TS]
            rewards_TS.append(profit_TS)


        print('Day {}, reward UCB1: {}, reward TS: {} '.format(time, np.sum(rewards_UCB1), np.sum(rewards_TS)))
        daily_rewards_UCB1.append(np.sum(rewards_UCB1))
        daily_rewards_TS.append(np.sum(rewards_TS))

    print('Yearly profit with UCB1: {}'.format(np.sum(daily_rewards_UCB1)))
    print('Yearly profit with TS: {}'.format(np.sum(daily_rewards_TS)))

        print('Day {}, reward UCB1: {}, reward TS: {} '.format(time, np.sum(rewards_UCB1), np.sum(rewards_TS)))
        daily_rewards_UCB1.append(np.sum(rewards_UCB1))
        daily_rewards_TS.append(np.sum(rewards_TS))
        daily_rewards_g.append(np.sum(rewards_g))

    print('Yearly profit with UCB1: {}'.format(np.sum(daily_rewards_UCB1)))
    print('Yearly profit with TS: {}'.format(np.sum(daily_rewards_TS)))
    print('Yearly profit with greedy: {}'.format(np.sum(daily_rewards_g)))

    ### PLOT ###

    plt.style.use('seaborn')  # pretty matplotlib plots

    plt.rcParams['figure.figsize'] = (12, 8)
    opt = []
    for t in range(T): opt.append((marginal_profit[opt_idx]*p[opt_idx])*n_experiments)


    plt.plot((daily_rewards_UCB1), color='blue', label='UCB1')
    plt.plot((daily_rewards_TS), color='green', label='TS')
    plt.plot(daily_rewards_g, color='orange', label='greedy')
    plt.plot((opt), color='red', label='opt')
    plt.xlabel('t')
    plt.ylabel('Profit')
    plt.title('Profit comparison')
    plt.legend()
    plt.show()

    regret_UCB1 = [np.abs(x1 - x2) for (x1, x2) in zip(opt, daily_rewards_UCB1)]
    regret_TS = [np.abs(x1 - x2) for (x1, x2) in zip(opt, daily_rewards_TS)]

    regret_g = [np.abs(x1 - x2) for (x1, x2) in zip(opt, daily_rewards_g)]


    plt.plot(np.cumsum(regret_UCB1), color='blue', label='UCB1')
    plt.plot(np.cumsum(regret_TS), color='green', label='TS')
    plt.plot(np.cumsum(regret_g), color='orange', label='greedy')
    plt.xlabel('t')
    plt.ylabel('Regret')
    plt.title('Regret Comparison')
    plt.legend()
    plt.show()





