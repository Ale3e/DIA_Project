from Part1.Pricing_TS_learner import *
from Part1.Pricing_UCB1_learner import *

if __name__ == '__main__':

    n_arms = 4
    n_experiments = 1000
    T = 365

    p = np.array([0.363, 0.3, 0.23, 0.12])
    marginal_profit = [2.5, 5.0, 7.5, 10]
    opt_idx = 3

    ucb_learner = UCB1Learner(n_arms, marginal_profit)
    ts_learner = TSLearner(n_arms, marginal_profit)

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

            pulled_arm_TS = ts_learner.pull_arm()
            reward_TS = env.round(pulled_arm_TS)
            ts_learner.update(pulled_arm_TS, reward_TS)
            profit_TS = reward_TS * marginal_profit[pulled_arm_TS]
            rewards_TS.append(profit_TS)

        print('Day {}, reward UCB1: {}, reward TS: {} '.format(time, np.sum(profit_UCB1), np.sum(profit_TS)))
        daily_rewards_UCB1.append(np.sum(reward_UCB1))
        daily_rewards_TS.append(np.sum(reward_TS))

    print('Yearly profit with UCB1: {}'.format(np.sum(daily_rewards_UCB1)))
    print('Yearly profit with UCB1: {}'.format(np.sum(daily_rewards_TS)))



    ### PLOT ###

    plt.style.use('seaborn')  # pretty matplotlib plots
    plt.rcParams['figure.figsize'] = (12, 8)
    opt = []
    for t in range(T): opt.append(marginal_profit[opt_idx]*p[opt_idx])

    plt.plot(np.cumsum(daily_rewards_UCB1), color='blue', label='UCB1')
    plt.plot(np.cumsum(daily_rewards_TS), color='green', label='TS')
    # plt.plot(spreads_LUCB, color='orange', label='LinUCB')
    plt.plot(opt, color='red', label='opt')
    plt.xlabel('t')
    plt.ylabel('Profit')
    plt.title('Profit comparison')
    plt.legend()
    plt.show()

    regret_UCB1 = np.abs(opt - daily_rewards_UCB1)
    regret_TS = np.abs(opt - daily_rewards_TS)

    plt.plot(np.cumsum(regret_UCB1), color='blue', label='UCB1')
    plt.plot(np.cumsum(regret_TS), color='green', label='TS')
    # plt.plot(np.cumsum(regret_LUCB), color='orange', label='LinUCB')
    plt.xlabel('t')
    plt.ylabel('Regret')
    plt.title('Regret Comparison')
    plt.legend()
    plt.show()





