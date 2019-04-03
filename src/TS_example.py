import matplotlib.pyplot as plt

from algorithms.Environment import *
from algorithms.thompson_sampling.TSLearner import *
from algorithms.GreedyLearner import *

price = list(range(300, 500, 25))
print(price)

n_arms = len(price)
print(n_arms)

p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
print(p)

assumed_optimal_price_bin = p[0]
optBin = np.array([assumed_optimal_price_bin])

assumed_optimal_price = price[0]
opt = np.array([assumed_optimal_price])

T = 365

n_experiments = 500
ts_bin_rewards_per_experiment = []
ts_rewards_per_experiment = []

for e in range(0, n_experiments):

    env = Environment(n_arms=n_arms, probabilities=p, price=price)

    ts_bin_learner = TSLearner(n_arms=n_arms)
    ts_bin_learner.initialize(price)

    ts_learner = TSLearner(n_arms=n_arms)
    ts_learner.initialize(price)

    if (e % (n_experiments/100)) == 0:
        loading = e/(n_experiments/100)
        print(str(loading) + '%')

    for t in range(0, T):

        # Thompson Sampling Binomial

        pulled_arm = ts_bin_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_bin_learner.update(pulled_arm, reward)

        # Thompson Sampling Gaussian

        # pulled_arm = ts_learner.pull_normal_arm()
        # reward = env.round(pulled_arm)
        # reward_price = env.round_price(pulled_arm)
        # ts_learner.update_price(pulled_arm, reward_price)

    ts_bin_rewards_per_experiment.append(ts_bin_learner.collected_rewards)
    # ts_rewards_per_experiment.append(ts_learner.collected_rewards_price)


# Plotting

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_bin_rewards_per_experiment, axis=0)), 'g')
# plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.legend(["TS_BIN", "TS"])

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.mean(ts_bin_rewards_per_experiment, axis=0), 'g')
# plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.legend(["TS_BIN", "TS"])
plt.show()

