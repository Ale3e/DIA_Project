import matplotlib.pyplot as plt

from algorithms.Environment import *
from algorithms.thompson_sampling.TSLearner import *
from algorithms.GreedyLearner import *

price = list(range(300, 500, 25))
n_arms = len(price)

p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

print(p)

assumed_optimal_price = p[4]
opt = np.array([assumed_optimal_price])

T = 365

n_experiments = 200
ts_rewards_per_experiment = []

print("Start algorithms...")

for e in range(0, n_experiments):
    env = Environment(n_arms=n_arms, probabilities=p, price=price)
    ts_learner = TSLearner(n_arms=n_arms)

    if (e % (n_experiments/100)) == 0:
        loading = e/(n_experiments/100)
        print(str(loading) + '%')

    for t in range(0, T):

        # Thompson Sampling Binomial

        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        # Thompson Sampling Gaussian

        # pulled_arm = ts_learner.pull_normal_arm()
        # ts_learner.update_price(pulled_arm, reward_price)
        # reward_price = env.round_price(pulled_arm)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    # ts_rewards_per_experiment.append(ts_learner.collected_rewards_price)

print("Start drawing...")

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.legend(["TS"])

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.legend(["TS"])
plt.show()

