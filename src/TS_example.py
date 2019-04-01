import matplotlib.pyplot as plt

from algorithms.Environment import *
from algorithms.thompson_sampling.TSLearner import *
from algorithms.GreedyLearner import *

price = list(range(300, 500, 25))
n_arms = len(price)

p = np.array([0.04, 0.038, 0.032, 0.025, 0.023, 0.021, 0.0125, 0.0075])

print(p)

assumed_optimal_price = price[4]
opt = np.array([assumed_optimal_price])

T = 365

n_experiments = 10000
ts_rewards_per_experiment = []

print("Start algorithms...")

for e in range(0, n_experiments):
    env = Environment(n_arms=n_arms, probabilities=p, price=price)
    ts_learner = TSLearner(n_arms=n_arms)
    ts_rewards = []

    if (e % (n_experiments/100)) == 0:
        loading = e/(n_experiments/100)
        print(str(loading) + '%')

    for t in range(0, T):

        # Thompson Sampling

        pulled_arm = ts_learner.pull_arm()
        # pulled_arm = ts_learner.pull_normal_arm()
        reward = env.round(pulled_arm)
        reward_price = env.round_price(pulled_arm)
        ts_learner.update(pulled_arm, reward)
        # ts_learner.update_price(pulled_arm, reward_price)

        ts_rewards.append(reward_price)

    ts_rewards_per_experiment.append(ts_rewards)
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

