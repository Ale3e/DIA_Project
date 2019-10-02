import matplotlib.pyplot as plt
import random
from algorithms.Environment import *
from algorithms.thompson_sampling.TSLearner import *
from algorithms.GreedyLearner import *

# Environment variable
price = list(range(325, 450, 25))
print(price)
n_arms = len(price)
print(n_arms)
# p aggregate
p = np.array([0.0263, 0.0193, 0.0129, 0.0061, 0.0012])
# p student
ps = np.array([0.0251, 0.0151, 0.0118, 0.0054, 0.0009])
# p worker
pw = np.array([0.0298, 0.021, 0.0135, 0.0063, 0.0014])
# p retired
pr = np.array([0.0183, 0.0102, 0.009, 0.0053, 0.0004])
print(p)
random.randint(1, 4)

cnt = 1



assumed_optimal_price = price[1]
opt = np.array([assumed_optimal_price])
T = 365
n_experiments = 1000
# TS Variable
ts_rewards_per_experiment = []

for e in range(0, n_experiments):

    env = Environment(n_arms=n_arms, probabilities=p, price=price)
    ts_learner = TSLearner(n_arms=n_arms, price=price)

    # The if is only for showing loading percentage
    if (e % (n_experiments/100)) == 0:
        loading = e/(n_experiments/100)
        print(str(loading) + '%')

    for t in range(0, T):

        # Thompson Sampling routine
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

    # At the end of the T cycle, is necessary to update the reward for each experiment
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)


# Plotting

plt.figure("Regret")
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'b')
plt.legend(["TS"])

plt.figure("Reward")
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'b')
plt.legend(["TS"])
plt.show()

