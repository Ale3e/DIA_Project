import matplotlib.pyplot as plt

from algorithms.Environment import *
from algorithms.thompson_sampling.TSLearner import *
from algorithms.ucb1.UCB1Learner import *

# Environment variable
# price = list(range(300, 525, 25))
price = list(range(325, 450, 25))
print(price)
n_arms = len(price)
print(n_arms)
# p = np.array([0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001])
# p = np.array([0.0183, 0.0179, 0.0173, 0.0157, 0.0149, 0.0133, 0.0122, 0.0108, 0.0091])
p = np.array([0.0263, 0.0193, 0.0129, 0.0061, 0.0012])
print(p)
assumed_optimal_price = price[1]
opt = np.array([assumed_optimal_price])
T = 365
n_experiments = 50000

# TS Variable
ts_rewards_per_experiment = []

# UCB1 Variable
counts = np.zeros(n_arms)
values = np.zeros(n_arms)
ucb1_rewards_per_experiment = []

for e in range(0, n_experiments):

    env = Environment(n_arms=n_arms, probabilities=p, price=price)
    ts_learner = TSLearner(n_arms=n_arms, price=price)
    ucb1_learner = UCB1Learner(counts=counts, values=values, price=price)

    ucb1_learner.initialize(env=env)

    # The if is only for showing loading percentage
    if (e % (n_experiments/100)) == 0:
        loading = e/(n_experiments/100)
        print(str(loading) + '%')

    for t in range(0, T):

        # Thompson Sampling routine
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        # UBC1 routine
        pulled_arm = ucb1_learner.pull_arm()
        reward = env.round(pulled_arm)
        ucb1_learner.update(pulled_arm, reward)

    # At the end of the T cycle, is necessary to update the reward for each experiment
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)


# Plotting

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - ucb1_rewards_per_experiment, axis=0)), 'b')
plt.legend(["TS", "UCB1"])

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.plot((np.mean(ucb1_rewards_per_experiment, axis=0)), 'b')
plt.axhline(y=(0.0193 * 350), color='black', linestyle='--')
plt.legend(["TS", "UCB1"])
plt.show()
