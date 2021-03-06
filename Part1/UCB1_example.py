import matplotlib.pyplot as plt
from algorithms.ucb1.UCB1Learner import *
from algorithms.Environment import *

# Environment variable
price = list(range(150, 600, 100))
print(price)
n_arms = len(price)
print(n_arms)
p = np.array([0.938, 0.627, 0.423, 0.287, 0.091])
print(p)
assumed_optimal_price = price[1]
opt = np.array([assumed_optimal_price])
T = 365*50
n_experiments = 5000

# UCB1 Variable
counts = np.zeros(n_arms)
values = np.zeros(n_arms)
ucb1_rewards_per_experiment = []

for e in range(0, n_experiments):

    env = Environment(n_arms=n_arms, probabilities=p, price=price)
    ucb1_learner = UCB1Learner(counts=counts, values=values, price=price)

    # The if is only for showing loading percentage
    if (e % (n_experiments/100)) == 0:
        loading = e/(n_experiments/100)
        print(str(loading) + '%')

    # UBC1 routine init
    ucb1_learner.initialize(env=env)

    for t in range(0, T):

        # UBC1 routine
        pulled_arm = ucb1_learner.pull_arm()
        reward = env.round(pulled_arm)
        ucb1_learner.update(pulled_arm, reward)

    # At the end of the T cycle, is necessary to update the reward for each experiment
    ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)

# Plotting

plt.figure("Regret")
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ucb1_rewards_per_experiment, axis=0)), 'b')
plt.legend(["UCB1"])

plt.figure("Reward")
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.mean(ucb1_rewards_per_experiment, axis=0), 'b')
plt.legend(["UCB1"])
plt.show()
