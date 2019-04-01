import matplotlib.pyplot as plt
from algorithms.UCB1.UCB1Learner import *
from algorithms.Environment import *

prices = np.array([300, 350, 400, 450, 500])

n_arms = len(prices)

p = np.array([0.9, 0.8, 0.7, 0.4, 0.2])

counts = np.zeros(n_arms)
values = np.zeros(n_arms)

print("Prob :" + str(p))
print("Counts :" + str(counts))
print("Values :" + str(values))

T = 100

n_experiments = 200

ucb1_rewards_per_experiment = []

assumed_optimal_price = prices[2]
opt = np.array([assumed_optimal_price])

for e in range(0, n_experiments):
    print("Prob :" + str(p))
    print("Counts :" + str(counts))
    print("Values :" + str(values))

    env = Environment(n_arms=n_arms, probabilities=p, price=prices)
    ucb1_learner = UCB1Learner(counts, values)
    ucb1_rewards = []

    if (e % (n_experiments/100)) == 0:
        loading = e/(n_experiments/100)
        print(str(loading) + '%')

    for t in range(0, T):

        pulled_arm = ucb1_learner.pull_arm()
        reward = env.round(pulled_arm)
        reward_price = env.round_price(pulled_arm)
        ucb1_learner.update(pulled_arm, reward_price)

        # ucb1_rewards.append(reward_price)

    ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
    print(ucb1_learner.collected_rewards)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ucb1_rewards_per_experiment, axis=0)), 'r')

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot((np.mean(ucb1_rewards_per_experiment, axis=0)), 'r')
plt.show()
