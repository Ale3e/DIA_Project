import matplotlib.pyplot as plt

from algorithms.Environment import *
from algorithms.thompson_sampling.TSLearner import *
from algorithms.GreedyLearner import *

#n_arms = 5
#price = np.array([300, 325, 350, 375, 400, 425, 450, 475])
#p = np.array([0.04, 0.895, 0.0275, 0.0225, 0.02, 0.0150, 0.0125, 0.0075])

price = list(range(300, 500, 25))
n_arms = len(price)

p = []
a = np.random.uniform(1, 0, size=n_arms)
a = np.sort(a)
p = a[::-1]

p = np.array(p)

p = np.array([0.04, 0.038, 0.032, 0.025, 0.023, 0.021, 0.0125, 0.075])

print(p)

assumed_optimal_price = price[4]
opt = np.array([assumed_optimal_price])

T = 365

n_experiments = 1000
ts_rewards_per_experiment = []
gr_rewards_per_experiment = []

print("Start algorithms...")

for e in range(0, n_experiments):
    env = Environment(n_arms=n_arms, probabilities=p, price=price)
    ts_learner = TSLearner(n_arms=n_arms)
    gr_learner = GreedyLearner(n_arms=n_arms)
    ts_rewards = []
    gs_rewards = []
    for t in range(0, T):

        # Thompson Sampling

        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        reward_price = env.round_price(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        ts_rewards.append(reward_price)

        # Greedy Learner

        # pulled_arm = gr_learner.pull_arm()
        # reward = env.round(pulled_arm)
        # reward_price = env.round_price(pulled_arm)
        # gr_learner.update(pulled_arm, reward)
        #
        # gs_rewards.append(reward_price)

    ts_rewards_per_experiment.append(ts_rewards)
    # gr_rewards_per_experiment.append(gs_rewards)

print("Start drawing...")

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment, axis=0)), 'g')
plt.legend(["TS", "Greedy"])

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.cumsum(np.mean(ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(gr_rewards_per_experiment, axis=0)), 'g')
plt.legend(["TS", "Greedy"])
# plt.show()

