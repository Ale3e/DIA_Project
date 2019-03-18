import numpy as np
import  matplotlib.pyplot as plt

from Environment import *
from TS_Learner import *
from Greedy_Learner import *

n_arms = 4

p = np.array([0.15, 0.1, 0.1, 0.35])
opt = p[3]

T = 300

n_experiments = 1000
ts_rewards_per_experiment = []
gr_rewards_per_experiment = []

print("Start algorithms...")

for e in range(0, n_experiments):
    env = Environment(n_arms=n_arms, probabilities = p)
    ts_learner = TS_Learner(n_arms = n_arms)
    gr_learner = Greedy_Learner(n_arms = n_arms)
    ts_rewards = []
    gs_rewards = []
    for t in range(0, T):
        #Thompson Sampling
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        ts_rewards.append(reward)

        #Greedy Learner
        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm, reward)

        gs_rewards.append(reward)

    ts_rewards_per_experiment.append(ts_rewards)
    gr_rewards_per_experiment.append(gs_rewards)

print("Start drawing...")

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment, axis=0)), 'g')
plt.legend(["TS", "Greedy"])
plt.show()
