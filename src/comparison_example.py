import numpy as np
import matplotlib.pyplot as plt

from algorithms.ThompsonSampling.Environment import *
from algorithms.ThompsonSampling.TS_Learner import *
from algorithms.ThompsonSampling.Greedy_Learner import *

n_arms = 17

p = np.array([0.0022, 0.01, 0.0159, 0.0175, 0.0189, 0.0190, 0.02, 0.0198, 0.0180, 0.0161, 0.0145, 0.0129, 0.0113, 0.0084, 0.0058, 0.0029, 0.0016])
opt = p[8]

T = 365

n_experiments = 5000
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

