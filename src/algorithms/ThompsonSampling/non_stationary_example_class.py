import numpy as np
import matplotlib.pyplot as plt
from Non_Stationary_Environment import *
from TS_Learner import *
from SWTS_Learner import *

n_arms = 17
p = np.array([[0.0022, 0.01, 0.0159, 0.0175, 0.0189, 0.0190, 0.02, 0.0198, 0.0180, 0.0161, 0.0145, 0.0129, 0.0113, 0.0084, 0.0058, 0.0029, 0.0016],
              [0.0012, 0.015, 0.029, 0.025, 0.029, 0.020, 0.02, 0.0198, 0.0180, 0.0191, 0.0185, 0.0119, 0.0113, 0.0084, 0.0058, 0.0029, 0.0016],
              [0.0002, 0.001, 0.0059, 0.0075, 0.0089, 0.0090, 0.01, 0.0098, 0.0080, 0.0061, 0.0045, 0.0029, 0.0013, 0.0004, 0.0008, 0.0009, 0.0006]])

T = 365

n_experiments = 1000
ts_rewards_per_experiment = []
swts_rewards_per_experiment = []
window_size = int(np.sqrt(T))

for e in range(0, n_experiments):
    ts_env = Non_Stationary_Environment(n_arms, probabilities=p, horizon=T)
    ts_learner = TS_Learner(n_arms=n_arms)

    swts_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T)
    swts_learner = SWTS_Learner(n_arms=n_arms, window_size=window_size)

    for t in range(0, T):
        pulled_arm = ts_learner.pull_arm()
        reward = ts_env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        pulled_arm = swts_learner.pull_arm()
        reward = swts_env.round(pulled_arm)
        swts_learner.update(pulled_arm, reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    swts_rewards_per_experiment.append(swts_learner.collected_rewards)

ts_instantaneous_regret = np.zeros(T)
swts_instantaneous_regret = np.zeros(T)
n_phases = len(p)
phases_len = T/n_phases
opt_per_phases = p.max(axis=1)
optimum_per_round = np.zeros(T)

for i in range(0, n_phases):
    optimum_per_round[int(i*phases_len) : int((i+1)*phases_len)] = opt_per_phases[i]
    ts_instantaneous_regret[int(i*phases_len) : int((i+1)*phases_len)] = opt_per_phases[i] - np.mean(ts_rewards_per_experiment, axis=0)[int(i*phases_len)]
    swts_instantaneous_regret[int(i*phases_len) : int((i+1)*phases_len)] = opt_per_phases[i] - np.mean(swts_rewards_per_experiment, axis=0)[int(i*phases_len)]

plt.figure(0)
plt.ylabel("Reward")
plt.xlabel("t")
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(swts_rewards_per_experiment, axis=0), 'b')
plt.plot(optimum_per_round, '--k')
plt.legend(["TS", "SW-TS", "Optimum"])
plt.show()

plt.figure(1)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(ts_instantaneous_regret), 'r')
plt.plot(np.cumsum(swts_instantaneous_regret), 'b')
plt.legend(["TS", "SW-TS"])
plt.show()
