import numpy as np
import matplotlib.pyplot as plt
from algorithms.ThompsonSampling.Non_Stationary_Environment import *
from algorithms.ThompsonSampling.TS_Learner import *
from algorithms.ThompsonSampling.SWTS_Learner import *

n_arms = 17
p = np.array([[0.0072, 0.015, 0.0209, 0.0225, 0.0239, 0.0240, 0.0250, 0.0248, 0.023, 0.0211, 0.0195, 0.0179, 0.0163, 0.0134, 0.0108, 0.0079, 0.0066],
              [0.0003, 0.0051, 0.011, 0.0125, 0.0139, 0.0140, 0.015, 0.0148, 0.013, 0.0111, 0.0095, 0.0079, 0.0064, 0.0038, 0.0012, 0.0001, 0.0001],
              [0.0005, 0.007, 0.0129, 0.0145, 0.0159, 0.016, 0.017, 0.0168, 0.0150, 0.0131, 0.0115, 0.0099, 0.0083, 0.0054, 0.0028, 0.0002, 0.0001]])

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

