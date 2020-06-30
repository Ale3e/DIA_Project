import matplotlib.pyplot as plt
from algorithms.Non_Stationary_Environment import *
from algorithms.thompson_sampling.SWTSLearner import *

# Environment variable
price = list(range(325, 450, 25))
print(price)
n_arms = len(price)
print(n_arms)
p = np.array([[0.0312, 0.0224, 0.014, 0.0068, 0.0016],
              [0.0312, 0.0224, 0.014, 0.0068, 0.0016],
              [0.0245, 0.0157, 0.0112, 0.005, 0.0007],
              [0.0262, 0.0166, 0.0123, 0.0055, 0.0011]])
print(p)
assumed_optimal_price = price[1]
opt = np.array([assumed_optimal_price])
T = 365*2
n_experiments = 5000

# TS variable
ts_rewards_per_experiment = []
# SWTS variable
swts_rewards_per_experiment = []
window_size = int(np.sqrt(T))

for e in range(0, n_experiments):

    ts_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T, price=price)
    ts_learner = TSLearner(n_arms=n_arms, price=price)

    swts_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T, price=price)
    swts_learner = SWTSLearner(n_arms=n_arms, price=price, window_size=window_size)

    if (e % (n_experiments/100)) == 0:
        loading = e/(n_experiments/100)
        print(str(loading) + '%')

    for t in range(0, T):

        pulled_arm = ts_learner.pull_arm()
        reward = ts_env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        pulled_arm = swts_learner.pull_arm()
        reward = swts_env.round(pulled_arm)
        swts_learner.update(pulled_arm, reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    swts_rewards_per_experiment.append(swts_learner.collected_rewards_ns)


ts_instantaneous_regret = np.zeros(T)
swts_instantaneous_regret = np.zeros(T)

n_phases = len(p)
phases_len = T/n_phases
opt_per_phases = p.max(axis=1)*325
optimum_per_round = np.zeros(T)

for i in range(0, n_phases):
    optimum_per_round[int(i*phases_len): int((i+1)*phases_len)] = opt_per_phases[i]
    ts_instantaneous_regret[int(i*phases_len): int((i+1)*phases_len)] = opt_per_phases[i] - np.mean(ts_rewards_per_experiment, axis=0)[int(i*phases_len)]
    swts_instantaneous_regret[int(i*phases_len): int((i+1)*phases_len)] = opt_per_phases[i] - np.mean(swts_rewards_per_experiment, axis=0)[int(i*phases_len)]


plt.figure(0)
plt.ylabel("Reward")
plt.xlabel("t")
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(swts_rewards_per_experiment, axis=0), 'b')
plt.plot(optimum_per_round, '--k')
plt.legend(["TS", "SW-TS", "Optimum"])

plt.figure(1)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(ts_instantaneous_regret), 'r')
plt.plot(np.cumsum(swts_instantaneous_regret), 'b')
plt.legend(["TS", "SW-TS"])

plt.show()
