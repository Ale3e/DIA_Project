import numpy as np
import matplotlib.pyplot as plt
from Non_Stationary_Environment import *
from TS_Learner import *
from SWTS_Learner import *
from UCB1_Learner import *
from SWUCB1_Learner import *

np.random.seed(10)
n_arms = 5
#p = np.array(([[0.15, 0.1, 0.2, 0.35],
 #               [0.32, 0.21, 0.2, 0.35],
  #             [0.5, 0.1, 0.1, 0.15],
   #           [0.8, 0.21, 0.1, 0.15]]))


p = np.array([[0.312, 0.224, 0.140, 0.068, 0.016],[0.312, 0.224, 0.140, 0.068, 0.016],[0.245, 0.157, 0.112, 0.050, 0.007],[0.262, 0.166, 0.123, 0.055, 0.011]])


T = 364
n_experiments = 2500
swts_reward_per_experiment = []
ts_reward_per_experiment = []
swucb_reward_per_experiment = []
ucb_reward_per_experiment = []
window_size = int(np.sqrt(n_experiments)*n_arms)
prices = np.array([325, 350, 375,400,425])
n_phases = len(p)
phases_len = int(T/n_phases)

def generate_reward_phase(probabilities,pulled_arm,phase):
    probabilities = probabilities[phase,:]
    reward = np.random.binomial(1, probabilities[pulled_arm])
    return reward


for e in range(0,n_experiments):
    print(e)
    ts_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T)
    ts_learner = TS_Learner(n_arms=n_arms,prices=prices)

    swts_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T)
    swts_learner = SWTS_Learner(n_arms=n_arms, window_size=window_size,prices=prices)


    for t in range(0,T):
        current_phase = int(t / phases_len)
        pulled_arm = ts_learner.pull_arm()
        reward = ts_env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        pulled_arm = swts_learner.pull_arm()
        reward = generate_reward_phase(p, pulled_arm, current_phase)
        swts_learner.update(pulled_arm, reward)

    ts_reward_per_experiment.append(ts_learner.collected_rewards)
    swts_reward_per_experiment.append(swts_learner.collected_rewards)
    #ucb_reward_per_experiment.append(ucb_learner.collected_rewards)
    #swucb_reward_per_experiment.append(swucb_learner.collected_rewards)

ts_instantaneus_regret = np.zeros(T)
swts_instantaneus_regret = np.zeros(T)
#ucb_instantaneus_regret = np.zeros(T)
#swucb_instantaneus_regret = np.zeros(T)
n_phases = len(p)
print(n_phases)
phases_len = int(T/n_phases)
opt_per_phases = p.max(axis=1)*prices[0]
print(opt_per_phases)
opt_per_round = np.zeros(T)

for i in range(0, n_phases):
    opt_per_round[i*phases_len : (i+1)*phases_len+1] = opt_per_phases[i]
    ts_instantaneus_regret[i*phases_len : (i+1)*phases_len] = opt_per_phases[i] - np.mean(ts_reward_per_experiment,axis=0)[i*phases_len : (i+1)*phases_len]
    swts_instantaneus_regret[i*phases_len : (i+1)*phases_len] = opt_per_phases[i] - np.mean(swts_reward_per_experiment,axis=0)[i*phases_len : (i+1)*phases_len]
    #ucb_instantaneus_regret[i*phases_len : (i+1)*phases_len] = opt_per_phases[i] - np.mean(ucb_reward_per_experiment,axis=0)[i*phases_len : (i+1)*phases_len]
    #swucb_instantaneus_regret[i*phases_len : (i+1)*phases_len] = opt_per_phases[i] - np.mean(swucb_reward_per_experiment,axis=0)[i*phases_len : (i+1)*phases_len]




#In the first figure we show the reward
plt.figure(0)
plt.xlabel('t')
plt.ylabel('Reward')
plt.plot(np.mean(ts_reward_per_experiment, axis=0), 'r')
plt.plot(np.mean(swts_reward_per_experiment, axis=0), 'b')
#plt.plot(np.mean(ucb_reward_per_experiment, axis=0), 'g')
#plt.plot(np.mean(swucb_reward_per_experiment, axis=0), 'y')
plt.plot(opt_per_round, '--k')
plt.legend(['SW-TS','TS','Optimum'])
plt.show()


#In the second plot we show the regret
plt.figure(1)
plt.xlabel ('t')
plt.ylabel('Regret')
plt.plot(np.cumsum(ts_instantaneus_regret), 'r')
plt.plot(np.cumsum(swts_instantaneus_regret), 'b')
#plt.plot(np.cumsum(ucb_instantaneus_regret), 'g')
#plt.plot(np.cumsum(swucb_instantaneus_regret), 'y')
plt.legend(['SW-TS','TS'])
plt.show()
