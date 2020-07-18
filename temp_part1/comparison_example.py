import numpy as np
import matplotlib.pyplot as plt
from Environment import *
from TS_Learner import *
from Greedy_Learner import *
from UCB1_Learner import *

from DIA.Pricing.Environment import Environment
from DIA.Pricing.TS_Learner import TS_Learner
from DIA.Pricing.UCB1_Learner import UCB1_Learner

np.random.seed(10)
n_arms = 4
#p = np.array([0.15, 0.1, 0.1, 0.35])

p = np.array([0.363, 0.3, 0.23, 0.12])

prices = np.array([325, 350, 375,400])



opt = p[0]*prices[0]
T = 365

n_experiments = 1000

ts_rewards_per_experiment = []
gr_rewards_per_experiment = []

for e in range(0,n_experiments):
    print(e*100/1000)
    env = Environment(n_arms=n_arms, probabilities=p)
    ts_learner = TS_Learner(n_arms=n_arms,prices=prices)
    gr_learner = UCB1_Learner(n_arms=n_arms,prices=prices)
    for i in range(0,T):
        #Thomposon Sampling Learner
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm,reward)

        #Greedy Learner
        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm,reward)
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    gr_rewards_per_experiment.append(gr_learner.collected_rewards)

plt.figure(0)
plt.xlabel('T')
plt.ylabel('Regret')

plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment,axis=0)),'r')
plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment,axis=0)),'g')
plt.legend(['TS','UCB-1'])
plt.show()


print(np.mean(ts_rewards_per_experiment))
print(np.mean(gr_rewards_per_experiment))
plt.figure(0)
plt.xlabel('t')
plt.ylabel('Reward')
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(gr_rewards_per_experiment, axis=0), 'g')
plt.plot(T * [opt], '--k')
plt.legend(['TS','UCB-1'])
plt.show()