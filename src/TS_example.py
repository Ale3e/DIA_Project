import matplotlib.pyplot as plt

from algorithms.Environment import *
from algorithms.thompson_sampling.TSLearner import *
from algorithms.GreedyLearner import *

price = list(range(300, 500, 25))
print(price)

n_arms = len(price)
print(n_arms)

p = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01])
print(p)

assumed_optimal_price = price[2]
opt = np.array([assumed_optimal_price])

T = 10

<<<<<<< Updated upstream
=======
<<<<<<< HEAD
<<<<<<< HEAD
n_experiments = 500

=======
=======
>>>>>>> 6d5b50a6a3309856e927fa7bebedfea639cdb29a
>>>>>>> Stashed changes
n_experiments = 5
ts_bin_rewards_per_experiment = []
>>>>>>> 6d5b50a6a3309856e927fa7bebedfea639cdb29a
ts_rewards_per_experiment = []

for e in range(0, n_experiments):

    env = Environment(n_arms=n_arms, probabilities=p, price=price)
    ts_learner = TSLearner(n_arms=n_arms, price=price)

<<<<<<< HEAD
    # The if is only for showing loading percentage
=======
    ts_bin_learner = TSLearner(n_arms=n_arms)
    ts_bin_learner.initialize(price)

    ts_learner = TSLearner(n_arms=n_arms)
<<<<<<< Updated upstream
=======
<<<<<<< HEAD
=======

    # -- inizializza Normale --
    # ts_learner.initialize(price)
>>>>>>> 6d5b50a6a3309856e927fa7bebedfea639cdb29a
>>>>>>> Stashed changes

    # -- inizializza Normale --
    # ts_learner.initialize(price)

>>>>>>> 6d5b50a6a3309856e927fa7bebedfea639cdb29a
    if (e % (n_experiments/100)) == 0:
        loading = e/(n_experiments/100)
        print(str(loading) + '%')

    for t in range(0, T):

        # Thompson Sampling routine
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

    # At the end of the T cycle, is necessary to update the reward for each experiment
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)


# Plotting

plt.figure("Regret")
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'b')
plt.legend(["TS"])

plt.figure("Reward")
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'b')
plt.legend(["TS"])
plt.show()

