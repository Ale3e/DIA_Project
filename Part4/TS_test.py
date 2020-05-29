import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

plt.style.use('seaborn')  # pretty matplotlib plots
plt.rcParams['figure.figsize'] = (12, 8)


def plot_beta(x_range, a, b, mu=0, sigma=1, cdf=False, **kwargs):
    """
    Plots the f distribution function for a given x range, a and b
    If mu and sigma are not provided, standard beta is plotted
    If cdf=True cumulative distribution is plotted
    Passes any keyword arguments to matplotlib plot function
    """
    x = x_range
    if cdf:
        y = ss.beta.cdf(x, a, b, mu, sigma)
    else:
        y = ss.beta.pdf(x, a, b, mu, sigma)
    plt.plot(x, y, **kwargs)


plays = np.array([100, 20, 20])
wins = np.array([32, 8, 5])
num_samples = 1000000

p_A_sample = np.random.beta(wins[0], plays[0] - wins[0], num_samples)
p_B_sample = np.random.beta(wins[1], plays[1] - wins[1], num_samples)
p_C_sample = np.random.beta(wins[2], plays[2] - wins[2], num_samples)

mab_wins = np.array([0.0, 0.0, 0.0])

for i in range(num_samples):
    winner = np.argmax([p_A_sample[i], p_B_sample[i], p_C_sample[i]])
    mab_wins[winner] += 1

mab_wins = mab_wins / num_samples

print(mab_wins)


x = np.linspace(0, 1, 1000)

plt.subplot(121)    # subplot() command specifies numrows, numcols, plot_number
plot_beta(x, wins[0], (plays[0] - wins[0]), 0, 1, color='red', lw=2, ls='-', alpha=0.5, label='pdf')
plot_beta(x, wins[1], (plays[1] - wins[1]), 0, 1, color='blue', lw=2, ls='-', alpha=0.5, label='pdf')
plot_beta(x, wins[2], (plays[2] - wins[2]), 0, 1, color='green', lw=2, ls='-', alpha=0.5, label='pdf')

plt.subplot(122)
plot_beta(x, wins[0], (plays[0] - wins[0]), 0, 1, cdf=True, color='red', lw=2, ls='-', alpha=0.5, label='pdf')
plot_beta(x, wins[1], (plays[1] - wins[1]), 0, 1, cdf=True, color='blue', lw=2, ls='-', alpha=0.5, label='pdf')
plot_beta(x, wins[2], (plays[2] - wins[2]), 0, 1, cdf=True, color='green', lw=2, ls='-', alpha=0.5, label='pdf')
plt.legend()
plt.show()

