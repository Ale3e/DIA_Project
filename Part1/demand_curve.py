import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

if __name__ == "__main__":

    plt.style.use('seaborn')  # pretty matplotlib plots
    plt.rcParams['figure.figsize'] = (12, 8)

    fig, ax = plt.subplots(1, 1)

    a = 1.99
    mean, var, skew, kurt = ss.gamma.stats(a, moments='mvsk')

    x = np.linspace(ss.gamma.ppf(0.01, a),
                    ss.gamma.ppf(0.99, a), 100)
    ax.plot(x, ss.gamma.pdf(x, a),
            'r-', lw=2, alpha=0.6, label='gamma pdf')

    r = ss.gamma.rvs(a, size=10000)
    bins = 50
    ax.hist(r, bins=bins, edgecolor='black', density=True, histtype='stepfilled', alpha=0.5)
    ax.legend(loc='best', frameon=False)

    plt.show()


    # Create models from data
    def best_fit_distribution(data, bins=200, ax=None):
        """Model data by finding best fit distribution to data"""
        # Get histogram of original data
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0