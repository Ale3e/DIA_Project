import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt


class ABTest:

    def __init__(self, p1, p2, alpha=0.05, beta=0.8):
        self.p1 = p1
        self.p2 = p2
        self.alpha = alpha
        self.beta = beta

    def Z_test(self, x1, x2, price1, price2):

        mu1 = np.mean(x1) * price1
        mu2 = np.mean(x2) * price2
        n1 = int(len(x1))
        n2 = int(len(x2))

        y = ((n1 * mu1) + (n2 * mu2)) / (n1 + n2)
        var1 = self.p1 * (1 - self.p1)
        var2 = self.p2 * (1 - self.p2)

        # z = (mu1 - mu2) / np.sqrt(y * (1 - y) * ((1/n1)+(1/n2)))
        z = (mu1 - mu2) / np.sqrt(((var1 / n1) + (var2 / n2)))
        p_val = 1 - ss.norm.cdf(z)

        print('alpha = {}'.format(self.alpha))
        print('Z: {}'.format(z))
        print('p_val = {}'.format(p_val))

        if (p_val < self.alpha):
            print('Since p-value is lower than alpha, reject null hypothesis')
            return 1
        else:
            print('P-value greater than alpha, can not reject null hypothesis')
            return 0

    def calculate_sample_size(self):

        z_alpha = ss.norm.ppf(1 - self.alpha)
        z_beta = abs(ss.norm.ppf(self.beta))
        var = self.p1 * (1 - self.p1) + self.p2 * (1 - self.p2)
        min_var = abs(self.p1 - self.p2)
        n_samples = (((z_alpha + z_beta) ** 2) * var) / (min_var ** 2)

        return int(n_samples)

    def collect_samples(self, n_samples):

        x1 = [np.random.binomial(1, self.p1) for _ in range(0, int(n_samples / 2))]
        x2 = [np.random.binomial(1, self.p2) for _ in range(0, int(n_samples / 2))]
        return x1, x2




if __name__ == '__main__':
    np.random.seed(12)
    # p aggregate
    p = [0.0263, 0.0193, 0.0129, 0.0061, 0.0012]
    # prices = [1, 50, 75, 100, 125]
    # prices = {0.0263: 325, 0.0193: 350, 0.0129: 375, 0.0061: 400, 0.0012: 425}
    prices = [325, 350, 375, 400, 425]

    # p *10 aggregate
    # p = [0.263, 0.193, 0.129, 0.061, 0.012]
    # prices = {0.263: 325, 0.193: 350, 0.129: 375, 0.061: 400, 0.012: 425}


    reward = []

    # n_experiments = 100
    # opt = p[0]
    # best_price = prices[0]

    p = p[::-1]
    prices = prices[::-1]

    p_control = p.pop()
    price_control = prices.pop()
    print(p)
    print(prices)

    while p:

        p_test = p.pop()
        price_test = prices.pop()

        ab_tester = ABTest(p_control, p_test)
        control_group = []
        test_group = []

        n_samples = ab_tester.calculate_sample_size()
        print('Sample size: {}'.format(n_samples))
        control_group, test_group = ab_tester.collect_samples(n_samples)

        print('Testing {} vs {} control group'.format(price_test, price_control))
        mu_test_greater_than_control = ab_tester.Z_test(control_group, test_group, p_control, p_test)

        if (mu_test_greater_than_control == 1):
            print('Best price: {}\n'.format(price_control))
            best_price = price_control
        else:
            print('Best price: {}\n'.format(price_test))
            p_control = p_test
            price_control = price_test
            best_price = price_test



    print('No more ')

    # while p:
    #
    #     rew_control = []
    #     rew_test = []
    #     test = p.pop()
    #
    #     print('Testing {} vs {}'.format(control, test))
    #     ab_tester = ABNTest(p1=control, p2=test, alpha=0.05)
    #     n_samples = ab_tester.calculate_sample_size2()
    #     print('N samples: {}'.format(n_samples))
    #
    #     x1, x2 = ab_tester.collect_samples_2(n_samples)
    #
    #     rew_control.append(x1)
    #     rew_test.append(x2)
    #     winner = ab_tester.best_candidate(rew_control, rew_test, prices[control], prices[test])
    #     mean_control = np.mean(rew_test)
    #
    #     mean_test = np.mean(rew_test)
    #     reward += n_samples * [(mean_control * prices[control] + mean_test * prices[test]) / 2]
    #     plt.plot(reward, 'r')
    #
    #     if (winner == 1):
    #         print('Best price: {}'.format(control))
    #         best_price = control
    #     else:
    #         print('Best price: {}'.format(test))
    #         control = test
    #         best_price = test
    #
    #     # reward += n_samples * [(control*prices[control] + test*prices[test])/2]
    #     plt.plot(reward, 'r')
    #
    # a = [np.random.binomial(1, best_price) for _ in range(0, 300)]
    # reward += len(a) * [np.mean(a) * prices[best_price]]
    # clairvoyant = 800 * [best_price * prices[best_price]]
    # plt.plot(reward, 'r', label='AVG Reward')
    # plt.plot(clairvoyant, 'b--', label='Clairvoyant')
    # plt.legend()
    # plt.show()
