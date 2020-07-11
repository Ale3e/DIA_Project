import numpy as np
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
import tqdm
from errors import *

def draw_demand_curve (n_simulazioni, probabilities, color, label):
    '''
    :param n_simulazioni:
    :param probabilities: needs to be an array of len(probs)= 8
    :param color:
    :return: plots the demand curve for a given probs array
    '''

    if len(probabilities)!= 8:
        raise ProbsLenError

    reward_count = dict.fromkeys(prices, 0)
    probs = probabilities
    print(probs)

    range_prob = {range(300, 325): probs[0],
                  range(325, 350): probs[1],
                  range(350, 375): probs[2],
                  range(375, 400): probs[3],
                  range(400, 425): probs[4],
                  range(425, 450): probs[5],
                  range(450, 475): probs[6],
                  range(475, 500): probs[7]
                  }
    range_prices = {range(300, 325): 300,
                    range(325, 350): 325,
                    range(350, 375): 350,
                    range(375, 400): 375,
                    range(400, 425): 400,
                    range(425, 450): 425,
                    range(450, 475): 450,
                    range(475, 500): 475
                    }
    for n in tqdm.tqdm(range(n_simulazioni)):
        rand_pick = np.random.default_rng().integers(low=300, high=500)
        switch_prob_ranges = {k: v for rng, v in range_prob.items() for k in rng}
        switch_price_ranges = {k: v for rng, v in range_prices.items() for k in rng}
        prob = switch_prob_ranges[rand_pick]
        ranges = switch_price_ranges[rand_pick]
        print('Pick: {} Prob: {} and Range: {}'.format(rand_pick, prob, ranges))

        if np.random.rand() <= prob:
            reward_count[ranges] += 1

    print(reward_count)
    values = []
    values = list(reward_count.values())
    print(type(values))

    sns.distplot(values, bins=5, color=color, label=label)




if __name__ == "__main__":

    plt.style.use('seaborn')  # pretty matplotlib plots
    plt.rcParams['figure.figsize'] = (12, 8)

    ### discrete interval values ###

    prices = np.array([300, 325, 350, 375, 400, 425, 450, 475])


    p1 = np.array([0.045, 0.04, 0.0325, 0.0275, 0.0250, 0.02, 0.0175, 0.0125])
    p2 = np.array([0.035, 0.03, 0.0225, 0.0175, 0.015, 0.01, 0.0075, 0.0025])
    p3 = np.array([0.037, 0.032, 0.0245, 0.0195, 0.0170, 0.0120, 0.0095, 0.0045])
    p_aggregate = np.array((p1 + p2 + p3) / 3.0)
    print(p_aggregate)

    N_sample = 10000

    draw_demand_curve(N_sample, p1, 'yellow', 'p1')
    draw_demand_curve(N_sample, p2, 'orange', 'p2')
    draw_demand_curve(N_sample, p3, 'pink', 'p3')
    draw_demand_curve(N_sample, p_aggregate, 'green', 'aggregate')
    plt.show()


    ### continuous random values ###

    c = 1.25
    lam = 0.85
    ax = plt.subplots(1, 1)
    x = np.linspace(300, 500, 1000)
    f1 = ss.norm.pdf(x, loc=400, scale=10)
    f2 = ss.norm.pdf(x, loc=450, scale=10)

    f3 = np.convolve(f1, f2)


    # plt.plot(x, f1, color='blue')
    # plt.plot(x, f2, color='green')

    # sns.distplot(f1, color='green')
    # sns.distplot(f2, color='blue')
    # sns.distplot(f3)
    # sns.distplot(f11, bins=50)
    # sns.distplot(f21, bins=50)
    # sns.distplot(f31, bins=50)

    # sns.kdeplot(p1, color='green')
    # sns.kdeplot(p2, color='yellow')
    # sns.kdeplot(p3, color='pink')
    # sns.kdeplot((p1+p2), color='blue')

    # plt.hist(np.dot(p1,prices), bins=prices)
    # plt.show()

    # plt.bar(prices, p1)
    # plt.xticks(prices)
    # plt.yticks(p1)
    # plt.show()


# # Initialize click-through rate and signup rate dictionaries
# ct_rate = {'low':0.01, 'high':np.random.uniform(low=0.01, high=1.2*0.01)}
# su_rate = {'low':0.2, 'high':np.random.uniform(low=0.2, high=1.2*0.2)}
#
# def get_signups(cost, ct_rate, su_rate, sims):
#     lam = np.random.normal(loc=100000, scale=2000, size=sims)
#     # Simulate impressions(poisson), clicks(binomial) and signups(binomial)
#     impressions = np.random.poisson(lam=lam)
#     clicks = np.random.binomial(n=impressions, p=ct_rate[cost])
#     signups = np.random.binomial(n=impressions, p=su_rate[cost])
#     return signups
#
# print("Simulated Signups = {}".format(get_signups('high', ct_rate, su_rate, 1)))