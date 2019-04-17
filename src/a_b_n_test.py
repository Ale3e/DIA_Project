import numpy as np
import matplotlib.pyplot as plt
from algorithms.a_b_n.data import *

p = np.array([0.0263, 0.0193, 0.0129, 0.0061, 0.0012])
price = list(range(325, 450, 25))
N_A = 10000
N_B = 10000
ab_data = []
best = np.random.randint(1, 5)
ab_summary = []
for n in range(len(p)):
    print(best, n)
    if best == n:
        continue
    ab_data.append(generate_data(N_A, N_B, p[best], p[n], days=365))
    ab_summary.append(ab_data[n].pivot_table(values='converted', index='group', aggfunc=np.sum))
    ab_summary[n]['total'] = ab_data[n].pivot_table(values='converted', index='group', aggfunc=lambda x: len(x))
    ab_summary[n]['rate'] = ab_data[n].pivot_table(values='converted', index='group')
    if ab_summary[n]['converted'][0] * price[best] < ab_summary[n]['converted'][1] * price[n]:
        best = n
print("Best ", best)
total = []
converted = []
cr = []
fig, ax = plt.subplots(figsize=(12, 6))
x = []
y = []
tot = []
for n in range(len(p)):
    total.append(ab_summary[n]['total'][1])
    converted.append(ab_summary[n]['converted'][1])
    cr.append(ab_summary[n]['rate'][1])
    x.append(np.linspace(converted[n] - 49, converted[n] + 50, 100))
    y.append(scs.binom(total[n], cr[n]).pmf(x[n]))
    tot.append(converted[n]*price[n])
    if n == best:
        ax.bar(x[n], y[n], alpha=0.5)
    ax.plot(x[n], y[n], alpha=0.5)
array_legend = ['Price 325 - Total = ' + str(tot[0]), 'Price 350 - Total = ' + str(tot[1]),
                'Price 375 - Total = ' + str(tot[2]), 'Price 400 - Total = ' + str(tot[3]),
                'Price 425 - Total = ' + str(tot[4])]
ax.legend(array_legend)
plt.xlabel('converted')
plt.ylabel('probability')
plt.show()
