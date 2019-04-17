import numpy as np
import matplotlib.pyplot as plt
import pandas as ps
from algorithms.a_b_n.data import *




p = np.array([0.0263, 0.0193, 0.0129, 0.0061, 0.0012])
# A is control; B is test
price = list(range(325, 450, 25))
N_A = 10000
N_B = 10000
N_C = 10000
ab_data = []
best = np.random.randint(0, 5)
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


# A_converted=ab_summary[0]['converted'][1]
# A_total=ab_summary[0]['total'][1]
# A_cr=ab_summary[0]['rate'][1]

total=[]
converted=[]
cr=[]

# total.append(ab_summary[0]['total'][0])
# cr.append(ab_summary[0]['converted'][0])
# converted.append(ab_summary[0]['rate'][0])
#
# xA = np.linspace(A_converted-49, A_converted+50, 100)
# yA = scs.binom(A_total, A_cr).pmf(xA)
fig, ax = plt.subplots(figsize=(12,6))
# ax.bar(xA, yA, alpha=0.5)
# ax.plot(xA, yA, alpha=0.5)
x=[]
y=[]
for n in range(len(p)):
    total.append(ab_summary[n]['total'][1])
    converted.append(ab_summary[n]['converted'][1])
    cr.append(ab_summary[n]['rate'][1])
    x.append(np.linspace(converted[n] - 49, converted[n] + 50, 100))
    y.append(scs.binom(total[n], cr[n]).pmf(x[n]))
    if n == best:
        ax.bar(x[n], y[n], alpha=0.5)
    ax.plot(x[n], y[n], alpha=0.5)
array_legend = ['Price 325', 'Price 350', 'Price 375', 'Price 400', 'Price 425']
array_legend[best] += "-Best"
ax.legend(array_legend)



# ax.axvline(x=B_cr * A_total, c='blue', alpha=0.75, linestyle='--')
plt.xlabel('converted')
plt.ylabel('probability')
plt.show()
# fig, ax = plt.subplots(figsize=(12,6))
# xA = np.linspace(A_converted-49, A_converted+50, 100)
# yA = scs.binom(A_total, p_A).pmf(xA)
# ax.bar(xA, yA, alpha=0.5)
#
# xB = np.linspace(B_converted-49, B_converted+50, 100)
# yB = scs.binom(B_total, p_B).pmf(xB)
# ax.bar(xB, yB, alpha=0.5)
# plt.xlabel('converted')
# plt.ylabel('probability')