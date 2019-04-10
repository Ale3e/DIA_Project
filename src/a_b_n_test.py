import numpy as np
import matplotlib.pyplot as plt
from algorithms.a_b_n.data import *


p_A = 0.06

p = np.array([0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001])
# A is control; B is test
N_A = 1000
N_B = 1000
N_C = 1000
ab_data = []
ab_summary = []
for n in range(len(p)):
    ab_data.append(generate_data(N_A, N_B, p_A, p[n], days=365))
    ab_summary.append(ab_data[n].pivot_table(values='converted', index='group', aggfunc=np.sum))
    ab_summary[n]['total'] = ab_data[n].pivot_table(values='converted', index='group', aggfunc=lambda x: len(x))
    ab_summary[n]['rate'] = ab_data[n].pivot_table(values='converted', index='group')

A_converted = ab_summary[0]['converted'][0]
A_total = ab_summary[0]['total'][0]
A_cr = ab_summary[0]['rate'][0]

total = []
converted = []
cr = []

total.append(ab_summary[0]['total'][0])
cr.append(ab_summary[0]['converted'][0])
converted.append(ab_summary[0]['rate'][0])
fig, ax = plt.subplots(figsize=(12,6))
xA = np.linspace(A_converted-49, A_converted+50, 100)
yA = scs.binom(A_total, A_cr).pmf(xA)
ax.bar(xA, yA, alpha=0.5)
ax.plot(xA, yA, alpha=0.5)
x = []
y = []
for n in range(len(p)):
    total.append(ab_summary[n]['total'][1])
    converted.append(ab_summary[n]['converted'][1])
    cr.append(ab_summary[n]['rate'][1])
    x.append(np.linspace(converted[n] - 49, converted[n] + 50, 100))
    y.append(scs.binom(total[n], cr[n]).pmf(x[n]))
    ax.bar(x[n], y[n], alpha=0.5)
    ax.plot(x[n], y[n], alpha=0.5)
ax.legend(['Group A', 'Group B', 'Group C', 'Group D', 'Group E', 'Group F', 'Group G', 'Group H'])

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
