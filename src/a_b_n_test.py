import numpy as np
import matplotlib.pyplot as plt
import pandas as ps
from algorithms.a_b_n.data import *


# code examples presented in Python
p_A = 0.04  # baseline conversion rate
p_B = 0.032  # difference between the groups

# A is control; B is test
N_A = 1000
N_B = 1000


ab_data = generate_data(N_A, N_B, p_A, p_B)

ab_summary = ab_data.pivot_table(values='converted', index='group', aggfunc=np.sum)
# add additional columns to the pivot table

ab_summary['total'] = ab_data.pivot_table(values='converted', index='group', aggfunc=lambda x: len(x))
ab_summary['rate'] = ab_data.pivot_table(values='converted', index='group')



A_converted=ab_summary['converted'][0]
A_total=ab_summary['total'][0]
A_cr=ab_summary['rate'][0]
B_converted=ab_summary['converted'][1]
B_total=ab_summary['total'][1]
B_cr=ab_summary['rate'][1]



fig, ax = plt.subplots(figsize=(12,6))
xA = np.linspace(A_converted-49, A_converted+50, 100)
yA = scs.binom(A_total, A_cr).pmf(xA)
ax.bar(xA, yA, alpha=0.5)
xB = np.linspace(B_converted-49, B_converted+50, 100)
yB = scs.binom(B_total, B_cr).pmf(xB)
ax.bar(xB, yB, alpha=0.5)


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

print(ab_summary['rate'][0] - ab_summary['rate'][1])