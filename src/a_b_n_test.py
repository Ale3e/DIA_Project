import numpy as np
import matplotlib.pyplot as plt
import pandas as ps
from algorithms.a_b_n.data import *


p_A = 0.06
p_B = 0.032
p_C = 0.015

# A is control; B is test
N_A = 1000
N_B = 1000
N_C = 1000


ab_data = generate_data(N_A, N_B, p_A, p_B, days=365)
ab_data1 = generate_data(N_A, N_B, p_A, p_C, days=365)

ab_summary = ab_data.pivot_table(values='converted', index='group', aggfunc=np.sum)
# add additional columns to the pivot table
ab_summary1 = ab_data1.pivot_table(values='converted', index='group', aggfunc=np.sum)

ab_summary['total'] = ab_data.pivot_table(values='converted', index='group', aggfunc=lambda x: len(x))
ab_summary['rate'] = ab_data.pivot_table(values='converted', index='group')
ab_summary1['total'] = ab_data1.pivot_table(values='converted', index='group', aggfunc=lambda x: len(x))
ab_summary1['rate'] = ab_data1.pivot_table(values='converted', index='group')


A_converted=ab_summary['converted'][0]
A_total=ab_summary['total'][0]
A_cr=ab_summary['rate'][0]
B_converted=ab_summary['converted'][1]
B_total=ab_summary['total'][1]
B_cr=ab_summary['rate'][1]
C_converted=ab_summary1['converted'][1]
C_total=ab_summary1['total'][1]
C_cr=ab_summary1['rate'][1]



fig, ax = plt.subplots(figsize=(12,6))
xA = np.linspace(A_converted-49, A_converted+50, 100)
yA = scs.binom(A_total, A_cr).pmf(xA)
ax.bar(xA, yA, alpha=0.5)
xB = np.linspace(B_converted-49, B_converted+50, 100)
yB = scs.binom(B_total, B_cr).pmf(xB)
ax.bar(xB, yB, alpha=0.5)
xC = np.linspace(C_converted-49, C_converted+50, 100)
yC = scs.binom(C_total, C_cr).pmf(xB)
ax.bar(xC, yC, alpha=0.5)

ax.legend(['Group A', 'Group B', 'Group C'])



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