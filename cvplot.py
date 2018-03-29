# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 21:34:15 2018

@author: dykua

This script creates visualization of the cross validation result
"""

E_MLP = [4.56, 7.35, 9.53]
E_CNN = [2.43, 2.12, 2.26]
lnA_MLP = [5.01, 8.36, 10.47]
lnA_CNN = [2.92, 2.69, 3.14]
acc_MLP = [98.07, 90.23, 83.43]
acc_CNN = [ 99.54, 99.30, 98.98]

acc_MLP_std= [1.56, 1.60, 2.51]
acc_CNN_std= [0.5, 0.5, 0.98]

sigma = [0.1,0.2,0.3]

import numpy as np
import matplotlib.pyplot as plt

plt.figure()
plt.errorbar(sigma, acc_MLP, acc_MLP_std, marker = '+', color = 'orangered', capsize = 3)
plt.errorbar(sigma, acc_CNN, acc_CNN_std, marker = '*', color = 'teal', capsize = 3)
plt.title('Accuracy with standard deviation')
plt.xlabel('Noise intensity')
plt.ylabel('Accuracy (%)')
plt.legend(['MLP', 'CNN'])
plt.xticks([0.1, 0.2, 0.3], ['0.1', '0.2', '0.3'])

N = 3
fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, E_MLP, width, color='orangered')
p2 = ax.bar(ind + width, E_CNN, width, color='teal')

ax.set_title('MAPE of predictions for E')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('0.1', '0.2', '0.3'))
ax.set_xlabel('Noise intensity')
ax.set_ylabel('MAPE (%) of E')
ax.set_ylim([0, 11])
ax.legend((p1[0], p2[0]), ('MLP', 'CNN'))

ax.autoscale_view()
#plt.show()

fig1, ax1 = plt.subplots()
p11 = ax1.bar(ind, lnA_MLP, width, color='orangered')
p22 = ax1.bar(ind + width, lnA_CNN, width, color='teal')

ax1.set_title('MAPE of predictions for ln A')
ax1.set_xticks(ind + width / 2)
ax1.set_xticklabels(('0.1', '0.2', '0.3'))
ax1.set_xlabel('Noise intensity')
ax1.set_ylabel('MAPE (%) of ln A')
ax1.set_ylim([0, 11])
ax1.legend((p11[0], p22[0]), ('MLP', 'CNN'))

ax1.autoscale_view()
plt.show()
