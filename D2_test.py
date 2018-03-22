# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 12:12:44 2018

@author: dykuang

This script tests the inverse of 'D2'
"""
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.special import lambertw
from math import e

E = 160*1e3
lnA = 10
R=8.3145
Temp = np.linspace(500, 1000, 101)

inte = lambda z: np.exp(lnA-E/R/z)
inv = lambda z: 1-np.exp(lambertw((z-1)/e,0)+1)
D2 = lambda x: (1-x)*np.log(1-x) + x

alpha = []

for T in Temp:
#    alpha.append(fsolve(lambda x: D2(x) == integrate.quadrature(inte, 0, T)[0]/5, 0.5))
    alpha.append(inv(integrate.quadrature(inte, 0, T)[0]/5))


plt.plot(Temp, alpha)