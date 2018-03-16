# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:25:23 2018

@author: Dongyang

This script generates kinetic data for training/testing the network

data format: T(alpha)| E, A, g(\alpha)

g(\alpha) is an integer label and should be one-hot encoded before training
"""

import numpy as np
np.random.seed(1234)
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


#------------------------------------------------------------------------------
# Some basic setting
#------------------------------------------------------------------------------
R = 8.3145
num_E = 20
num_lnA = 20
BETA = [10, 20, 30 ,40]

E = np.linspace(50, 300, num_E)
LNA = np.linspace(1, 30, num_E) # make it random? rule out unreal cases

#EE, lnAA = np.meshgrid(E, LNA)

alpha = np.linspace(0.01, 0.9999, 101)

Temp = np.linspace(500, 1500, 101) # temperature in Kelvin

model ={'A2': lambda x: (-np.log(1-x))**0.5,
        'A3': lambda x: (-np.log(1-x))**(1.0/3),  
        'A4': lambda x: (-np.log(1-x))**0.25,  
        'R1': lambda x: x,  
        'R2': lambda x: 1-(1-x)**0.5, 
        'R3': lambda x: 1-(1-x)**(1.0/3), 
        'D1': lambda x: x**2,
#        'D2': lambda x: (1-x)*np.log(1-x) + x,
        'D3': lambda x: (1-(1-x)**(1./3.)) **2,
#        'D4': lambda x: (1-(2./3.)*x)-(1-x)**(2.0/3.0),
        'F1': lambda x: -np.log(1-x),
        'F2': lambda x: 1/(1-x)-1,
        'F3': lambda x: (1/(1-x)**2-1)/2
          }

model_inv = {
        'A2': lambda x: 1-np.exp(-x**2),
        'A3': lambda x: 1-np.exp(-x**3),  
        'A4': lambda x: 1-np.exp(-x**4),  
        'R1': lambda x: x,  
        'R2': lambda x: 1-(1-x)**2, 
        'R3': lambda x: 1-(1-x)**3, 
        'D1': lambda x: x**0.5,
        'D3': lambda x: 1-(1-x**0.5)**3,
        'F1': lambda x: 1-np.exp(-x),
        'F2': lambda x: 1-1/(1+x),
        'F3': lambda x: 1-1/(2*x+1)**0.5
          }

def integrand(x, y, z):
     return np.exp(y-1000*x/R/z)


#------------------------------------------------------------------------------
# Generating data
#------------------------------------------------------------------------------
RIGHT = []
alpha_T = np.zeros([20, 20, 101])
alpha_T_app = np.zeros([20, 20, 101])
model_ind = 0

for i, EE in enumerate(E):
     for j, lnA in enumerate(LNA):
           inte = lambda x: integrand(EE, lnA, x)
           for k, T in enumerate(Temp):
                alpha_T[i,j,k] = integrate.quad(inte, 0, T)[0]
#                alpha_T_app[i,j,k] = R*T**2/EE*(1e-3)*(1-2*R*T/EE*(1e-3))*np.exp(lnA-1000*EE/R/T)
     




# =============================================================================
# Temp =   np.zeros([13, 20, 20, 101])   
# ind = 0      
# for key, g in model.items():
#      g_alpha = g(alpha)
#      for i, EE in enumerate(E):
#           for j, lnA in enumerate(LNA):
#                for k, val in enumerate(g_alpha):
#                      Temp[ind,i,j,k]=fsolve(lambda T: R*T**2/EE*(1e-3)*(1-2*R*T/EE*(1e-3))*np.exp(lnA-1000*EE/R/T) 
#                                          - g_alpha*10, 0.5)[0]
# =============================================================================
          




