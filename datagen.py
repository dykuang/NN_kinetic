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
num_Temp = 301
BETA = [5, 10, 15, 20, 25, 30]

E = np.linspace(50, 300, num_E)
LNA = np.linspace(5, 30, num_E) # make it random? rule out unreal cases

#EE, lnAA = np.meshgrid(E, LNA)

alpha = np.linspace(0.01, 0.99, 101)


Temp = np.linspace(300, 1500, num_Temp) # temperature in Kelvin

model ={'A2': lambda x: (-np.log(1-x))**0.5,
        'A3': lambda x: (-np.log(1-x))**(1.0/3),  
        'A4': lambda x: (-np.log(1-x))**0.25,  
        'R1 ': lambda x: x,  
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
alpha_T = np.zeros([num_E,num_lnA,num_Temp])
alpha_T_app = np.zeros([num_E,num_lnA,num_Temp])
model_ind = 0

y_train = []
for i, EE in enumerate(E):
     for j, lnA in enumerate(LNA):
           inte = lambda x: integrand(EE, lnA, x)
           for k, T in enumerate(Temp):
                alpha_T[i,j,k] = integrate.quadrature(inte, 0, T)[0]
#                alpha_T_app[i,j,k] = R*T**2/EE*(1e-3)*(1-2*R*T/EE*(1e-3))*np.exp(lnA-1000*EE/R/T)
           y_train.append(np.array([EE, lnA]))

y_train = np.stack(y_train)

from scipy.interpolate import interp1d
data=[]
for beta in BETA:
    for key, g in model_inv.items():
        data.append(np.clip(g(alpha_T/beta), 0, 1))

data = np.stack(data)

data = np.reshape(data, [66*20*20, 301])

train = []
for dat in data:
    train.append(interp1d(dat, Temp, fill_value = 'extrapolate')(alpha))
    
train = np.stack(train)

k = np.sort(np.tile(range(11), num_E*num_lnA))
k = np.reshape(k,[len(k),1])
y_train = np.tile(y_train, (11,1))
y_train = np.hstack([k, y_train])
y_train = np.tile(y_train, (6,1))

rgmax = np.max(train, axis = 1)
rgmin = np.min(train, axis = 1)

mask = (rgmax<1500) & (train[:,0]>300)
train_processed = train[mask,:]
y_train = y_train[mask,:]


# =============================================================================
# Temp =   np.zeros([11, 20, 20, 101])   
# ind = 0      
# for key, g in model.items():
#      g_alpha = g(alpha)
#      for i, EE in enumerate(E):
#           for j, lnA in enumerate(LNA):
#                for k, val in enumerate(g_alpha):
#                      Temp[ind,i,j,k]=fsolve(lambda T: R*T**2/EE*(1e-3)*np.exp(lnA-1000*EE/R/T) 
#                                          - val*10, 900)[0]
#      ind = ind + 1
# =============================================================================
          




