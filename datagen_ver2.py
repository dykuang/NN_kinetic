#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:41:29 2018

@author: dykuang

A different data generator that randomly selectes (E, lnA) points inside

a given polygon
"""

import numpy as np
np.random.seed(1234)
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.special import lambertw
#------------------------------------------------------------------------------
# Some basic setting
#------------------------------------------------------------------------------
R = 8.3145
num_Temp = 150
BETA = [5, 10, 15]
separate = True

alpha = np.linspace(0.01, 0.99, 51)


Temp = np.linspace(300, 1500, num_Temp) # temperature in Kelvin

model_inv = {
        'A2': lambda x: 1-np.exp(-x**2),
        'A3': lambda x: 1-np.exp(-x**3),  
        'A4': lambda x: 1-np.exp(-x**4),  
        'R1': lambda x: x,  
        'R2': lambda x: 1-(1-x)**2, 
        'R3': lambda x: 1-(1-x)**3, 
        'D1': lambda x: x**0.5,
#        'D2': lambda x: 1- np.exp()
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
import random
from shapely.geometry import Polygon, Point

poly = Polygon([(50, 0), (50, 5), (150, 0), (300, 5), (300, 40), (250, 40)])
def random_points_within(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds

    points = []

    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            points.append(random_point)

    return points

num_pts = 600
E = np.zeros(num_pts)
LNA = np.zeros(num_pts)

points = random_points_within(poly, num_pts)
for ite, pts in enumerate(points):
    E[ite], LNA[ite] = pts.xy[0][0], pts.xy[1][0]
    

right = np.zeros([num_pts, num_Temp])
model_ind = 0

y_train = []

for i in range(num_pts):
    inte = lambda x: integrand(E[i], LNA[i], x)
    for k, T in enumerate(Temp):
        right[i,k] = integrate.quadrature(inte, 0, T)[0]
#        right[i, k] = 1e3*E[i]/R*np.exp(LNA[i]-5.331-1.0516*E[i]*1e3/R/T)  #using Doyle's approximation
    y_train.append(np.array([E[i], LNA[i]]))

y_train = np.stack(y_train)

from scipy.interpolate import interp1d
data=[]
for beta in BETA:
    for key, g in model_inv.items():
#        data.append(np.clip(g(right/beta), 0, 1))
        data.append(g(right/beta))
data = np.stack(data)
data = np.reshape(data, [len(BETA)*11*num_pts, num_Temp])

train = []
for dat in data:
    train.append(interp1d(dat, Temp, fill_value = 'extrapolate')(alpha))
    
train = np.stack(train)


#------------------------------------------------------------------------------
# If separate is TURE, return generated data under different heating rate
# in separated files. Else, return a single training file
#------------------------------------------------------------------------------

k = np.sort(np.tile(range(11), num_pts))
k = np.reshape(k,[len(k),1])
y_train = np.tile(y_train, (11,1))
y_train = np.hstack([k, y_train])

if separate:  
    rg = num_pts*11
    mask = np.ones([11*num_pts, 1])
    mask = mask[:,0] > 0
    for i in range(len(BETA)): # clean the data 
        tempX = train[i*rg:(i+1)*rg]
        diffX = np.diff(tempX, axis = 1)

        mask = mask & (np.max(tempX, axis = 1) < 1500) & (~np.isnan(diffX).all(axis=1)) & ((diffX>0).all(axis=1))
    for i in range(len(BETA)):
        tempX = train[i*rg:(i+1)*rg]
        tempX = tempX[mask,:]
        np.save('dataset/xtrain_sep{}'.format(i+1), tempX)
    y_train = y_train[mask,:]    
    np.save('dataset/ytrain_sep.npy', y_train)
    
else:
    y_train = np.tile(y_train, (len(BETA),1))
    
    rgmax = np.max(train, axis = 1)
    rgmin = np.min(train, axis = 1)
    
    mask = (rgmax<1500) & (train[:,0]>300)
    train_processed = train[mask,:]
    y_train = y_train[mask,:]

    np.save('dataset/ytrain.npy', y_train)     
    np.save('dataset/xtrain.npy', train_processed)



