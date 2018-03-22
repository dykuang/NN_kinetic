# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 10:35:53 2018

@author: dykuang

This script reads experiment data from .csv
"""
import numpy as np
import os
import csv
from scipy.interpolate import interp1d

alpha = np.linspace(0.01, 0.99, 101)
folder = r'D:\life\Google\Matlab Package\XuBang\paper 2\data'

pine_data=[]
with open(os.path.join(folder, 'pine5.csv'), 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        pine_data.append(row)

pine_data = np.stack(pine_data)
pine_data = pine_data.astype(float)
extract = interp1d(pine_data[:,6], pine_data[:,1], fill_value = 'extrapolate')(alpha)

