#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:14:44 2019

@author: felipeflores
"""

from rmm import rmm_plot
from rmm import rmm_utils
import numpy as np
import importlib
#import matplotlib.pyplot as plt

importlib.reload(rmm_utils)
importlib.reload(rmm_plot)

X = np.linspace(-10,10,100)
Y = np.linspace(-5,5,100)
true_poles = np.array([3+1j*3,5])
true_residues = np.array([2,1])
entire_coefficients = np.array([1, -0.2, 0.01])
path = 'training_data/test'
sigma = 3

Z,R = rmm_utils.generate_true_data(path,
                                 X, Y, true_poles, true_residues,
                                 entire_coefficients)

noised_data = rmm_utils.add_gaussian_noise(path, sigma)

rmm_plot.plot_training_data(path)
