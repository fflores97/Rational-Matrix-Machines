from oct2py import Oct2Py
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_data(number_of_poles = 10, relative_width_of_poles = 1e-2,\
                  number_points_per_pole = 100, signal_to_noise_ratio = 1e-3, seed = None):
    
    """
    Generate data from Matlab code by Vladimir Sobes. Here implemented using GNU Octave
    due to licensing.
    """
    with Oct2Py() as oc:
        oc.eval('pkg load statistics')
        if seed != None:
            oc.eval('rand ("state", ' + str(seed) + ")")
        output = \
            oc.generate_data(number_of_poles,relative_width_of_poles, \
                number_points_per_pole, signal_to_noise_ratio, nout = 7)

    for i, value in enumerate(output[:-1]):
        output[i] = value.reshape(len(value),)

    return output

