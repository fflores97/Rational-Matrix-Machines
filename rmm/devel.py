import numpy as np
from rmm import rmm_plot
from rmm import rmm_utils

import matplotlib.pyplot as plt

def generate_true_data(path, X, Y, poles, residues, entire_coefficients=(), offset=0):
    """
    Function generates figures and data files from a space of real and
    imaginary values to a specified path. Returns real data
    """
    # rational_function = np.vectorize(model, excluded=[1,2,3,4], otypes=['complex128'])

    Z = complex_mesh(X,Y)

    complex_data = rational_function(Z, poles, residues, offset, entire_coefficients)

    #noised_data = toy_data + np.random.normal(0, 2, len(toy_data))
    
    real_data = rational_function(X, poles, residues, offset, entire_coefficients)
    real_data = real_data.imag
    
    imaginary_data = rational_function(1j*Y, poles, residues, offset, entire_coefficients)
    imaginary_data = imaginary_data.imag
    
    rmm_utils.save_data(path, poles, residues, X, Y, complex_data)
    
    plot_3d = rmm_plot.plot_poles(X, Y, complex_data)
    plot_3d.savefig(path + '/plots/3d_poles.svg')

    fig, (ax1,ax2) = plt.subplots(2,1)
    ax1.plot(X, real_data)
    ax1.set(xlabel = "Re(z)", ylabel = "f(z)")
    ax2.plot(Y, imaginary_data)
    ax2.set(xlabel = "Im(z)", ylabel = "f(z)")
    fig.savefig(path + '/plots/2d_poles.svg')

    return complex_data


def complex_mesh(real, imaginary):
    """
    Function takes in arrays of real and imaginary values
    and returns a meshgrid of complex numbers
    """
    
    XX, YY = np.meshgrid(real,imaginary)
    Z = (XX + 1j*YY)
    
    return Z

def model(z, poles, residues, offset=0, poly_coeff=()):
    """
    Apply rational function defined by parameters to a complex number and
    return another complex number.
    
    Function has rational, entire, and 
    constant parts. Coefficients for entire part can be left blank.  In such 
    a case the function will be purely rational. Constant coefficient (offset)
    is defaulted to 0
    """
    
    if np.size(poles) != np.size(residues):
        raise ValueError("Number of poles not equal to number of residues")
    
    rational_part = np.sum([residue/(z - pole) for residue, pole in zip(residues, poles)])
    entire_part = np.sum([coefficient * (z ** (idx + 1)) for idx, coefficient in enumerate(poly_coeff)])
    total_function = rational_part + entire_part + offset

    return total_function


def rational_function(z, poles, residues,offset=0,poly_coeff=()):
    """
    Vectorize model function
    """
    
    mod = np.vectorize(model, excluded=[1,2,3,4], otypes=['complex128'])

    # return value
    return mod(z, poles, residues, offset , poly_coeff)