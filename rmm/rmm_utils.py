import numpy as np
import matplotlib.pyplot as plt
import os

from rmm import rmm_plot

def complex_mesh(real, imaginary):
    """
    Function takes in arrays of real and imaginary values
    and returns a meshgrid of complex numbers
    """
    
    XX, YY = np.meshgrid(real,imaginary)
    Z = (XX + 1j*YY).flatten()
    
    return Z


def rational_function(z, poles, residues, offset=0, poly_coeff=()):
    """
    Apply rational function defined by parameters to a complex number and
    return another complex number.
    
    Function has rational, entire, and 
    constant parts. Coefficients for entire part can be left blank.  In such 
    a case the function will be purely rational. Constant coefficient (offset)
    is defaulted to 0
    """
    
    if len(poles) != len(residues):
        raise ValueError("Number of poles not equal to number of residues")

#    if poly_coeff == ():
#        poly_order = int(0)
#        # print("The rational_function function was not given an entire polynomial part to build.")
#    else:
#        poly_order = int(poly_coeff.shape[0])

    total_function = np.zeros(len(z), dtype='complex128')
    for i, value in enumerate(z):
        rational_part = np.sum([residues[idx]/(value - pole) for idx, pole in enumerate(poles)])
        entire_part = np.sum([coefficient * (value ** (idx + 1)) for idx, coefficient in enumerate(poly_coeff)])
        total_function[i] = rational_part + entire_part

    return total_function
    

def random_poles(n, ranges):
    """
    Function generates n random poles as complex numbers. 
    
    Ranges should be specified as a list of the form 
    [[xmin, xmax], [ymin, ymax]].
    """
    
    real_range = ranges[0]
    imaginary_range = ranges[1]
    
    # Generate poles from a random uniform scaled by the range
    real_part = abs(real_range[1] - real_range[0])*np.random.rand(n)
    real_part = real_part + real_range[0]

    imaginary_part = abs(imaginary_range[1] - imaginary_range[0])*np.random.rand(n)
    imaginary_part = imaginary_part + imaginary_range[0]
    
    # Convert to complex number
    poles = real_part + 1j*imaginary_part

    return poles
    
def save_data(path, poles, residues, X, Y, output_data):
    """
    Function takes in a relative path and data objects and saves them to txt.
    
    Note that numpy converts them all to float before saving, so they should be
    re-converted to complex numbers on read
    """
    try:
        os.makedirs(path + '/plots')
        print('Directories created')
    except FileExistsError:
        print('Directories exist; saving data')
    
    np.savetxt(path + '/poles.txt', poles.view(float))
    np.savetxt(path + '/residues.txt', residues.view(float))
    np.savetxt(path + '/X.txt', X.view(float))
    np.savetxt(path + '/Y.txt', Y.view(float))
    np.savetxt(path + '/output_data.txt', output_data.view(float))
    #np.savetxt(path + '/noised_data.txt', noised_data.view(float))
    
def load_data(path):
    """
    Function takes in a relative path and will scan for .txt files
    containing poles, residues, X and Y (real and imaginary parts)
    input to the rational function. 
    """
    poles = np.loadtxt(path + '/poles.txt').view(complex)
    residues = np.loadtxt(path + '/residues.txt').view(complex)
    X = np.loadtxt(path + '/X.txt').view(complex)
    Y = np.loadtxt(path + '/Y.txt').view(complex)
    output_data = np.loadtxt(path + '/output_data.txt').view(complex)
    noised_data = np.loadtxt(path + '/noised_data.txt').view(complex)
              
    return [poles, residues, X, Y, output_data, noised_data]
              
    
def generate_true_data(path, X, Y, poles, residues, entire_coefficients=(), offset=0):
    """
    Function generates figures and data files from a space of real and
    imaginary values to a specified path. Returns real data
    """
    
    Z = complex_mesh(X,Y)

    complex_data = rational_function(Z, poles, residues, offset, entire_coefficients)
    #noised_data = toy_data + np.random.normal(0, 2, len(toy_data))
    
    real_data = rational_function(X, poles, residues, offset, entire_coefficients)
    real_data = real_data.real
    
    save_data(path, poles, residues, X, Y, real_data)
    
    plot_3d = rmm_plot.plot_poles(X, Y, complex_data)
    plot_3d.savefig(path + '/plots/3d_poles.svg')

    fig, ax = plt.subplots()
    ax.plot(X, real_data.real)
    ax.set(xlabel = "x", ylabel = "f(x)")
    fig.savefig(path + '/plots/2d_poles.svg')
    
    return complex_data, real_data

def add_gaussian_noise(path, sigma):
    """
    Input path with data files and standard deviation for random Gaussian
    noise
    """
    
    data = np.loadtxt(path + '/output_data.txt')
    noised_data = data + np.random.normal(0,sigma,data.size)
    np.savetxt(path + '/noised_data.txt', noised_data)
    return noised_data