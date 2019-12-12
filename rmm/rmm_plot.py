import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def plot_poles(X, Y, rational_data, plot_imaginary=False):
    """
    Plot surface plot with poles
    """
    XX, YY = np.meshgrid(X,Y)
    
    Z = np.reshape(rational_data, XX.shape)
    
    if plot_imaginary:
        Z = Z.imag
    else:
        Z = Z.real
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(XX,YY,Z, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
    ax.set_zlim(-10,10)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set(xlabel = "Re(z)", ylabel = "Im(z)", zlabel = "f(z)")
    
    return fig

def plot_training_data(path):
    
    X = np.loadtxt(path + '/X.txt')
    data = np.loadtxt(path + '/output_data.txt')
    noised_data = np.loadtxt(path + '/noised_data.txt')
    
    fig, ax = plt.subplots()
    ax.plot(X, data, 'r')
    ax.set(xlabel = "x", ylabel = "f(x)")
    ax.plot(X, noised_data, 'x')
    fig.savefig(path + '/plots/training.svg')