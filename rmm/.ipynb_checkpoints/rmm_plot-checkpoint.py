import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def plot_poles(X, Y, rational_data, plot_imaginary=False):
    
    XX, YY = np.meshgrid(X,Y)
    
    Z = np.reshape(rational_data, XX.shape)
    
    if plot_imaginary:
        Z = Z.imag
    else:
        Z = Z.real
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(XX,YY,Z, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
    #ax.contour3D(XX,YY,Z, c = Z)
    ax.set_zlim(-10,10)
    ax.zaxis.set_major_locator(LinearLocator(10))
    
    return fig