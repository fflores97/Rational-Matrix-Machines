import numpy as np
from rmm import vectfit

def rational_function_at_z(z,poles,residues,offset=0):
    mod = sum([residues[n]/(z-poles[n]) for n in range(len(poles))]) + offset
    return mod

def model_real(z,poles,residues,offset=0):
    mod = sum([(residues[2*n]+1j*residues[2*n+1])/(z-poles[n]) for n in range(len(poles))]) + offset
    return mod

def rational_function(z, poles, residues,offset=0):
    """
    Vectorize model function
    """
    
    # mod = np.vectorize(model, excluded=[1,2,3], otypes=['complex128'])

    # # return value
    # return mod(z, poles, residues, offset)

    value = np.array([rational_function_at_z(point, poles, residues, offset) for point in z])

    return value

# def save_data(path, X, data_1, data_2, true_value_1, true_value_2, poles, residues):
#     """
#     Function takes in a relative path and data objects and saves them to txt.
    
#     Note that numpy converts them all to float before saving, so they should be
#     re-converted to complex numbers on read
#     """
#     try:
#         os.makedirs(path + '/plots')
#         print('Directories created; saving data')
#     except FileExistsError:
#         print('Directories exist; saving data')

#     np.savetxt(path + '/X.txt', X.view(float))
#     np.savetxt(path + '/poles.txt', poles.view(float))
#     np.savetxt(path + '/residues.txt', residues.view(float))
#     np.savetxt(path + '/data_1.txt', data_1.view(float))
#     np.savetxt(path + '/data_2.txt', data_2.view(float))
#     np.savetxt(path + '/true_value_1.txt', true_value_1.view(float))
#     np.savetxt(path + '/true_value_2.txt', true_value_2.view(float))
def load_data(path):
     """
     Function takes in a relative path and will scan for .txt files
     containing poles, residues, X and Y (real and imaginary parts)
     input to the rational function. 
     """
     X = np.loadtxt(path + '/X.txt')
     data_1 = np.loadtxt(path + '/data_1.txt').view(complex)
     data_2 = np.loadtxt(path + '/data_2.txt').view(complex)
     true_value_1 = np.loadtxt(path + '/true_value_1.txt').view(complex)
     true_value_2 = np.loadtxt(path + '/true_value_2.txt').view(complex)
     poles = np.loadtxt(path + '/poles.txt').view(complex)
     residues = np.loadtxt(path + '/residues.txt').view(complex)
              
     return [X, data_1, data_2, true_value_1, true_value_2, poles, residues]

# def plot_data(X, data_1, data_2, true_value_1, true_value_2, poles, \
#               residues, path = None):

#     plt.subplot(221)
#     plt.plot(X, data_1, '.', X, true_value_1)
#     for pole in poles:
#         vec = [pole.real, pole.real]
#         plt.plot(vec, [min(true_value_1), max(true_value_1)], 'g')
#     plt.xlim([-1,1])

#     plt.subplot(222)
#     plt.plot(X, data_2, '.', X, true_value_2)
#     for pole in poles:
#         vec = [pole.real, pole.real]
#         plt.plot(vec, [min(true_value_2), max(true_value_2)], 'g')
#     plt.xlim([-1,1])

#     plt.subplot(223)
#     plt.semilogy(X, abs(data_1-true_value_1) / abs(true_value_1), '.')
#     plt.ylim([1e-5,1])

#     plt.subplot(224)
#     plt.semilogy(X, abs(data_2-true_value_2) / abs(true_value_2), '.')
#     plt.ylim([1e-5,1])

#     if path != None:
#         plt.savefig(path + "/plots/generated_data.png")