import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
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

def save_data(path, X, data_1, data_2, true_value_1, true_value_2, poles, residues):
    """
    Function takes in a relative path and data objects and saves them to txt.
    
    Note that numpy converts them all to float before saving, so they should be
    re-converted to complex numbers on read
    """
    try:
        os.makedirs(path + '/plots')
        print('Directories created; saving data')
    except FileExistsError:
        print('Directories exist; saving data')

    np.savetxt(path + '/X.txt', X.view(float))
    np.savetxt(path + '/poles.txt', poles.view(float))
    np.savetxt(path + '/residues.txt', residues.view(float))
    np.savetxt(path + '/data_1.txt', data_1)
    np.savetxt(path + '/data_2.txt', data_2)
    # np.savetxt(path + '/data_1.txt', data_1.view(float))
    # np.savetxt(path + '/data_2.txt', data_2.view(float))
    np.savetxt(path + '/true_value_1.txt', true_value_1.view(float))
    np.savetxt(path + '/true_value_2.txt', true_value_2.view(float))

def load_data(path):
    """
    Function takes in a relative path and will scan for .txt files
    containing poles, residues, X and Y (real and imaginary parts)
    input to the rational function. 
    """
    X = np.loadtxt(path + '/X.txt')
    data_1 = np.loadtxt(path + '/data_1.txt')
    data_2 = np.loadtxt(path + '/data_2.txt')
    true_value_1 = np.loadtxt(path + '/true_value_1.txt')
    true_value_2 = np.loadtxt(path + '/true_value_2.txt')
    poles = np.loadtxt(path + '/poles.txt').view(complex)
    residues = np.loadtxt(path + '/residues.txt').view(complex)
              
    return [X, data_1, data_2, true_value_1, true_value_2, poles, residues]

def plot_data(X, data_1, data_2, true_value_1, true_value_2, poles, \
              residues, path = None):

    plt.subplot(221)
    plt.plot(X, data_1, '.', X, true_value_1)
    for pole in poles:
        vec = [pole.real, pole.real]
        plt.plot(vec, [min(true_value_1), max(true_value_1)], 'g')
    plt.xlim([-1,1])

    plt.subplot(222)
    plt.plot(X, data_2, '.', X, true_value_2)
    for pole in poles:
        vec = [pole.real, pole.real]
        plt.plot(vec, [min(true_value_2), max(true_value_2)], 'g')
    plt.xlim([-1,1])

    plt.subplot(223)
    plt.semilogy(X, abs(data_1-true_value_1) / abs(true_value_1), '.')
    plt.ylim([1e-5,1])

    plt.subplot(224)
    plt.semilogy(X, abs(data_2-true_value_2) / abs(true_value_2), '.')
    plt.ylim([1e-5,1])

    if path != None:
        plt.savefig(path + "/plots/generated_data.png")

def main(path,number_of_poles = 10, relative_width_of_poles = 1e-2,\
                  number_points_per_pole = 100, signal_to_noise_ratio = 1e-3, seed = None):
    # Generate Data
    x, data_1, data_2, true_value_1, true_value_2, poles, residues = \
        generate_data(number_of_poles, relative_width_of_poles,\
            number_points_per_pole, signal_to_noise_ratio, seed)

    # Save Data
    save_data(path, x, data_1,\
        data_2, true_value_1,true_value_2,poles,residues)

    # Plot and save to path
    plot_data(x, data_1,data_2,true_value_1,\
        true_value_2,poles,residues, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
        "Provide path with data files")
    parser.add_argument('path', help='Path to data files')
    parser.add_argument('number_of_poles', help = 'default = 10', nargs='?', default = 10, type = int)
    parser.add_argument('relative_width_of_poles', help = 'default = 1e-2',nargs='?',default = 1e-2, type=float)
    parser.add_argument('number_points_per_pole', help = 'default = 100',nargs='?',default=100, type=int)
    parser.add_argument('signal_to_noise_ratio', help = 'default = 1e-3',nargs='?',default=1e-3, type=float)
    args = parser.parse_args()
    print(args)
    main(args.path, args.number_of_poles, args.relative_width_of_poles, 
    args.number_points_per_pole,args.signal_to_noise_ratio)