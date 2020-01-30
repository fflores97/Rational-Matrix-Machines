import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from oct2py import Oct2Py
import numpy as np
import matplotlib.pyplot as plt
import os
from rmm import generate_data

def generate_data_with_poles(poles, residues, relative_width_of_poles = 1e-2,\
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
            oc.generate_data_with_poles(poles, residues, relative_width_of_poles, \
                number_points_per_pole, signal_to_noise_ratio, nout = 5)

    for i, value in enumerate(output):
        # value = np.array(value)
        output[i] = value.reshape(len(value),)

    return output


def main(path, path2, relative_width_of_poles = 1e-2,\
                  number_points_per_pole = 100, signal_to_noise_ratio = 1e-3, seed = None):

    x, data_1, data_2, true_value_1, true_value_2, poles, residues = generate_data.load_data(path)
    
    # Generate Data
    x, data_1, data_2, true_value_1, true_value_2 = \
        generate_data_with_poles(poles, residues, relative_width_of_poles,\
            number_points_per_pole, signal_to_noise_ratio, seed)

    # Save Data
    generate_data.save_data(path2, x, data_1,\
        data_2, true_value_1,true_value_2,poles,residues)

    # Plot and save to path
    generate_data.plot_data(x, data_1,data_2,true_value_1,\
        true_value_2,poles,residues, path2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
        "Provide path with data files")
    parser.add_argument('path', help='Path to origin files')
    parser.add_argument('path2', help='Path to target files')
    # parser.add_argument('number_of_poles', help = 'default = 10', nargs='?', default = 10, type = int)
    parser.add_argument('relative_width_of_poles', help = 'default = 1e-2',nargs='?',default = 1e-2, type=float)
    parser.add_argument('number_points_per_pole', help = 'default = 100',nargs='?',default=100, type=int)
    parser.add_argument('signal_to_noise_ratio', help = 'default = 1e-3',nargs='?',default=1e-3, type=float)
    args = parser.parse_args()
    print(args)
    main(args.path, args.path2, args.relative_width_of_poles, 
    args.number_points_per_pole,args.signal_to_noise_ratio)