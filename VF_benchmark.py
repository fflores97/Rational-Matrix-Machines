#%%
from rmm import VF
from rmm import generate_data_from_poles
from itertools import product
import importlib
import numpy as np
import matplotlib.pyplot as plt

importlib.reload(VF)
importlib.reload(generate_data_from_poles)

master_path = 'training_data/run8'

# poles = [10, 15]
# widths = [1e-3]
# points = [6]
# ratios = [1e-3]

poles = [10]
widths = [1e-2]
points = np.arange(2,100,5)
ratios = np.linspace(1e-4,1e-1,20)

iterable = product(poles, widths, points, ratios)

def generate_benchmark(iterable, master_path):
    PFR1= []
    PFR2 = []
    tuples_file = open(master_path + '/arguments.txt', 'w')
    tuples_file.write('Poles, Width, Points, Ratio\n')
    # print(PFR)
    for i, point in enumerate(iterable):
        
        tuples_file.write('Run '+ str(i) + ': ' + str(point) + '\n')

        path = master_path + "/synthetic/run" + str(i)
        generate_data_from_poles.main(master_path, path, point[1], point[2], point[3])
        pfr1, pfr2 = VF.main(path, figurepath = '/../../plots/VF' + str(i) + '.png', iterations=30, poles = point[0])
        PFR1.append(pfr1)
        PFR2.append(pfr2)

        args_file = open(path + '/arguments.txt', 'w')
        args_file.write("Poles\t" + str(point[0]) + '\n')
        args_file.write("Width\t" + str(point[1])+'\n')
        args_file.write("Points per pole\t" + str(point[2])+'\n')
        args_file.write("Signal to noise ratio\t" + str(point[3]))
        args_file.close()

    tuples_file.close()
    PFR_array_1 = np.array(PFR1)
    np.savetxt(master_path + '/PFRs_1.txt', PFR_array_1)
    PFR_array_2 = np.array(PFR2)
    np.savetxt(master_path + '/PFRs_2.txt', PFR_array_2)

def plot_contour(master_path, points, ratios):
    for i in range(1):
        PFR_array = np.loadtxt(master_path + '/PFRs_' + str(i) + '.txt')
        PFR_array = PFR_array.reshape(len(points),len(ratios)).T
        X,Y = np.meshgrid(points, ratios)

        fig, ax = plt.subplots()
        contour_plot = ax.contourf(X,Y,PFR_array)
        # ax.clabel(CS, inline=1, fontsize=10)
        cbar = fig.colorbar(contour_plot)
        cbar.ax.set_xlabel("PFR")
        ax.set_ylabel("Noise ratio")
        ax.set_xlabel("Points per pole")

        plt.savefig(master_path + "/plots/benchmark_VF" + str(i)+ ".pdf")

# %%
generate_benchmark(iterable, master_path)
#%%
plot_contour(master_path, points, ratios)

# %%
importlib.reload(VF)
master_path = 'training_data/run8'
VF.main(master_path + '/synthetic/run270',figurepath1 = '/../../plots/RMM_test_benchmark_1.pdf',figurepath2 = '/../../plots/RMM_test_benchmark_2.pdf',poles=10)


# %%

z_train, Y_train_1, Y_true_1, Y_train_2, Y_true_2, true_poles, true_residues, \
true_offset, number_true_poles, number_train_points = VF.prepare_data(master_path + '/synthetic/run270')

VF.VF_run_and_plot('training_data/test_matrix', z_train, np.hstack([Y_train_1, Y_train_2]), np.hstack([Y_true_1, Y_true_2]), true_poles, true_residues, poles =10)
# %%
np.hstack([Y_train_1, Y_train_2])[0]

# %%
Y_train_2[0]

# %%
