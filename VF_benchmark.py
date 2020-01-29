#%%
from rmm import VF
from rmm import generate_data_from_poles
from itertools import product
import importlib
import numpy as np

# importlib.reload(rmm)

master_path = 'training_data/run1'

# poles = [10, 15]
# widths = [1e-3]
# points = [6]
# ratios = [1e-3]

poles = [10]
widths = [1e-2]
points = [3 ,6, 30, 100]
ratios = [1e-4, 1e-3, 1e-2, 1e-1]

iterable = product(poles, widths, points, ratios)
PFR = []
tuples_file = open(master_path + '/arguments.txt', 'w')
tuples_file.write('Poles, Width, Points, Ratio\n')


# print(PFR)
for i, point in enumerate(iterable):
    
    tuples_file.write('Run '+ str(i) + ': ' + str(point) + '\n')

    path = master_path + "/synthetic/run" + str(i)
    generate_data_from_poles.main(master_path, path, point[1], point[2], point[3])
    pfr = VF.main(path, figurepath = '/../../plots/VF' + str(i) + '.png', iterations=30, poles = point[0])
    PFR.append(pfr)

    args_file = open(path + '/arguments.txt', 'w')
    args_file.write("Poles\t" + str(point[0]) + '\n')
    args_file.write("Width\t" + str(point[1])+'\n')
    args_file.write("Points per pole\t" + str(point[2])+'\n')
    args_file.write("Signal to noise ratio\t" + str(point[3]))
    args_file.close()

tuples_file.close()

# %%

PFR_array = np.array(PFR)
PFR_array = PFR.reshape(4,4)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.contour(points, ratios, Z= PFR_array)

# %%
PFR

# %%
