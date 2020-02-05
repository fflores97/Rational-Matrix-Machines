#%%
from rmm import VF
from rmm import generate_data_from_poles
from itertools import product
import importlib
import numpy as np
import matplotlib.pyplot as plt

importlib.reload(VF)
importlib.reload(generate_data_from_poles)

master_path = 'training_data/paper_run'

PFR = []
pole_numbers = np.arange(4,35,3).tolist()
pole_numbers.append(20)

z_train, Y_train_1, Y_true_1, Y_train_2, Y_true_2, true_poles, true_residues, \
    true_offset, number_true_poles, number_train_points = VF.prepare_data(master_path)

for pole_num in pole_numbers:
    print(type(pole_num))
    pfr1= VF.VF_run_and_plot(master_path + '/poles_benchmark',z_train, Y_train_1, Y_true_1, true_poles, true_residues, iterations = 30, poles = pole_num)
    plt.savefig(master_path + '/poles_benchmark/plots/poles_'+str(pole_num)+'.pdf')
    PFR.append(pfr1)
print(PFR)

# %%
fig, ax = plt.subplots()
plt.scatter(pole_numbers, PFR)
ax.set_xlabel("Pole number")
ax.set_ylabel("PFR")
plt.savefig(master_path + '/plots/pole_number_PFR.pdf')
# %%
