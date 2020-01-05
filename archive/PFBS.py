# %%
import matplotlib.pyplot as plt
import importlib
import numpy as np
import rmm

importlib.reload(rmm)

# Define path
path = "training_data/test_data_generation"

# Generate Data
x, data_1, data_2, true_value_1, true_value_2, poles, residues = \
    rmm.generate_data.generate_data()

# Save Data
rmm.generate_data.save_data(path, x, data_1,\
                             data_2, true_value_1,true_value_2,poles,residues)

# Plot and save to path
rmm.generate_data.plot_data(x, data_1,data_2,true_value_1,\
                            true_value_2,poles,residues, path)

# Test loading
loaded_data = \
    rmm.generate_data.load_data(path)


# %%
importlib.reload(rmm)
test = rmm.error_function.error(x,np.transpose(np.array([true_value_1,true_value_2])),[1/len(x)]*len(x),poles,residues,[0]*len(poles),1,0,0,rmm.error_function.delta_5,rmm.error_function.delta_5)
test


# %%
Y = [np.real(rmm.utils.model(point, poles, residues[:,1],1)) for point in x]
Y-true_value_2

# %%

from scipy import optimize

