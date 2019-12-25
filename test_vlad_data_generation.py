# %%
from rmm import rmm_generate_data
import matplotlib.pyplot as plt
import importlib
import numpy as np

importlib.reload(rmm_generate_data)

# Define path
path = "training_data/test_data_generation"

# Generate Data
x, data_1, data_2, true_value_1, true_value_2, poles, residues = \
    rmm_generate_data.generate_data()

# Save Data
rmm_generate_data.save_data(path, x, data_1,\
                             data_2, true_value_1,true_value_2,poles,residues)

# Plot and save to path
rmm_generate_data.plot_data(x, data_1,data_2,true_value_1,\
                            true_value_2,poles,residues, path)

# Test loading
loaded_data = \
    rmm_generate_data.load_data(path)
