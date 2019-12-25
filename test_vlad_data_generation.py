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
