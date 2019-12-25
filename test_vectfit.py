# %%

import numpy as np
import matplotlib.pyplot as plt
import importlib
import rmm

importlib.reload(rmm)

test_s, data_1, data_2, true_value_1, true_value_2, poles, residues = \
    rmm.generate_data.generate_data(seed=123)


#%%
fitted_poles, fitted_residues, d, h = rmm.vectfit.vectfit_auto_rescale(data_1, test_s)

fitted = rmm.vectfit.model(test_s, fitted_poles, fitted_residues, d, h)

#%%

plt.figure()
plt.plot(test_s, data_1, 'r')
plt.plot(test_s, fitted.real, 'b')
plt.ylim([0.5,1.5])
