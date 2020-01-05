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

def minimization_function(res):

    return rmm.error_function.error(x,np.transpose(np.array([true_value_1,true_value_2])),[1/len(x)]*len(x),poles,res,[0]*len(poles),1,0,0,rmm.error_function.delta_5,rmm.error_function.delta_5)

res = optimize.minimize(minimization_function,[1,1]*len(poles))

# %%
np.array([[1,1]]*10)


# %%
res2 = optimize.minimize(minimization_function,np.array([[1,1]]*10))

# %%
res2.x.reshape(10,2)

# %%
res2.x.reshape(10,2) - np.real(residues)
# %%

importlib.reload(rmm)
def minimization_function2(res):
    return rmm.error_function.error_real(x,np.transpose(np.array([true_value_1,true_value_2])),[1/len(x)]*len(x),poles,res,[0]*(4*len(poles)),1,0,0,rmm.error_function.delta_5,rmm.error_function.delta_5)


res6 = optimize.minimize(minimization_function2,[1]*(4*len(poles)))


# %%
importlib.reload(rmm)
rmm.error_function.error_real(x,np.transpose(np.array([true_value_1,true_value_2])),[1/len(x)]*len(x),poles,residues.view('float64'),[0]*(2*len(poles)),1,0,0,rmm.error_function.delta_5,rmm.error_function.delta_5)


# %%
res6.x.view('complex128')

# %%
residues

# %%
data_1.view('float').shape

# %%
data_1.shape

# %%
true_value_1

# %%
