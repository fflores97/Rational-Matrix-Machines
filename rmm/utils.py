import numpy as np
from rmm import vectfit

def model(z,poles,residues,offset=0):
    mod = sum([residues[n]/(z-poles[n]) for n in range(len(poles))]) + offset
    return mod

def rational_function(z, poles, residues,offset=0):
    """
    Vectorize model function
    """
    
    # mod = np.vectorize(model, excluded=[1,2,3], otypes=['complex128'])

    # # return value
    # return mod(z, poles, residues, offset)

    value = np.array([model(point, poles, residues, offset) for point in z])

    return value