import numpy as np
import matplotlib.pyplot as plt

def calculate_cauchy_matrix(x,y):
    """
    Calculate Cauchy matrix for vectors x and y
    """
    length_x = len(x)
    length_y = len(y)

    cauchy_matrix = np.zeros((length_x, length_y))

    for ix in range(length_x):
        for iy in range(length_y):
            cauchy_matrix[ix, iy] = 1 / (x[ix] - y[iy])

    return cauchy_matrix

def calculate_true_value(residue, pole, x):
    value = calculate_cauchy_matrix(x,pole)
    value = value @ residue
    value = value.real

    return value

def generate_data(number_of_poles = 10,
                  relative_width_of_poles = 1e-2,
                  number_points_per_pole = 100,
                  signal_to_noise_ratio = 1e-3):
    
    """
    Function takes in parameters of data generation to generate data that resembles
    nuclear resonance data
    """

    # Average Resonance Parameters

    average_level_spacing = 2/(number_of_poles/4)
    average_neutron_width = 2/3*relative_width_of_poles * average_level_spacing
    average_capture_width = average_neutron_width/2
    degrees_of_freedom_for_capture_channels = 9e2*np.random.uniform()+1e2

    # Assign poles

    vector_of_poles = np.empty([0,1])
    vector_of_residues = np.empty((0,2))

    while len(vector_of_poles) < number_of_poles:
        
        m = np.zeros((2,2))
        m[0,0] = np.random.uniform()
        m[1,1] = np.random.uniform()
        m[0,1] = np.sqrt(1/2)*np.random.uniform()
        m[1,0] = m[0,1]
        s = np.sort(np.linalg.eig(m)[0])
        d = average_level_spacing*(s[1] - s[0])/2

        # Place first pole randomly if:
        # 1) we don't have any poles
        # 2) This pole would go beyond +1

        if np.size(vector_of_poles) == 0 or vector_of_poles[-1:] +d > 1:
            dPole = -1 + average_level_spacing * np.random.uniform()
        else:
            dPole = vector_of_poles[-1:] + d

        #vector_of_poles = np.append(vector_of_poles, dPole)

        # Sampling from Chi-squared distribution
        Gn = average_neutron_width * np.random.chisquare(1) 
        Gg = average_capture_width * np.random.chisquare(degrees_of_freedom_for_capture_channels)\
            /degrees_of_freedom_for_capture_channels
        Gt = Gn + Gg

        residues = [[Gn -1j*Gn*Gg/Gt, -1j*Gn*Gg/Gt]]

        pole = dPole + 1j*Gt
        vector_of_poles = np.append(vector_of_poles, pole)
        vector_of_residues = np.append(vector_of_residues, residues, axis = 0)

    number_of_poles = len(vector_of_poles)

    ##########
    #%%
    ##########

    number_of_data_points = number_points_per_pole * number_of_poles
    x = np.linspace(-1,1,number_of_data_points)

    C = 1e0 # Additive constant manually chosen to give positive true values
    
    # Rescale true values to generate Poisson noise
    scale = 1/(signal_to_noise_ratio**2) 

    # Linear scaling of true value
    true_value_1 = calculate_true_value(vector_of_residues[:, 0], vector_of_poles, x) + C
    true_value_1 = scale * true_value_1
    true_value_2 = calculate_true_value(vector_of_residues[:, 1], vector_of_poles, x) + C
    true_value_2 = scale * true_value_2

    ##########
    #%% Noise model
    ##########

    data_1 = np.zeros(np.shape(true_value_1))
    data_2 = np.zeros(np.shape(true_value_2))

    for ix in range(number_of_data_points):
        data_1[ix] = np.random.poisson(true_value_1[ix])
        data_2[ix] = np.random.poisson(true_value_2[ix])

    # Rescale back

    true_value_1 = true_value_1/scale
    true_value_2 = true_value_2/scale

    data_1 = data_1/scale
    data_2 = data_2/scale

    return [x, data_1, data_2, true_value_1, true_value_2, vector_of_poles, vector_of_residues]
    

