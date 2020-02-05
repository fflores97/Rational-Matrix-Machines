#! /usr/bin/env python3
"""
Functions for the pole filtering step of RMM
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import argparse
from rmm import generate_data
from rmm import utils
from rmm import VF


## Proximal operator for L1-L2 group regularization
def prox_lambda(A,lam):
    return A*max(0,1-lam/np.linalg.norm(A))

# Define different modifications to Lasso regularizer

def one_lasso(z_train, poles, Y_train = ()):
    return 1

def delta_min_distance(z_train, poles, Y_train = ()):
    min_distances_z_p = np.zeros(len(poles), dtype=complex )
    for p in range(poles.size):
        min_distances_z_p[p] = np.abs(poles[p] - z_train[0])
        for k in range(z_train.size):
            if np.abs(poles[p] - z_train[k]) < min_distances_z_p[p] :
                min_distances_z_p[p] = np.abs(poles[p] - z_train[k])
    return min_distances_z_p

def delta_min_distance_sq(z_train,poles, Y_train = ()):
    value = delta_min_distance(z_train,poles)
    return value**2

def delta_learning_rate(z_train, poles, Y_train=()):
    Delta = np.zeros(len(poles) , dtype=complex )
    sqrt_weights_rho = build_square_weights_rho_k(z_train) 
    for p in range(poles.size):
        gamma_R_p_inv = 0
        for k in range(z_train.size):
            gamma_R_p_inv += 2*(sqrt_weights_rho[k]/np.abs(z_train[k] - poles[p]) )**2
        Delta[p] = (1/gamma_R_p_inv)**0.5
    return Delta 

def delta_learning_rate_sq(z_train, poles, Y_train=()):
    value = delta_learning_rate(z_train, poles, Y_train)
    return value**2

def delta_average_pole(z_train, poles, Y_train = ()):
    Delta = np.zeros(len(poles) , dtype=complex )
    rho = build_square_weights_rho_k(z_train)**2 
    for p in range(poles.size):
        temp = 0
        for k in range(z_train.size):
            temp += rho[k]/np.abs(z_train[k] - poles[p])
        Delta[p] = 1/temp
    return Delta

def delta_average_sq_pole(z_train, poles, Y_train = ()):
    Delta = np.zeros(len(poles) , dtype=complex )
    rho = build_square_weights_rho_k(z_train)**2 
    for p in range(poles.size):
        temp = 0
        for k in range(z_train.size):
            temp += (rho[k]/np.abs(z_train[k] - poles[p]))**2
        Delta[p] = 1/temp
    return Delta 

def delta_average_pole_sq(z_train, poles, Y_train = ()):
    value = delta_average_pole(z_train, poles, Y_train)
    return value**2


def delta_inv_im_real(z_train, poles, Y_train = ()):
    Delta = np.zeros(len(poles) , dtype=complex )
    for p, pole in enumerate(poles):
        Delta[p] = 1/np.real(pole) + 1/np.imag(pole)
    return Delta 


def delta_inv_im_real_sq(z_train, poles, Y_train = ()):
    value = delta_inv_im_real(z_train, poles, Y_train)
    return value**2

def RMM_pole_filtering(delta_function,z_train, Y_train, poles , lam=0 , mu=0, num_PFBS_iter = 2000 , ε = 1.0e-8):
    ## build the LS vectors and matrix
    number_poles = poles.size
    dim_residues = Y_train[0].size
    Y_LS_vector = VF.vectorize_Y_for_LS(z_train, Y_train, dim_residues)
    Z = VF.build_LS_matrix(z_train, Y_train, poles)
    ## calculate the least distances from z_k to poles:
    Delta = delta_function(z_train, poles, Y_train)
    ## calculate the gamma step size
    ZZ = np.tensordot(Z.conj().T,Z,1)
    eigenvals , eigenvects = np.linalg.eig(ZZ)
    gamma = 1/(2*np.amax(eigenvals))
    ## condition number of this system
    Condition_number = np.amax(eigenvals)/np.amin(eigenvals)
    ## initiatilze the PFBS descent with the LS residues
    LS_vector, LS_residual, LS_rank , LS_singular_values = np.linalg.lstsq(Z, Y_LS_vector)
    PFBS_vector = LS_vector
    ## start PFBS iterations
    PFBS_iter_num = 0
    for i in range(num_PFBS_iter):
        ## report the old vector
        PFBS_old_residues, PFBS_old_offset = VF.extract_from_LS_vector(PFBS_vector,number_poles, dim_residues)
        ## iteration count
        PFBS_iter_num += 1 
        ## compute the Gradient
        DeltaE = 2*np.tensordot(Z.conj().T, (np.tensordot(Z,PFBS_vector, 1) - Y_LS_vector), 1)
        ## Add the Tichonov regularization on the residues
        DeltaEmu = DeltaE 
        DeltaEmu[:dim_residues*number_poles] = DeltaE[:dim_residues*number_poles] + 2*mu*PFBS_vector[:dim_residues*number_poles]
        ## take the Gradient descent step
        GD_new_PFBS_vector = PFBS_vector - gamma*DeltaEmu
        GD_new_residues, GD_new_offset = VF.extract_from_LS_vector(GD_new_PFBS_vector, number_poles, dim_residues)
        ## take the PFBS step for group LASSO regularization
        PFBS_new_residues = GD_new_residues
        DeltaL2_relative_step_size = np.zeros([number_poles])
        for p in range(number_poles):
            PFBS_new_residues[p] = prox_lam(GD_new_residues[p],lam/Delta[p])
            ## calculate the PFBS step size for convergence criteria
            if np.linalg.norm(PFBS_new_residues[p]) == 0:
                DeltaL2_relative_step_size[p] = 0
            else: 
                DeltaL2_relative_step_size[p] = np.linalg.norm(PFBS_new_residues[p] - PFBS_old_residues[p])/(np.linalg.norm(PFBS_new_residues[p]))
        ## update PFBS_vector
        PFBS_vector = build_LS_vector(PFBS_new_residues, GD_new_offset)
        ## convergence criteria on the relative step size 
        if np.amax(DeltaL2_relative_step_size) < ε:
            print("The lam regularization parameter is lam =", lam, "the maximum relative step sizes in norm is max(DeltaL2_relative_step_size) =" , np.amax(DeltaL2_relative_step_size), "for threshold ε =", ε, "and the PFBS iterations are breaking after PFBS_iter_num =", PFBS_iter_num, "iterations")
            break
    PFBS_residues, PFBS_offset = extract_from_LS_vector(PFBS_vector, number_poles, dim_residues)
    return PFBS_residues, PFBS_offset , gamma , Condition_number, PFBS_iter_num



def main(path):
    z_train, Y_train_1, Y_true_1, Y_train_2, Y_true_2, true_poles, true_residues, \
    true_offset, number_true_poles, number_train_points = VF.prepare_data(path)

    regularizing_functions = [delta_min_distance, delta_min_distance_sq,delta_average_pole,delta_average_sq_pole,delta_average_pole_sq,delta_learning_rate,delta_learning_rate_sq,delta_inv_im_real]
    lambdas = [1e-9, 1e-12, 5*1e-8, 1e-8, 1e-8,1e-8,1e-8,1e-8]
    Y_output = []
    for jj, regularizer in enumerate(regularizing_functions):
        RMM_residues, RMM_offset , gamma , RMM_Condition_number, RMM_iter_num = RMM_pole_filtering(regularizer,z_train, Y_train_1, VF_poles , lambdas[jj], 0.0)
        ## Pole filtered results
        z_train  ## for complex values : 2*np.random.rand(number_CV_points)*np.exp(1j*2*np.pi*np.random.rand(number_CV_points))
        dim_residues = Y_train[0].size
        Y_RMM = np.zeros([z_train.size, dim_residues] , dtype=complex) 
        for k in range(z_train.size):
            Y_RMM[k] = rational_function(z_train[k], VF_poles, RMM_residues, RMM_offset )
        Y_output.append(Y_RMM)