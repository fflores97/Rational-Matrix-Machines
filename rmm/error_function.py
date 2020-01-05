import numpy as np
from rmm import utils

def error(Z,Y,rho,poles,R,r,C,mu,lam,d_1,d_2):
    #sk_term = 0
    #for k in range(len(Z)):
    #    sk_term += rho[k]*np.linalg.norm(utils.rational_function(Z[k],poles,R,C) - utils.model(Z[k],poles,r,1)*Y[k])**2
    sk_term = sum([rho[k]*(np.linalg.norm(np.real(utils.model(z,poles,R,C)) - utils.rational_function_at_z(z,poles,r,1)*Y[k]))**2 for k, z in enumerate(Z)])
    tikhonov_term = mu*sum([np.linalg.norm(R[n])**2/d_1(Z,rho,pole) for n, pole in enumerate(poles)])
    lasso_term = lam*sum([np.linalg.norm(R[n])/d_2(Z,rho,pole) for n, pole in enumerate(poles)])

    return sk_term + tikhonov_term + lasso_term

def error_real(Z,Y,rho,poles,R,r,C,mu,lam,d_1,d_2):
    #sk_term = 0
    #for k in range(len(Z)):
    #    sk_term += rho[k]*np.linalg.norm(utils.rational_function(Z[k],poles,R,C) - utils.model(Z[k],poles,r,1)*Y[k])**2
    sk_term = sum([rho[k]*(np.linalg.norm(np.real(utils.model_real(z,poles,R,C)) - utils.model_real(z,poles,r,1)*Y[k]))**2 for k, z in enumerate(Z)])
    tikhonov_term = mu*sum([np.linalg.norm(R[n])**2/d_1(Z,rho,pole) for n, pole in enumerate(poles)])
    lasso_term = lam*sum([np.linalg.norm(R[n])/d_2(Z,rho,pole) for n, pole in enumerate(poles)])

    return sk_term + tikhonov_term + lasso_term

def delta_5(Z,rho,pole):

    value = sum([rho[k]/(np.abs(point - pole)**2) for k, point in enumerate(Z)])

    return 1/value