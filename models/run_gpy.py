import GPy
import sys

import pickle
import pandas as pd
import numpy as np
import time
import os

def get_users(users,userstwo):
        
        xx,yy = np.meshgrid(users,userstwo,sparse=True)
        #.99999999999
        return (xx==yy).astype('float')

def get_first_mat(sigma_theta,data,baseline_indices):
    new_data = data[:,[baseline_indices]].reshape((data.shape[0],data.shape[1]-1))

    new_data_two = data[:,[baseline_indices]].reshape((data.shape[0],data.shape[1]-1))
    result = np.dot(new_data,sigma_theta)

    results = np.dot(result,new_data_two.T)
    return results

def get_sigma_u(u1,u2,rho):
    off_diagaonal_term = u1**.5*u2**.5*(rho-1)
    return np.array([[u1,off_diagaonal_term],[off_diagaonal_term,u2]])

def run(X,y,initial_u1,initial_u2,initial_rho,initial_noise,baseline_indices,psi_indices,user_index):

    user_mat= get_users(X[:,user_index],X[:,user_index])

    first_mat = get_first_mat(np.eye(len(baseline_indices)),X,baseline_indices)

    kernel = GPy.kern.CustomKernel(len(baseline_indices),initial_u1=initial_u1,initial_u2= initial_u2,initial_rho=initial_rho,initial_noise= 1,psi_indices=psi_indices,baseline_indices=baseline_indices,user_mat=user_mat,first_mat = first_mat)

    m = GPy.models.GPRegression(X,y,kernel)


    m.Gaussian_noise.variance=initial_noise

    m.optimize(max_iters=100)

    sigma_u = get_sigma_u(m.kern.u1.values[0],m.kern.u2.values[0],m.kern.rho.values[0])

    noise = m.Gaussian_noise.variance.values

    cov = m.kern.K(X)

    return {'sigma_u':sigma_u,'cov':cov,'noise':noise,'like':m.objective_function()}
