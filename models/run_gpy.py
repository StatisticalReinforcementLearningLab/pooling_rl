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
    new_data = data[:,[baseline_indices]].reshape((data.shape[0],data.shape[1]))

    new_data_two = data[:,[baseline_indices]].reshape((data.shape[0],data.shape[1]))
    result = np.dot(new_data,sigma_theta)

    results = np.dot(result,new_data_two.T)
    return results

def get_sigma_u(u1,u2,rho):
    off_diagaonal_term = u1**.5*u2**.5*(rho-1)
    return np.array([[u1,off_diagaonal_term],[off_diagaonal_term,u2]])

def run(X,users,y,global_params):
    #initial_u1,initial_u2,initial_rho,initial_noise,baseline_indices,psi_indices,user_index
    user_mat= get_users(users,users)
    #print(user_mat.shape)
    #print(X.shape)
    #print(global_params.baseline_indices)
    first_mat = get_first_mat(np.eye(len(global_params.baseline_indices)),X,global_params.baseline_indices)
    #print(first_mat.shape)
    
    kernel = GPy.kern.CustomKernel(len(global_params.baseline_indices),baseline_indices=global_params.baseline_indices,psi_indices=global_params.psi_indices,user_index=global_params.user_id_index,initial_u1=global_params.sigma_u[0][0],initial_u2=global_params.sigma_u[1][1],initial_rho=global_params.rho_term,initial_noise=global_params.noise_term,user_mat=user_mat,first_mat = first_mat)
    #print('initialized kernel')
    #print(X.shape)
    #print(y.shape)
    #print(y[0])
    m = GPy.models.GPRegression(X,y,kernel)
    #print('got m')

    m.Gaussian_noise.variance=global_params.noise_term

    m.optimize(max_iters=100)
    print('optimization worked')
    #sigma_u = get_sigma_u(m.kern.u1.values[0],m.kern.u2.values[0],m.kern.rho.values[0])
    #print('sigma u ')
    #print(sigma_u)
    
    sigma_u =global_params.sigma_u
    
    noise = m.Gaussian_noise.variance.values

    cov = m.kern.K(X)
    #print('cov')
    #print(cov)
    return {'sigma_u':sigma_u,'cov':cov,'noise':noise,'like':m.objective_function()}


def get_cov(X,y,global_params):
    #initial_u1,initial_u2,initial_rho,initial_noise,baseline_indices,psi_indices,user_index
    user_mat= get_users(X[:,global_params.user_id_index],X[:,global_params.user_id_index])
    
    first_mat = get_first_mat(np.eye(len(global_params.baseline_indices)),X,global_params.baseline_indices)
    
    kernel = GPy.kern.CustomKernel(len(global_params.baseline_indices),baseline_indices=global_params.baseline_indices,psi_indices=global_params.psi_indices,user_index=global_params.user_id_index,initial_u1=global_params.sigma_u[0][0],initial_u2=global_params.sigma_u[1][1],initial_rho=global_params.rho_term,initial_noise=global_params.noise_term,user_mat=user_mat,first_mat = first_mat)
    
    m = GPy.models.GPRegression(X,y,kernel)
    
    
    m.Gaussian_noise.variance=global_params.noise_term
    
    #m.optimize(max_iters=100)
    
    #sigma_u = get_sigma_u(m.kern.u1.values[0],m.kern.u2.values[0],m.kern.rho.values[0])
    
    #noise = m.Gaussian_noise.variance.values
    
    cov = m.kern.K(X)
    
    return {'sigma_u':global_params.sigma_u,'cov':cov,'noise':global_params.noise_term,'like':m.objective_function()}
