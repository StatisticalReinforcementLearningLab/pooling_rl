
import sys

import pickle
import pandas as pd
import numpy as np
import time
import os
import torch
import warnings
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.lazy import MatmulLazyTensor, RootLazyTensor
from gpytorch.constraints import constraints

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



class MyKernel(Kernel):
  
    
    def __init__(self, num_dimensions,user_mat, first_mat,gparams, variance_prior=None, offset_prior=None, active_dims=None):
        super(MyKernel, self).__init__(active_dims=active_dims)
        self.user_mat = user_mat
        self.first_mat = first_mat
       
      
    
    def forward(self, x1, x2, batch_dims=None, **params):
   
        return self.first_mat




class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,user_mat,first_mat,gparams):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        
        
        
        self.mean_module = gpytorch.means.ZeroMean()
       
        self.covar_module =  MyKernel(len(gparams.baseline_indices),user_mat,first_mat,gparams)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)









def run(X,users,y,global_params):
    #initial_u1,initial_u2,initial_rho,initial_noise,baseline_indices,psi_indices,user_index
    torch.manual_seed(111)
    #np.random.seed(111)
    user_mat= get_users(users,users)
    #print(user_mat.shape)
    #print(X.shape)
    #print(global_params.baseline_indices)
    first_mat = get_first_mat(np.eye(len(global_params.baseline_indices)),X,global_params.baseline_indices)
    #print(first_mat.shape)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.initialize(noise=(global_params.noise_term)*torch.ones(1))
    #print('going on')
    #print((global_params.noise_term)*torch.ones(X.shape[0]))
    # likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=(1.0)*torch.ones(X.shape[0]), learn_additional_noise=True)
    print('like worked')
    X = torch.from_numpy(np.array(X)).float()
    y = torch.from_numpy(y).float()
    #print(X.size())
    first_mat = torch.from_numpy(first_mat).float()
    user_mat = torch.from_numpy(user_mat).float()
    
    model = GPRegressionModel(X, y, likelihood,user_mat,first_mat,global_params)
    
    model.train()
    likelihood.train()
    sigma_u=None
    cov=None
    noise=None
    
    optimizer = torch.optim.Adam([
                                  {'params': model.parameters()},  # Includes GaussianLikelihood parameters
                                  ], lr=global_params.lr)
                                  
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        #def train(num_iter):
    num_iter=25
    with gpytorch.settings.use_toeplitz(False):
            for i in range(num_iter):
                try:
                   
                    optimizer.zero_grad()
                    output = model(X)
                #print(type(output))
                    loss = -mll(output, y)
                    loss.backward()
                    
                    print('Iter %d/%d - Loss: %.3f' % (i + 1, num_iter, loss.item()))
                    optimizer.step()
                    sigma_temp = get_sigma_u(model.covar_module.u1.item(),model.covar_module.u2.item(),model.covar_module.rho.item())
                    ##print('linalg {}'.format(np.linalg.eig(sigma_temp)))
                    
                    ##print(sigma_temp)
                    eigs = np.linalg.eig(sigma_temp)
                    f_preds = model(X)
                    f_covar = f_preds.covariance_matrix
                    covtemp = f_covar.detach().numpy()
                    noise = likelihood.noise_covar.noise.item()
               

                except Exception as e:
                    print(e)
                    print('here')
                    break

    if i<2:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
       
    likelihood.noise_covar.initialize(noise=(global_params.noise_term)*torch.ones(1))
        
        model = GPRegressionModel(X, y, likelihood,user_mat,first_mat,global_params)
       
        noise =global_params.noise_term
      
        f_preds = model(X)
        ##print('ok 5')
        f_covar = f_preds.covariance_matrix
     
        cov = f_covar.detach().numpy()


    return {'sigma_u':[],'cov':cov,'noise':noise,'like':0,'iters':i}


