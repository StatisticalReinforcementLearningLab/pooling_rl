
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
       
        self.psi_dim_one = gparams.psi_indices[0]
        self.psi_dim_two = gparams.psi_indices[1]
        self.psi_indices =gparams.psi_indices
        #print(self.psi_dim_one)
        #print(self.psi_dim_two)
        
        #init_u1 = gparams.sigma_u[0][0]
        
        print(gparams.u1)
        print(gparams.u2)
        self.init_u1 = gparams.u1*torch.tensor(1.0)
        
        #init_u2 = gparams.sigma_u[1][1]
        self.init_u2 = gparams.u2*torch.tensor(1.0)
        self.init_u3 = gparams.u3*torch.tensor(1.0)
        self.init_u4 = gparams.u4*torch.tensor(1.0)
        
        self.r12 = gparams.r12*torch.tensor(1.0)
        self.r13 = gparams.r13*torch.tensor(1.0)
        self.r14 = gparams.r14*torch.tensor(1.0)
        
        self.r23 = gparams.r23*torch.tensor(1.0)
        self.r24 = gparams.r24*torch.tensor(1.0)
        
        self.r34 = gparams.r34*torch.tensor(1.0)
        
        #self.register_parameter(name="u1", parameter=torch.nn.Parameter(init_u1*torch.tensor(1.0)))
        self.register_parameter(name="raw_u1", parameter=torch.nn.Parameter(self.init_u1*torch.tensor(1.0)))
        
        #self.register_parameter(name="u2", parameter=torch.nn.Parameter(init_u2*torch.tensor(1.0)))
        self.register_parameter(name="raw_u2", parameter=torch.nn.Parameter(self.init_u2*torch.tensor(1.0)))
        
        #self.register_parameter(name="u3", parameter=torch.nn.Parameter(init_u3*torch.tensor(1.0)))
        self.register_parameter(name="raw_u3", parameter=torch.nn.Parameter(self.init_u3*torch.tensor(1.0)))
        
        #self.register_parameter(name="u4", parameter=torch.nn.Parameter(init_u4*torch.tensor(1.0)))
        self.register_parameter(name="raw_u4", parameter=torch.nn.Parameter(self.init_u4*torch.tensor(1.0)))
        
        
        #t =gparams.sigma_u[0][0]**.5 * gparams.sigma_u[1][1]**.5
        #r = (gparams.sigma_u[0][1]+t)/t
        r = gparams.rho_term
        #self.register_parameter(name="rho_12", parameter=torch.nn.Parameter(r12*torch.tensor(1.0)))
        self.register_parameter(name="raw_rho_12", parameter=torch.nn.Parameter(self.r12*torch.tensor(1.0)))
        
        #self.register_parameter(name="rho_13", parameter=torch.nn.Parameter(r13*torch.tensor(1.0)))
        self.register_parameter(name="raw_rho_13", parameter=torch.nn.Parameter(self.r13*torch.tensor(1.0)))
        
        #self.register_parameter(name="rho_14", parameter=torch.nn.Parameter(r14*torch.tensor(1.0)))
        self.register_parameter(name="raw_rho_14", parameter=torch.nn.Parameter(self.r14*torch.tensor(1.0)))

        #self.register_parameter(name="rho_23", parameter=torch.nn.Parameter(r23*torch.tensor(1.0)))
        self.register_parameter(name="raw_rho_23", parameter=torch.nn.Parameter(self.r23*torch.tensor(1.0)))

        #self.register_parameter(name="rho_24", parameter=torch.nn.Parameter(r24*torch.tensor(1.0)))
        self.register_parameter(name="raw_rho_24", parameter=torch.nn.Parameter(self.r24*torch.tensor(1.0)))

        #self.register_parameter(name="rho_34", parameter=torch.nn.Parameter(r34*torch.tensor(1.0)))
        self.register_parameter(name="raw_rho_34", parameter=torch.nn.Parameter(self.r34*torch.tensor(1.0)))
        
        self.register_constraint("raw_u1",constraint= constraints.Positive())
        self.register_constraint("raw_u2",constraint= constraints.Positive())
        self.register_constraint("raw_u3",constraint= constraints.Positive())
        self.register_constraint("raw_u4",constraint= constraints.Positive())

        self.register_constraint("raw_rho_12",constraint= constraints.Interval(0,2))
        self.register_constraint("raw_rho_13",constraint= constraints.Interval(0,2))
        self.register_constraint("raw_rho_14",constraint= constraints.Interval(0,2))
        self.register_constraint("raw_rho_23",constraint= constraints.Interval(0,2))
        self.register_constraint("raw_rho_24",constraint= constraints.Interval(0,2))
        self.register_constraint("raw_rho_34",constraint= constraints.Interval(0,2))
    
    
    
        self.u1 = gparams.u1
        
        #init_u2 = gparams.sigma_u[1][1]
        self.u2 = gparams.u2
        self.u3 = gparams.u3
        self.u4 = gparams.u4
        
        self.rho_12 = gparams.r12
        self.rho_13 = gparams.r13
        self.rho_14 = gparams.r14
        
        self.rho_23 = gparams.r23
        self.rho_24 = gparams.r24
        
        self.rho_34 = gparams.r34
        print('initialized')
    
    #self.register_prior("u1_prior", gpytorch.priors.SmoothedBoxPrior(a=0,b=10,sigma=1), "u1")
    #self.register_prior("u2_prior", gpytorch.priors.SmoothedBoxPrior(a=0,b=10,sigma=1), "u2")
    #self.register_prior("rho_prior", gpytorch.priors.SmoothedBoxPrior(a=0,b=2,sigma=.5), "rho")
    
    def forward(self, x1, x2, batch_dims=None, **params):
        
        #us = torch.cat([self.u1, self.u2], 0) # us is a vector of size 2
        #print(x1[0,:,0:2].size())
        # print(x1.size())
        #print(us.size())
        #x1_ =torch.stack((x1[:,self.psi_dim_one],x1[:,self.psi_dim_two]),dim=1)
        x1_ = torch.stack([x1[:,i] for  i in self.psi_indices],dim=1)
        #x1_ =    torch.stack((x1[:,self.psi_dim_one],x1[:,self.psi_dim_two]),dim=1)
        #x2_ =torch.stack((x2[:,self.psi_dim_one],x2[:,self.psi_dim_two]),dim=1)
        x2_ =    torch.stack([x2[:,i] for  i in self.psi_indices],dim=1)
        #print(x1_)
        #print(x2_)
        #u2_= self.u2
        #u1_ =self.u1
        #print(self.u1)
        #print(x1_)
        #print(x2_)
        if batch_dims == (0, 2):
            print('batch bims here')
        #pass
        #print(x1_.size())
        
        #x1_ = x1_.view(x1_.size(0), x1_.size(1), -1, 1)
        #x1_ = x1_.permute(0, 2, 1, 3).contiguous()
        #x1_ = x1_.view(-1, x1_.size(-2), x1_.size(-1))
        
        
        #x2_ = x2_.view(x2_.size(0), x2_.size(1), -1, 1)
        #x2_ = x2_.permute(0, 2, 1, 3).contiguous()
        #x2_ = x2_.view(-1, x2_.size(-2), x2_.size(-1))
        #print(x1_.size())
        #print(x2_.size())
        #prod = MatmulLazyTensor(x1_, x2_.transpose(1, 0))
        
        prod = MatmulLazyTensor(x1_[:,0:1], x2_[:,0:1].transpose(-1, -2))
        
        
        #.expand(1,100,100)
        tone = prod * (self.u1)
        
        
        prod = MatmulLazyTensor(x1_[:,1:2], x2_[:,1:2].transpose(-1, -2))
        
        ttwo = prod * (self.u2)
        
        prod = MatmulLazyTensor(x1_[:,2:3], x2_[:,2:3].transpose(-1, -2))
        
        tthree = prod * (self.u3)
        
        prod = MatmulLazyTensor(x1_[:,3:4], x2_[:,3:4].transpose(-1, -2))
        
        tfour = prod * (self.u4)
        
        
        diagone = MatmulLazyTensor(x1_[:,0:1], x2_[:,1:2].transpose(-1, -2))
        diagtwo = MatmulLazyTensor(x1_[:,1:2], x2_[:,0:1].transpose(-1, -2))
        cov_12 = (diagone+diagtwo)*((self.rho_12-1)*(self.u1)**.5*(self.u2)**.5)
        
        diagone = MatmulLazyTensor(x1_[:,0:1], x2_[:,2:3].transpose(-1, -2))
        diagtwo = MatmulLazyTensor(x1_[:,2:3], x2_[:,0:1].transpose(-1, -2))
        cov_13 = (diagone+diagtwo)*((self.rho_13-1)*(self.u1)**.5*(self.u3)**.5)

        diagone = MatmulLazyTensor(x1_[:,0:1], x2_[:,3:4].transpose(-1, -2))
        diagtwo = MatmulLazyTensor(x1_[:,3:4], x2_[:,0:1].transpose(-1, -2))
        cov_14 = (diagone+diagtwo)*((self.rho_14-1)*(self.u1)**.5*(self.u4)**.5)

        diagone = MatmulLazyTensor(x1_[:,1:2], x2_[:,2:3].transpose(-1, -2))
        diagtwo = MatmulLazyTensor(x1_[:,2:3], x2_[:,1:2].transpose(-1, -2))
        cov_23 = (diagone+diagtwo)*((self.rho_23-1)*(self.u2)**.5*(self.u3)**.5)

        diagone = MatmulLazyTensor(x1_[:,1:2], x2_[:,3:4].transpose(-1, -2))
        diagtwo = MatmulLazyTensor(x1_[:,3:4], x2_[:,1:2].transpose(-1, -2))
        cov_24 = (diagone+diagtwo)*((self.rho_24-1)*(self.u2)**.5*(self.u4)**.5)

        diagone = MatmulLazyTensor(x1_[:,2:3], x2_[:,3:4].transpose(-1, -2))
        diagtwo = MatmulLazyTensor(x1_[:,3:4], x2_[:,2:3].transpose(-1, -2))
        cov_34 = (diagone+diagtwo)*((self.rho_34-1)*(self.u3)**.5*(self.u4)**.5)
        
        random_effects = tone+ttwo+tthree+tfour+cov_12+cov_13+cov_14+cov_23+cov_24+cov_34
        
        #print(random_effects.evaluate())
        
        #print(random_effects)
        
        #print(random_effects.size())
        #print(self.user_mat.size())
        final = random_effects*self.user_mat
        
        #print(final.evaluate())
        #noise_term = (self.noise**2)*self.noise_mat
        #print(type(noise_term))
        #print(noise_term)
        #prod = MatmulLazyTensor(x1_, x2_.transpose(-1, -2))
        #prod = MatmulLazyTensor(prod,noise_term)
        #prod = prod*self.user_mat
        
        #final  = final + noise_term
        
        #final = torch.stack((tone,ttwo,tone,ttwo),dim=0)
        #print('one')
        #print(random_effects.evaluate())
        #print('two')
        #print(final.evaluate())
        #print(MatmulLazyTensor(random_effects,2*torch.eye(100)).evaluate())
        
        #n = self.first_mat
        #+noise_term
        
        
        final = final+self.first_mat
        #print(final.evaluate())
        return final

    @property
    def u2(self):
        return self.raw_u2_constraint.transform(self.raw_u2)
    
    @u2.setter
    def u2(self, value):
        self._set_u2(value)
    
    def _set_u2(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_u2)
            self.initialize(raw_u2=self.raw_u2_constraint.inverse_transform(value))

    @property
    def u1(self):
        return self.raw_u1_constraint.transform(self.raw_u1)
    
    @u1.setter
    def u1(self, value):
        self._set_u1(value)
    
    def _set_u1(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_u1)
            self.initialize(raw_u1=self.raw_u1_constraint.inverse_transform(value))

    @property
    def u3(self):
        return self.raw_u3_constraint.transform(self.raw_u3)
    
    @u3.setter
    def u3(self, value):
        self._set_u3(value)
    
    def _set_u3(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_u3)
            self.initialize(raw_u3=self.raw_u3_constraint.inverse_transform(value))

    @property
    def u4(self):
        return self.raw_u4_constraint.transform(self.raw_u4)
    
    @u4.setter
    def u4(self, value):
        self._set_u4(value)
    
    def _set_u4(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_u4)
            self.initialize(raw_u4=self.raw_u4_constraint.inverse_transform(value))
    
    
    
    @property
    def rho_12(self):
        return self.raw_rho_12_constraint.transform(self.raw_rho_12)
    
    @rho_12.setter
    def rho_12(self, value):
        self._set_rho_12(value)
    
    def _set_rho_12(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_rho_12)
            self.initialize(raw_rho_12=self.raw_rho_12_constraint.inverse_transform(value))

    @property
    def rho_13(self):
        return self.raw_rho_13_constraint.transform(self.raw_rho_13)
    
    @rho_13.setter
    def rho_13(self, value):
        self._set_rho_13(value)
    
    def _set_rho_13(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_rho_13)
            self.initialize(raw_rho_13=self.raw_rho_13_constraint.inverse_transform(value))

    @property
    def rho_14(self):
        return self.raw_rho_14_constraint.transform(self.raw_rho_14)
    
    @rho_14.setter
    def rho_14(self, value):
        self._set_rho_14(value)
    
    def _set_rho_14(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_rho_14)
            self.initialize(raw_rho_14=self.raw_rho_14_constraint.inverse_transform(value))

    @property
    def rho_23(self):
        return self.raw_rho_23_constraint.transform(self.raw_rho_23)
    
    @rho_23.setter
    def rho_23(self, value):
        self._set_rho_23(value)
    
    def _set_rho_23(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_rho_23)
            self.initialize(raw_rho_23=self.raw_rho_23_constraint.inverse_transform(value))

    @property
    def rho_24(self):
        return self.raw_rho_24_constraint.transform(self.raw_rho_24)
    
    @rho_24.setter
    def rho_24(self, value):
        self._set_rho_24(value)
    
    def _set_rho_24(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_rho_24)
            self.initialize(raw_rho_24=self.raw_rho_24_constraint.inverse_transform(value))

    @property
    def rho_34(self):
        return self.raw_rho_34_constraint.transform(self.raw_rho_34)
    
    @rho_34.setter
    def rho_34(self, value):
        self._set_rho_34(value)
    
    def _set_rho_34(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_rho_34)
            self.initialize(raw_rho_34=self.raw_rho_34_constraint.inverse_transform(value))



class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,user_mat,first_mat,gparams):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        
        # SKI requires a grid size hyperparameter. This util can help with that
        # We're setting Kronecker structure to False because we're using an additive structure decomposition
        #grid_size = gpytorch.utils.grid.choose_grid_size(train_x, kronecker_structure=False)
        
        self.mean_module = gpytorch.means.ZeroMean()
        #self.mean_module.constant.requires_grad=False
        self.covar_module =  MyKernel(len(gparams.baseline_indices),user_mat,first_mat,gparams)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)









def run(X,users,y,global_params):
    #initial_u1,initial_u2,initial_rho,initial_noise,baseline_indices,psi_indices,user_index
    torch.manual_seed(1e6)
    user_mat= get_users(users,users)
    #print(user_mat.shape)
    #print(X.shape)
    #print(global_params.baseline_indices)
    first_mat = get_first_mat(np.eye(len(global_params.baseline_indices)),X,global_params.baseline_indices)
    #print(first_mat.shape)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    print('noise one')
    likelihood.noise_covar.initialize(noise=(global_params.o_noise_term)*torch.ones(1))
    print('noise two')
    X = torch.from_numpy(np.array(X)).float()
    y = torch.from_numpy(y).float()
    #print(X.size())
    first_mat = torch.from_numpy(first_mat).float()
    print('probs')
    user_mat = torch.from_numpy(user_mat).float()
    print('probs')
    model = GPRegressionModel(X, y, likelihood,user_mat,first_mat,global_params)
    
    model.train()
    likelihood.train()
    sigma_u=None
    cov=None
    noise=None
    
    optimizer = torch.optim.Adam([
                                  {'params': model.parameters()},  # Includes GaussianLikelihood parameters
                                  ], lr=0.1)
                                  
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        #def train(num_iter):
    num_iter=50
    with gpytorch.settings.use_toeplitz(False):
            for i in range(num_iter):
                try:
                    optimizer.zero_grad()
                    output = model(X)
                #print(type(output))
                    loss = -mll(output, y)
                    loss.backward()
                    #print('Iter %d/%d - Loss: %.3f' % (i + 1, num_iter, loss.item()))
                    optimizer.step()
                    #sigma_temp = get_sigma_u(model.covar_module.u1.item(),model.covar_module.u2.item(),model.covar_module.rho.item())
                    sigma_temp = [model.covar_module.u1.item(),model.covar_module.u2.item(),model.covar_module.u3.item(),model.covar_module.u4.item(),model.covar_module.rho_12.item(),model.covar_module.rho_13.item(),model.covar_module.rho_14.item(),model.covar_module.rho_23.item(),model.covar_module.rho_24.item(),model.covar_module.rho_34.item()]
                    f_preds = model(X)
                    f_covar = f_preds.covariance_matrix
                    covtemp = f_covar.detach().numpy()
                    if np.isreal(sigma_temp).all() and not np.isnan(covtemp).all():
                        sigma_u = sigma_temp
                        cov=covtemp
                        #print(np.isreal( covtemp))
                        #print(cov)
                        noise = likelihood.noise_covar.noise.item()


                except Exception as e:
                    print(e)
                    print('here')
                    break
#train(50)
    
    

    print('cov')
    print(cov)
    print(sigma_u)
    return {'uparams':sigma_u,'cov':cov,'noise':noise,'like':0}


