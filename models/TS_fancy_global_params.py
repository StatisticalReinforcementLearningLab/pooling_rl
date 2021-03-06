import pickle
import participant
import random
import numpy as np
import psi

class TS_fancy_global_params:
    
    
    
    '''
    Keeps track of hyper-parameters for any TS procedure. 
    '''
    
    def __init__(self,xi=10,context_dimension=2):
        self.nums = set([np.float64,int,float])
        self.pi_max = .8
        self.pi_min = .1
        self.sigma = 1
        
        
        self.xi  = xi
        
 
        self.gamma_mdp = .9 
        self.lambda_knot = .9 
        self.prob_sedentary = .9 
        self.weight = .5 
        
        self.z_index = [i+1 for i in range(context_dimension)]
        self.x_index = len(self.z_index)+2
        self.avail_index = self.x_index+1
        self.action_index = self.avail_index+1
        self.prob_index = self.action_index+1
        self.reward_index = self.prob_index+1
        self.psi = psi.psi()
    
    
        
    def feat0_function(self,z,x):
        
        
        temp =  [1]
        temp.extend(z)
        #print(type(x))
        if type(x) in self.nums:
        
            temp.append(x)
        else:
            temp.extend(x)
        return temp

    def feat1_function(self,z,x):
        temp =  [1]
        temp.extend(z)
        if type(x) in self.nums:
        
            temp.append(x)
        else:
            temp.extend(x)
        return temp    
        
        
    def feat2_function(self,z,x):
        temp = [1,z[0]]
        if type(x) in self.nums:
        
            temp.append(x)
        else:
            temp.extend(x)
        
        return temp
            
    def get_mu0(self,z_init):
        return [0 for i in range(len(self.feat0_function(z_init,0)))]
    
    def get_mu1(self,z_init):
        return [0 for i in range(len(self.feat1_function(z_init,0)))]
    
    def get_mu2(self,z_init):
        return [0 for i in range(len(self.feat2_function(z_init,0)))]
    
    def get_asigma(self,adim):
        return np.diag([10 for i in range(adim)])
