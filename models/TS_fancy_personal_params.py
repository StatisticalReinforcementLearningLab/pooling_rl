import pickle
import participant
import random
import numpy as np

class TS_fancy_personal_params:
    
    
    
    '''
    Keeps track of parameters for each person in the population. 
    '''
    
    def __init__(self):
        
        
        
        self.mus0 = {}
        self.sigmas0 = {}
        
        self.mus1 = {}
        self.sigmas1 = {}
        
        self.mus2 = {}
        self.sigmas2 = {}
        
        self.batch = {}    
        self.batch_index = {}
    
        self.etas = {}
    
        self.last_update = {}
    
        
    def update_mus(self,pid,mu_value,which_mu):
        if which_mu==0:
            self.mus0[pid]=mu_value
            
        if which_mu==1:
            self.mus1[pid]=mu_value
            
        if which_mu==2:
            self.mus2[pid]=mu_value
            
    def update_sigmas(self,pid,sigma_value,which_sigma):
        if which_sigma==0:
            self.sigmas0[pid]=sigma_value
            
        if which_sigma==1:
            self.sigmas1[pid]=sigma_value
            
        if which_sigma==2:
            self.sigmas2[pid]=sigma_value
    
    def update_batch(self,pid,context):
   
        
        #context: [t,Z,X,I,A,prob,R]
        
        temp = [context[0]]
        temp.extend(context[1])
        temp.extend(context[1:])
        #print(self.batch_index[pid])
        self.batch[pid][self.batch_index[pid]]=temp
        self.batch_index[pid]=self.batch_index[pid]+1

        
