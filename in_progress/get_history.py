import numpy as np
import pandas as pd
import math
import pickle 
import random


class get_history:
    
    
    
    '''
   Generates some data fairly, arbitrarly. 
    '''
    
    def __init__(self,dim_features,num_train_examples,num_test_examples=None):
        self.root =  '../../../../Volumes/dav/HeartSteps/pooling_rl_shared_data/processed/'
        
        self.dim_features =  dim_features
        self.num_train_examples =  num_train_examples
        self.num_test_examples = num_test_examples
      
     
        
    def state_function(self,mean=None,std=None):
    
    if mean==None and loc==None:
        mean = random.random()*10*(random.random()+3)

            
        std = 2
    return [[np.random.normal(mean,std) for i in range(self.dim_features)] \
            for n in range(self.train_num_examples)]

    def action_function(self):
        return [int(random.random()>.6) for i in range(self.train_num_examples)]
    
    def reward_function(self):
        return [np.random.normal(150,50) for i in range(self.train_num_examples)]