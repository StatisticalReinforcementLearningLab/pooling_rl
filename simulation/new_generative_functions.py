
##get data
import numpy as np
import sys
sys.path
sys.path.append('../models')
import pickle
import math


class NewGen:
    
    
    
    '''
        Keeps track of hyper-parameters for any TS procedure.
        '''
    
    def __init__(self,a=None,b=None,beta=None):
        self.a = a
        self.b = b
        self.beta = beta


def get_add_one(self,action,state_vector):
    return action*np.dot(self.beta,state_vector)

def get_add_two(self,action,state_vector,Z):
    return action*(np.dot(self.beta,state_vector)+Z)

def get_add_three(self,action,state_vector,sigma):
    return action*(np.dot(self.beta,state_vector)+np.random.normal(0,sigma))

def get_additive(which='case_one',action,state_vector,Z=None,):
    if which=='case_one':
        extra = self.get_add_one(action,state_vector)
    elif which == 'case_two':
        extra = self.get_add_two(action,state_vector,Z)
    elif which=='case_three':
        extra = self.get_add_three(action,state_vector,sigma)
    else:
        return 'wrong case'


