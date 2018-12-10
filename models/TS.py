import bandit
from numpy.random import uniform
from numpy.linalg import solve
import numpy as np
import state_params
import random
from scipy.stats import norm


def gen_nextdosage(x,a):
    anti_sed = int(uniform() < 0.3)
    event = (anti_sed+a)>0
    x_next = 0.9*x + event
    
    return x_next



def prob_cal_ts(z,x,mu,Sigma,init):
    pos_mean = np.dot(bandit.feat2_function(z,x),mu)
    pos_var = np.dot(np.dot(np.transpose(bandit.feat2_function(z,x)),Sigma),bandit.feat2_function(z,x))
    pos_var = max(0,pos_var)

  
 
  
    # probability
    pit_zero = norm.cdf((pos_mean)/(pos_var**.5))
  
    # clipping
    prob =  min(bandit.py_c_func(init.pi_max, max(bandit.py_c_func(init.pi_min, pit_zero))))
  
    return prob

def make_batch(t,Z,X,I,A,prob,R):
    temp = [t]
    temp.extend(Z)
    temp.extend([X,I,A,prob,R])
    return temp
