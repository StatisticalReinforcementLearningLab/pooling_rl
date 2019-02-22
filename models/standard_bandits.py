import gpflow
import numpy as np
import math
import tensorflow as tf
import sys
import pickle
import pandas as pd
import CustomKernelStatic
import CustomKernel
from sklearn import preprocessing
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.simplefilter('ignore')
import os
import random
import sys
sys.path
sys.path.append('pooling_rl/simulation')
import sim_functions_cleaner as sf



def get_data_for_txt_effect_update(history,glob):
    all_data = []
    steps=[]
    probs = []
    actions = []
    
    ##might add pi to the user's history
    for user_id,history in history_dict.items():
        
        for hk,h in history.items():
            if h['avail'] and h['dt']:
                pi = h['prob']
          
                v.extend([h[i] for i in responsivity_features])
                steps.append(h['steps'])
                probs.append(pi)
                actions.append(h['action'])

    return np.array(all_data),np.array(steps),np.array(probs),np.array(actions)
