import gpflow
import numpy as np
import math
import tensorflow as tf
import sys
import pickle
import pandas as pd
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.simplefilter('ignore')


def gather_cols(params, indices, name=None):
    with tf.op_scope([params, indices], name, "gather_cols") as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')
        
        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],[-1, 1]) + indices, [-1])
        return tf.dtypes.cast(tf.reshape(tf.gather(p_flat, i_flat),[p_shape[0], -1]),dtype=tf.float64)

def get_theta(dim_baseline):
    m = 2*np.eye(dim_baseline)
    #m = np.add(m,.1)
    return m

class CustomKernel(gpflow.kernels.Kernel):
    def __init__(self,input_dim, mysession=None,rhos=None,select_users=None, active_dims=None, ARD=None, name=None,baseline_indices = None,psi_indices=None,user_index = None,user_day_index = None,num_data_points = None,initial_u1=1.0,initial_u2=1.0,initial_s1=1,initial_s2=5,initial_rho=.9,initial_noise=1.0):
        super().__init__(input_dim)
        
        #print('IN KERNEL')
        #print(initial_u1)
        #print(initial_u2)
        #print(initial_s1)
        #print(initial_rho)
        ##make this a function
        theta = get_theta(len(baseline_indices)).reshape(1,len(baseline_indices),len(baseline_indices))
        
        #sigmau = tf.reshape(np.array([[1.0,0.1],[0.1,1.0]]),(1,2,2))
        sigmav = np.array([[initial_s1,0.0],[0.0,initial_s2]]).reshape(1,2,2)
        
        #self.sigma_u = gpflow.Param(value = np.array([[1.0,0.1],[0.1,1.0]]).reshape(1,2,2),\transform=gpflow.transforms.DiagMatrix(2)(gpflow.transforms.positive))
        #self.sigma_u1 = gpflow.Param(1.0, transform=gpflow.transforms.positive,dtype=gpflow.settings.float_type)
        #self.sigma_u2 = gpflow.Param(1.0, transform=gpflow.transforms.positive, dtype=gpflow.settings.float_type)
        #self.sigma_rho =gpflow.Param(1.0, transform=gpflow.transforms.Logistic(a=0,b=2), dtype=gpflow.settings.float_type)
        
        #tf.constant(np.array([[1.0,0.1],[0.1,1.0]]))
        
        self.sigma_theta = tf.constant(theta)
        #gpflow.Param(theta, transform=gpflow.transforms.DiagMatrix(6)(gpflow.transforms.positive),
        #                           dtype=gpflow.settings.float_type,fix_shape=True)
        #print(self.sigma_theta)
        #gpflow.Param(1.0, transform=gpflow.transforms.positive,
        #dtype=gpflow.settings.float_type)
        
        self.sigma_v =  gpflow.Param(sigmav, transform=gpflow.transforms.DiagMatrix(2)(gpflow.transforms.positive),dtype=gpflow.settings.float_type)
        
        
        #self.noise_term = gpflow.Param(1.0, transform=gpflow.transforms.positive,  dtype=gpflow.settings.float_type)
        
        
        
        #sigmav = np.array([[10.0,0.0],[0.0,10.0]]).reshape(1,2,2)
        #self.sigma_v =  gpflow.Param(sigmav, transform=gpflow.transforms.DiagMatrix(2)(gpflow.transforms.positive),dtype=gpflow.settings.float_type)
        
        
        self.sigma_u1 = gpflow.Param(initial_u1,dtype=gpflow.settings.float_type,transform=gpflow.transforms.positive)
        self.sigma_u2 = gpflow.Param(initial_u2,dtype=gpflow.settings.float_type,transform=gpflow.transforms.positive)
        self.sigma_rho =gpflow.Param(initial_rho, transform=gpflow.transforms.Logistic(a=0,b=2), dtype=gpflow.settings.float_type)


        self.sigma_theta = tf.constant(theta)
        #gpflow.Param(theta, transform=gpflow.transforms.DiagMatrix(6)(gpflow.transforms.positive),
        #                           dtype=gpflow.settings.float_type,fix_shape=True)
        #print(self.sigma_theta)
        #gpflow.Param(1.0, transform=gpflow.transforms.positive,
        #dtype=gpflow.settings.float_type)
        
        #self.sigma_v =  tf.constant(sigmav)
        
        
        self.noise_term = gpflow.Param(initial_noise ,dtype=gpflow.settings.float_type)
        
        
                                    
        
        self.select_users = tf.constant(select_users)
                                    
        self.baseline_indices = baseline_indices
        self.psi_indices = psi_indices
                                    
        self.num_data_points = num_data_points
        self.user_index = user_index
        self.user_day_index = user_day_index
        self.mysession=mysession
        self.rhos = tf.constant(rhos)
##will it freak out if there is no parameter?


    @gpflow.params_as_tensors
    def rbf_custom(self, X, X2=None):
        #print(X)
        #print(X2)
        if X2 is None:
            X2=X
        return tf.exp(-tf.divide(tf.square(tf.subtract(X,X2)),tf.constant(100.0,dtype=tf.float64)))
        #return tf.constant(1.0,dtype=tf.float64)
    #tf.exp(-tf.subtract(X,X2) / float(2.2))
    
    
    
    @gpflow.params_as_tensors
    def K(self, X, X2=None):
        
        
        
        
        f_one = gather_cols(X, self.baseline_indices, name=None)
        f_one = tf.reshape(f_one,(1,self.num_data_points, len(self.baseline_indices)))
        g_one =gather_cols(X, self.psi_indices, name=None)
        g_one = tf.reshape(g_one,(1,self.num_data_points ,len(self.psi_indices)))
        user_id_one = gather_cols(X, [self.user_index], name=None)
        day_one = gather_cols(X, [self.user_day_index], name=None)
        
        t_one = self.sigma_u1
        t_two = tf.multiply(tf.math.sqrt(self.sigma_u1),tf.math.sqrt(self.sigma_u2))
        rho_term = tf.subtract(self.sigma_rho,1)
        t_two = tf.multiply(t_two,self.sigma_rho)
        t_three = self.sigma_u2
        t_four = t_two
        row_one = tf.stack([t_one, t_two],axis=0)
        row_two = tf.stack([t_three, t_four],axis=0)
        temp_sigma = tf.reshape(tf.stack([row_one,row_two]),[1,2,2])
        #print(f_one.get_shape())
        
        if not X2 is None:
            print('called')
            f_two =gather_cols(X2,self.baseline_indices, name=None)
            #f_two = tf.reshape(f_two,(1,100,6))
            g_two = gather_cols(X2,  self.psi_indices, name=None)
            #g_two = tf.reshape(g_two,(1,100,2))
            user_id_two = gather_cols(X2, [self.user_index], name=None)
            day_two = gather_cols(X2, [self.user_day_index], name=None)
        
        
        
        else:
            #print('called')
            user_id_two = user_id_one
            day_two = day_one
            f_two = gather_cols(X, self.baseline_indices, name=None)
            g_two=gather_cols(X, self.psi_indices, name=None)
        
        
        
        
        #rho_term = self.rbf_custom(day_one,X2=day_two)
        #print('rho')
        #print(rho_term.get_shape())
        
        #baselines = tf.tensordot(tf.transpose(f_one),self.sigma_theta,axes=[[0],[1]])
        baselines = tf.reshape(tf.tensordot(f_one,self.sigma_theta,axes=[[2],[1]]),(self.num_data_points ,len(self.baseline_indices)))
     
        baselines = tf.tensordot(baselines,tf.transpose(f_two),axes=[[1],[0]])
        
     
        effects = tf.reshape(tf.tensordot(g_one,temp_sigma,axes=[[2],[1]]),(self.num_data_points ,2))
        effects = tf.tensordot(effects,tf.transpose(g_two),axes = [[1],[0]])
        
        effects_one = tf.multiply(effects,self.select_users)
        
        
        effects = tf.reshape(tf.tensordot(g_one,self.sigma_v,axes=[[2],[1]]),(self.num_data_points ,2))
        #effects = tf.tensordot(tf.transpose(g_one),self.sigma_v[0],axes=[[0],[1]])
        
        
        effects = tf.tensordot(effects,tf.transpose(g_two),axes = [[1],[0]])
        effects_two = tf.multiply(effects,self.rhos)
        
        effects = tf.add(effects_one,effects_two)
        
        
        #print('eff')
        #print(effects.shape)
        result = tf.add(baselines,effects)
        
        #noise = 1000*np.eye(100)
        noise= tf.multiply(self.noise_term,tf.constant(np.eye(self.num_data_points )))
        noise = tf.reshape(noise,(1,self.num_data_points ,self.num_data_points ))
        
        result = tf.add(result,noise)
        result = tf.reshape(result,(self.num_data_points ,self.num_data_points ))
        #effect_term = tf.add(user_term,effect_term)
        #print('r')
        #print(result.shape)
        
        
        return result



