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
        return tf.reshape(tf.gather(p_flat, i_flat),[p_shape[0], -1])


def get_theta(dim_baseline):
    m = np.eye(dim_baseline)
    m = np.add(m,.1)
    return m



def rbf_custom_np( X, X2=None):
    #print(X)
    #print(X2)
    if X2 is None:
        X2=X
    return math.exp(-((X-X2)**2)/1.0)



def run(X,y,gp_train_type='Static'):
    #init_g = tf.global_variables_initializer()
    #init_l = tf.local_variables_initializer()
    #with tf.Session() as sess:
    #sess.run(init_g)
    #sess.run(init_l)
    users = np.array([[float(X[i][8]==X[j][8]) for j in range(len(X))] for i in range(len(X))])

    rdayone = [x[9] for x in X]
    rdaytwo = rdayone
    rhos = np.array([[rbf_custom_np( rdayone[i], X2=rdaytwo[j]) for j in range(len(X))] for i in range(len(X))])
    #print(type(rhos))
    from tensorflow.python.framework import ops
    ops.reset_default_graph()
    sess = tf.InteractiveSession()
    
    if gp_train_type=='empirical_bayes':
        k = CustomKernel.CustomKernel(10,mysession=sess,rhos=rhos,select_users=users,baseline_indices=[0,1,2,3,4,5,6],psi_indices=[0,7],user_day_index=9,user_index=8,num_data_points=X.shape[0])
    else:
        k = CustomKernelStatic.CustomKernelStatic(10,mysession=sess,rhos=rhos,select_users=users,baseline_indices=[0,1,2,3,4,5,6],psi_indices=[0,7],user_day_index=9,user_index=8,num_data_points=X.shape[0])

    m = gpflow.models.GPR(X,y, kern=k)
    #m.initialize(session=sess)
    if gp_train_type=='empirical_bayes':
        gpflow.train.ScipyOptimizer().minimize(m,session=sess)


    term = m.kern.K(X,X2=X)
    if gp_train_type=='empirical_bayes':
        trm = term.eval(session=sess)
    else:
        
        trm = term.eval(session=sess)
#if gp_train_type=='empirical_bayes':
    sigma_u = np.array([[m.kern.sigma_u1.value,m.kern.sigma_u1.value**.5*m.kern.sigma_u2.value**.5*m.kern.sigma_rho.value],\
                            [m.kern.sigma_u1.value**.5*m.kern.sigma_u2.value**.5*m.kern.sigma_rho.value,m.kern.sigma_u2.value]])
#np.array([[1.0,0.1],[0.1,1.0]])
    sigma_v =m.kern.sigma_v.value
    noise =m.kern.noise_term.value

    return {'sigma_u':sigma_u,'sigma_v':sigma_v,'cov':trm,'noise':noise}
        #else:
        
        #sigma_u = np.array([[m.kern.sigma_u1.eval(session=sess),m.kern.sigma_u1.eval(session=sess)**5*m.kern.sigma_u2.eval(session=sess)**.5*m.kern.sigma_rho.eval(session=sess)],[m.kern.sigma_u1.eval(session=sess)**.5*m.kern.sigma_u2.eval(session=sess)**.5*m.kern.sigma_rho.eval(session=sess),m.kern.sigma_u2.eval(session=sess)]])
#return {'sigma_u':sigma_u,'sigma_v':m.kern.sigma_v.eval(session=sess),'cov':trm,'noise':m.kern.noise_term.eval(session=sess)}
###get posterior mu and theta for each user .... at the end of the night calculate once for the next day or what?

def get_history(write_dir,dt):
    to_return = {}
    for d in [f for f in os.listdir(write_dir) if f!='.DS_Store']:
        participant = {}
        for f in os.listdir('{}/{}'.format(write_dir,d)):
            if f!='.DS_Store':
                time = int(f.split('_')[1])
                if time <=dt:
                    with open('{}/{}/{}'.format(write_dir,d,f),'rb') as f:
                        ld = pickle.load(f)
                    participant[time]=ld
    
        pid = d.split('_')[1]
        if len(participant)>0:
            to_return[int(pid)]=participant
    return to_return

def create_phi_new(history_dict,pi,global_params):
    #these things will be accessed by the global params
    indices = ['weather','location']
    g0 = ['location']
    f1=['ltps']
    
    ##returns phi and psi indices
    ##this could be a bit faster not appending all the time
    all_data = []
    steps=[]
    for user_id,history in history_dict.items():
        #history = d.history
        #history_keys = sorted(history)
        for hk,h in history.items():
            
            h = history[hk]
            if h['decision_time']:
                v = [1]
                v.extend([h[i] for i in indices])
                v.append(pi*1)
                v.extend([pi*h[i] for i in f1])
                action = h['action']
                if action<0:
                    action=0
                
                v.append((action-pi)*1)
                v.extend([(action-pi)*h[i] for i in f1])
                v.append(action)
                v.append(float(user_id))
                v.append(float(h['study_day']))
                all_data.append(v)
                steps.append(h['steps'])
    return all_data,steps

def make_history_new(write_directory,pi,glob):
    g = get_history(write_directory,glob.decision_times)
    ad = create_phi_new(g,pi,glob)
    if len(ad[0])==0:
        return [[],[]]
    
    new_x = preprocessing.scale(np.array(ad[0]))
    new_y = preprocessing.scale(np.array(ad[1]))
    y = np.array([[float(r)] for r in new_y])
    X = new_x
    return [X,y]

def create_phi(exp,pi):
    indices = ['weather','location']
    g0 = ['location']
    f1=['ltps']
    
    ##returns phi and psi indices
    ##this could be a bit faster not appending all the time
    all_data = []
    steps=[]
    for user_id,d in exp.population.items():
        history = d.history
        history_keys = sorted(history)
        for hk in history_keys:
            
            h = history[hk]
            if h['decision_time']:
                v = [1]
                v.extend([h[i] for i in indices])
                v.append(pi*1)
                v.extend([pi*h[i] for i in f1])
                action = h['action']
                if action<0:
                    action=0
                
                v.append((action-pi)*1)
                v.extend([(action-pi)*h[i] for i in f1])
                v.append(action)
                v.append(float(user_id))
                v.append(float(h['study_day']))
                all_data.append(v)
                steps.append(h['steps'])
    return all_data,steps


def make_history(exp):
    ad = create_phi(exp,.6)
    if len(ad[0])==0:
        return [[],[]]
    
    new_x = preprocessing.scale(np.array(ad[0]))
    new_y = preprocessing.scale(np.array(ad[1]))
    y = np.array([[float(r)] for r in new_y])
    X = new_x
    return [X,y]

##make function of pZ, not too hard
def create_H(context_len):
    return np.transpose(np.array([[1]+[0 for i in range(context_len)]+[0,0,0,0],[0]+[0 for i in range(context_len)]+[1,0,1,0]]))


    

        

def get_M(global_params,user_id,user_study_day,history):
  
  
    day_id =user_study_day
    
    M = [[] for i in range(history.shape[0])]

    H = create_H(1)
    for x_old_i in range(history.shape[0]):
        x_old = history[x_old_i]
        
        ##these indices all need to be parameters
        phi = x_old[1:7]
        
        old_user_id = x_old[8]
        old_day_id = x_old[9]
        
        inner = float(old_user_id==user_id)*global_params.sigma_u.reshape(2,2)+\
rbf_custom_np(day_id,old_day_id)*global_params.sigma_v.reshape(2,2)
        #print(inner.shape)
        #print(H.shape)
        inner = np.dot(H,inner)
        #print(inner.shape)
        inner = np.dot(inner,np.transpose(H))
        #print(inner.shape)
        inner = np.add(global_params.sigma_theta,inner)
        
        
        
        
        term = np.dot(np.transpose(phi),inner)
        M[x_old_i]=[i for i in term]
            
            #print(len(M))
            #print(np.array(M).shape)
    return np.array(M)

def get_RT(y,X,sigma_theta):
    
    to_return = [y[i]-np.dot(X[i][0:6],np.ones(6)) for i in range(len(X))]
    return np.array([i[0] for i in to_return])

def calculate_posterior(global_params,user_id,user_study_day,X,y):
    H = create_H(1)
    M = get_M(global_params,user_id,user_study_day,X)
    ##change this to be mu_theta
    ##is it updated?  the current mu_theta?
    adjusted_rewards =get_RT(y,X,global_params.mu_theta)
    
    mu = get_middle_term(X.shape[0],global_params.cov.reshape(X.shape[0],X.shape[0]),\
                global_params.noise_term,M,adjusted_rewards,global_params.mu_theta)
    sigma = get_post_sigma(H,global_params.cov.reshape(X.shape[0],X.shape[0]),global_params.sigma_u.reshape(2,2),\
                           global_params.sigma_v.reshape(2,2),global_params.noise_term,M,X.shape[0],global_params.sigma_theta)

    return mu[-3:],[j[-3:] for j in sigma[-3:]]


def get_middle_term(X_dim,cov,noise_term,M,adjusted_rewards,mu_theta):
    #M = get_M(global_params,user_id,user_study_day,history[0])
    
    ##change this to be mu_theta
    ##is it updated?  the current mu_theta?
    #adjusted_rewards =[history[1][i]-np.dot(history[0][i][0:6],np.ones(6)) for i in range(len(history[0]))]
    
    noise = noise_term * np.eye(X_dim)
    #print(noise.shape)
    #print(cov.shape)
    middle_term = np.add(cov,noise)
    
    middle_term = np.dot(M.T,np.linalg.inv(middle_term))

    middle_term = np.dot(middle_term,adjusted_rewards)
    return np.add(mu_theta,middle_term)


def get_post_sigma(H,cov,sigma_u,sigma_v,noise_term,M,x_dim,sigma_theta):
    #M = get_M(global_params,user_id,user_study_day,history[0])
    
    ##change this to be mu_theta
    ##is it updated?  the current mu_theta?
    #adjusted_rewards =[history[1][i]-np.dot(history[0][i][0:6],np.ones(6)) for i in range(len(history[0]))]
    
    
    
    first_term = np.add(sigma_u,sigma_v)
    first_term = np.dot(H,first_term)
    first_term = np.dot(first_term,H.T)
    
    
    noise = noise_term * np.eye(x_dim)
    middle_term = np.add(cov,noise)
    
    middle_term = np.dot(M.T,np.linalg.inv(middle_term))
    
    middle_term = np.dot(middle_term,M)
    
    last = np.add(sigma_theta,first_term)
    last = np.subtract(last,middle_term)
    
    return middle_term
