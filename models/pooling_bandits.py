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
    #m = np.add(m,.1)
    return m



def rbf_custom_np( X, X2=None):
    #print(X)
    #print(X2)
    if X2 is None:
        X2=X
    return math.exp(-((X-X2)**2)/100.0)

def get_sigma_u(u1,u2,rho):
    off_diagaonal_term = u1**.5*u2**.5*(rho-1)
    return np.array([[u1,off_diagaonal_term],[off_diagaonal_term,u2]])

def run(X,y,global_params,gp_train_type='Static'):
    #init_g = tf.global_variables_initializer()
    #init_l = tf.local_variables_initializer()
    #with tf.Session() as sess:
    #sess.run(init_g)
    #sess.run(init_l)
    #print(X)
    #print(y)
    users = np.array([[float(X[i][global_params.user_id_index]==X[j][global_params.user_id_index]) for j in range(len(X))] for i in range(len(X))])

    rdayone = [x[global_params.user_day_index] for x in X]
    rdaytwo = rdayone
    rhos = np.array([[rbf_custom_np( rdayone[i], X2=rdaytwo[j]) for j in range(len(X))] for i in range(len(X))])
    #print(type(rhos))
    #from tensorflow.python.framework import ops
    #ops.reset_default_graph()
    #sess = tf.InteractiveSession()
    
    #sess = tf.Session()
    
    with tf.Graph().as_default():
        sess = tf.Session()
        print(global_params.kdim)
        
        if gp_train_type=='empirical_bayes':
            k = CustomKernel.CustomKernel(global_params.kdim,mysession=sess,rhos=rhos,select_users=users,baseline_indices=global_params.baseline_indices,psi_indices=global_params.psi_indices,user_day_index=global_params.user_day_index,user_index=global_params.user_id_index,num_data_points=X.shape[0],initial_u1=global_params.sigma_u[0][0],initial_u2=global_params.sigma_u[1][1],initial_s1=global_params.sigma_v[0][0],initial_s2=global_params.sigma_v[1][1],initial_rho=global_params.rho_term,initial_noise=global_params.noise_term)
        else:
            k = CustomKernelStatic.CustomKernelStatic(global_params.kdim,mysession=sess,rhos=rhos,select_users=users,baseline_indices=global_params.baseline_indices,psi_indices=global_params.psi_indices,user_day_index=global_params.user_day_index,user_index=global_params.user_id_index,num_data_points=X.shape[0])

        m = gpflow.models.GPR(X,y, kern=k)
        m.initialize(session=sess)
        m.likelihood.variance=0
        m.likelihood.variance.trainable =False
#if gp_train_type=='Static':
    
#m.initialize(session=sess)
        if gp_train_type=='empirical_bayes':
#           m.initialize(session=sess)
            try:
            
                gpflow.train.ScipyOptimizer().minimize(m,session=sess)
        #print(m.as_pandas_table())
        #print('did work')
            except Exception as e:
# shorten the giant stack trace

                lines = str(e).split('\n')
                print ('\n'.join(lines[:15]+['...']+lines[-30:]))
    


        term = m.kern.K(X,X2=X)
        if gp_train_type=='empirical_bayes':
            trm = term.eval(session=sess)
        else:
        
            trm = term.eval(session=sess)
        sess.close()
#if gp_train_type=='empirical_bayes':
    sigma_u = get_sigma_u(m.kern.sigma_u1.value,m.kern.sigma_u2.value,m.kern.sigma_rho.value)

#np.array([[1.0,0.1],[0.1,1.0]])
    sigma_v =m.kern.sigma_v.value
    noise =m.kern.noise_term.value
    #print('lll')
    #print(sigma_v.shape)
    
    #print(noise.shape)
    
    sess.close()
    #print(sess._closed)
    return {'sigma_u':sigma_u,'sigma_v':sigma_v.reshape(2,2),'cov':trm,'noise':noise}
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

def get_history_norw(exp,glob):
    to_return = {}

    for userid,data in exp.population.items():
        to_return[userid]= {k:v for k,v in data.history.items() if k<glob.last_global_update_time and v['avail'] and v['decision_time']}
        

    return to_return



def create_phi_new(history_dict,pi,baseline_features,responsivity_features):
    all_data = []
    steps=[]
    
    
    ##might add pi to the user's history
    for user_id,history in history_dict.items():
     
            for hk,h in history.items():
                pi = h['prob']
                
                
                v = [1]
                v.extend([h[i] for i in baseline_features])
                v.append(pi*1)
                v.extend([pi*h[i] for i in responsivity_features])
                action = h['action']
                if action<0:
                    action=0
                
                v.append((action-pi)*1)
                v.extend([(action-pi)*h[i] for i in responsivity_features])
                #v.append(action)
                v.append(float(user_id))
                v.append(float(h['study_day']))
                all_data.append(v)
                steps.append([h['steps']])
        
        
    return np.array(all_data),np.array(steps)



def make_history_new(pi,glob,exp=None):
    g=get_history_norw(exp,glob)
    #g = get_history(glob.write_directory,glob.decision_times)
    ad = create_phi_new(g,pi,glob.baseline_features,glob.responsivity_features)
    if len(ad[0])==0:
        return [[],[]]
    
    #z = new_standardize(ad[0],ad[1])
    #new_x = preprocessing.scale(np.array(ad[0]))
    #new_y = preprocessing.scale(np.array(ad[1]))
#y = np.array([[float(r)] for r in z[1]])
    return [ad[0],ad[1]]

def create_phi_one_hot(glob,history_dict):
    all_data = []
    all_steps = []
    
    
    for user_id,history in history_dict.items():
        ##change to find pi in h
        for hk,h in history.items():
            one_hot = get_one_hot_encodings(glob,h)
            pi = h['prob']
            if pi==-1:
                print('yikes')
            #print(pi)
            v = [1]
            v.extend(list(one_hot))
            v.append(pi)
            v.extend(list(pi*one_hot))
            v.append(h['action']-pi)
            v.extend(list((h['action']-pi)*one_hot))
            v.append(float(user_id))
            v.append(h['study_day'])
            v = [float(i) for i in v]
            v = np.array(v)
            all_data.append(v)
            all_steps.append([float(h['steps'])])
    return [np.array(all_data),all_steps]

def make_history_one_hot(pi,glob,exp=None):
    g=get_history_norw(exp,glob)
    #g = get_history(glob.write_directory,glob.decision_times)
    ad = create_phi_one_hot(glob,g)
    ad = new_standardize(ad[0],ad[1])
    return [ad[0],ad[1]]




##make function of pZ, not too hard
def create_H(num_baseline_features,num_responsivity_features):
    ##for now have fixed random effects size one
    
    random_effect_one = [1]
    random_effect_two = [1]
    
    column_one = [1]
    column_one = column_one+[0]*num_baseline_features
    column_one = column_one+[0]
    column_one = column_one+[0]*num_responsivity_features
    column_one = column_one+[0]
    column_one = column_one+[0]*num_responsivity_features
    
    
    column_two = [0]
    column_two = column_two+[0]*num_baseline_features
    column_two = column_two+[1]
    column_two = column_two+[0]*num_responsivity_features
    column_two = column_two+[1]
    column_two = column_two+[0]*num_responsivity_features
    
    return np.transpose(np.array([column_one,column_two]))


    
def new_standardize(X,y):
    new_x = [x[:-2] for x in X]
    new_x = np.array(new_x)
    #ds = np.diag(np.random.rand(new_x.shape[0]))
    #ds = np.random.rand(new_x.shape[0],new_x.shape[1])
    #print(ds)
    #print(new_x)
    #new_x = np.add(ds,new_x)
    mm = preprocessing.MinMaxScaler(feature_range=(.01, 1))
    new_x =mm.fit_transform(np.array(new_x))
    to_return = np.zeros((len(X),len(X[0])))
    for i in range(len(X)):
        #temp=np.zeros(len(X[i]))
        to_return[i][:-2]=new_x[i]
        
        to_return[i][-2]=X[i][-2]
        to_return[i][-1]=X[i][-1]
            #mm = preprocessing.MinMaxScaler(feature_range=(.5, 1))
#reprocessing.scale(np.array([[float(yi)] for yi in y]))
    return [to_return,y]
        
def get_one_hot_encodings(glob,context_dict):
    #tod =sf.get_time_of_day(context_dict['time'])
    
    #dow =sf.get_day_of_week(context_dict['time'])

    #pre = sf.get_pretreatment(context_dict['ltps'])
    
    
    tod = context_dict['tod']
    
    dow = context_dict['dow']
    
    
    pre=context_dict['pretreatment']
    
    weather = context_dict['weather']
    
    location = context_dict['location']
    
    key = 'tod-{}-dow-{}-wea-{}-pre-{}-loc-{}'.format(tod,dow,weather,pre,location)

    skip ='tod-1-dow-0-wea-1-pre-0-loc-0'
    vector = np.zeros(63)
    if key!=skip:
        index = glob.one_hot_indices[key]
        vector[index]=1
    return vector



def get_M(global_params,user_id,user_study_day,history):
  
  
    day_id =user_study_day
    #print(history)
    M = [[] for i in range(history.shape[0])]

    H = create_H(global_params.num_baseline_features,global_params.num_responsivity_features)
    for x_old_i in range(history.shape[0]):
        x_old = history[x_old_i]
        old_user_id = x_old[global_params.user_id_index]
        old_day_id = x_old[global_params.user_day_index]
        
        ##these indices all need to be parameters
        phi = np.array([x_old[i] for i in global_params.baseline_indices])
        
        t_one = np.dot(np.transpose(phi),global_params.sigma_theta)
        #first_terms.append(t_one)
        
        temp = np.dot(H,global_params.sigma_u)
        temp = np.dot(temp,H.T)
        temp = np.dot(np.transpose(phi),temp)
        temp = float(old_user_id==user_id)*temp
        t_two = temp
        #middle_terms.append(t_two)
        temp = np.dot(H,global_params.sigma_v.reshape(2,2))
        temp = np.dot(temp,H.T)
        temp = np.dot(np.transpose(phi),temp)
        temp = rbf_custom_np(user_study_day,old_day_id)*temp
        t_three = temp
        #print(user_study_day)
        
        #last_terms.append(t_three)
        term = np.add(t_one,t_two)
        
        term = np.add(term,t_three)
        #print(term.shape)
        #print(term)
        M[x_old_i]=term

    return np.array(M)

def get_RT(y,X,sigma_theta,x_dim):
    
    to_return = [y[i]-np.dot(X[i][0:x_dim],np.ones(x_dim)) for i in range(len(X))]
    return np.array([i[0] for i in to_return])


def get_g_one(context_dict):
    pass

def get_f_one(context_dict):
    pass


def get_M_faster(global_params,user_id,user_study_day,history):
    
    
    day_id =user_study_day
    #print(history)
    M = [[] for i in range(history.shape[0])]
    
    H = create_H(global_params.num_baseline_features,global_params.num_responsivity_features)
    
    phi = history[:,global_params.baseline_indices]
    ##should be fine
    t_one = np.dot(phi,global_params.sigma_theta)
    temp = np.dot(H,global_params.sigma_u)
    temp = np.dot(temp,H.T)
    temp = np.dot(phi,temp)
    
    
    
    #print(history)
  
    user_ids = history[:,global_params.user_id_index]
    days_ids = history[:,global_params.user_day_index]
  
    my_days = np.ma.masked_where(user_ids==user_id, user_ids).mask.astype(float)
    if type(my_days)!=np.ndarray:
        my_days = np.zeros(history.shape[0])
    user_matrix = np.diag(my_days)
    
    rho_diag = np.diag([rbf_custom_np(d,user_study_day) for d in days_ids])
    
    t_two = np.matmul(user_matrix,temp)
    
    temp = np.dot(H,global_params.sigma_v.reshape(2,2))
    temp = np.dot(temp,H.T)
    temp = np.dot(phi,temp)
    #temp = rbf_custom_np(user_study_day,old_day_id)*temp
    t_three = np.matmul(rho_diag,temp)
    term = np.add(t_one,t_two)
    
    term = np.add(term,t_three)
    
    
    
    return term

#rdayone = [x[global_params.user_day_index] for x in X]
#rdaytwo = rdayone
#rhos = np.array([[rbf_custom_np( rdayone[i], X2=rdaytwo[j]) for j in range(len(X))] for i in range(len(X))])





def calculate_posterior_faster(global_params,user_id,user_study_day,X,y):
    H = create_H(global_params.num_baseline_features,global_params.num_responsivity_features)
    M = get_M_faster(global_params,user_id,user_study_day,X)
    ##change this to be mu_theta
    ##is it updated?  the current mu_theta?
    adjusted_rewards =get_RT(y,X,global_params.mu_theta,global_params.theta_dim)
    #print('current global cov')
    #print(global_params.cov)
    #.reshape(X.shape[0],X.shape[0])
    mu = get_middle_term(X.shape[0],global_params.cov,global_params.noise_term,M,adjusted_rewards,global_params.mu_theta)
    #.reshape(X.shape[0],X.shape[0])
    sigma = get_post_sigma(H,global_params.cov,global_params.sigma_u.reshape(2,2),global_params.sigma_v.reshape(2,2),global_params.noise_term,M,X.shape[0],global_params.sigma_theta)
    
    return mu[-(global_params.num_responsivity_features+1):],[j[-(global_params.num_responsivity_features+1):] for j in sigma[-(global_params.num_responsivity_features+1):]]


def calculate_posterior(global_params,user_id,user_study_day,X,y):
    H = create_H(global_params.num_baseline_features,global_params.num_responsivity_features)
    M = get_M(global_params,user_id,user_study_day,X)
    ##change this to be mu_theta
    ##is it updated?  the current mu_theta?
    adjusted_rewards =get_RT(y,X,global_params.mu_theta,global_params.theta_dim)
    #print('current global cov')
    #print(global_params.cov)
    #.reshape(X.shape[0],X.shape[0])
    mu = get_middle_term(X.shape[0],global_params.cov,global_params.noise_term,M,adjusted_rewards,global_params.mu_theta)
    #.reshape(X.shape[0],X.shape[0])
    sigma = get_post_sigma(H,global_params.cov,global_params.sigma_u.reshape(2,2),global_params.sigma_v.reshape(2,2),global_params.noise_term,M,X.shape[0],global_params.sigma_theta)

    return mu[-(global_params.num_responsivity_features+1):],[j[-(global_params.num_responsivity_features+1):] for j in sigma[-(global_params.num_responsivity_features+1):]]


def get_middle_term(X_dim,cov,noise_term,M,adjusted_rewards,mu_theta):
    #M = get_M(global_params,user_id,user_study_day,history[0])
    
    ##change this to be mu_theta
    ##is it updated?  the current mu_theta?
    #adjusted_rewards =[history[1][i]-np.dot(history[0][i][0:6],np.ones(6)) for i in range(len(history[0]))]
    
    ##noise = noise_term * np.eye(X_dim)
    #print(noise.shape)
    #print(cov.shape)
    ##middle_term = np.add(cov,noise)
    
    ##middle_term = np.dot(M.T,np.linalg.inv(middle_term))

    ##middle_term = np.dot(middle_term,adjusted_rewards)
    ##return np.add(mu_theta,middle_term)
    noise = noise_term * np.eye(X_dim)
    #print(noise.shape)
    #print(cov.shape)
    #print('in get middle')
    middle_term = np.add(cov,noise)
    #print(middle_term)
    #print(M.shape)
    middle_term = np.matmul(M.T,np.linalg.inv(middle_term))
    #print(middle_term)
    middle_term = np.matmul(middle_term,adjusted_rewards)
    #print(middle_term)
    return np.add(mu_theta,middle_term)

def get_post_sigma(H,cov,sigma_u,sigma_v,noise_term,M,x_dim,sigma_theta):
    #M = get_M(global_params,user_id,user_study_day,history[0])
    
    ##change this to be mu_theta
    ##is it updated?  the current mu_theta?
    #adjusted_rewards =[history[1][i]-np.dot(history[0][i][0:6],np.ones(6)) for i in range(len(history[0]))]
    
    
    
    first_term = np.add(sigma_u,sigma_v)
    #print(first_term.shape)
    #print(H.shape)
    first_term = np.dot(H,first_term)
    #print(first_term.shape)
    first_term = np.dot(first_term,H.T)
    #print(first_term)
    
    noise = noise_term * np.eye(x_dim)
    #print(noise.shape)
    middle_term = np.add(cov,noise)
    #print(middle_term.shape)
    middle_term = np.dot(M.T,np.linalg.inv(middle_term))
    #print(middle_term.shape)
    middle_term = np.dot(middle_term,M)
    #print(middle_term.shape)
    last = np.add(sigma_theta,first_term)
    last = np.subtract(last,middle_term)
    
    return middle_term
