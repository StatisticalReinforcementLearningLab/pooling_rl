
##get data
import numpy as np
import sys
sys.path
sys.path.append('../models')
import pickle
import math
import tensorflow as tf
import gpflow
import CustomKernelStatic
import random.random()

def rbf_custom_np( X, X2=None):
    #print(X)
    #print(X2)
    if X2 is None:
        X2=X
        return math.exp(-((X-X2)**2)/1.0)
#tf.divide(tf.square(-tf.subtract(X,X2)),tf.constant(1.0,dtype=tf.float64)))
def get_likelihood(u1,u2,s1,s2,rho,noise,X,y,rhos,users):
    sess = tf.Session()
    k =  CustomKernelStatic.CustomKernelStatic(20,mysession=sess,rhos=rhos,select_users=users,baseline_indices=[i for i in range(18)],psi_indices=[0,6],user_day_index=19,user_index=18,num_data_points=len(X),initial_u1=u1,initial_u2=u2,initial_s1=s1,initial_s2=s2,initial_rho=rho,initial_noise=noise)
        
    m = gpflow.models.GPR(X,y, kern=k)
    m.initialize(session=sess)
                                               
    try:
        l =m.compute_log_likelihood()
        print('huzzah')
        print(l)
    except:
        l=-100
                                                       
    sess.close()
    return l
##get likelihood #return likelihood


def run():
    to_save = {}
    u1 = np.arange(.1,1.1,.1)
    u2 = np.arange(.1,1.1,.1)
    first_rhos = np.arange(.1,1.1,.2)
    v1 = np.arange(1,10,1)
    v2 = np.arange(1,10,1)
    noise = np.arange(.1,1.1,.1)

    #with open('../../regal/murphy_lab/pooling/processed_data_for_gridsearch_GP.pkl','rb') as f:
        #data = pickle.load(f)
    with open('../../Downloads/processed_data_for_gridsearch_GP.pkl','rb') as f:
        data = pickle.load(f)
    
    X=data[0]
    y=data[1]
    
    rdayone = [x[19] for x in X]
    rdaytwo = rdayone
    rhos = np.array([[rbf_custom_np( rdayone[i], X2=rdaytwo[j]) for j in range(len(X))] for i in range(len(X))])
    users = np.array([[float(X[i][18]==X[j][18]) for j in range(len(X))] for i in range(len(X))])
    
    print(X.shape)
    print(y.shape)
    for x in X:
        x[0]=1+random.random()/.0001
    for x in X:
        x[6]=1+random.random()/.0001
#for x in X:
#x[7]=1+random.random()/.0001
    for uone in u1:
        for utwo in u2:
            for rho in first_rhos:
                for s1 in v1:
                    for s2 in v2:
                        for n in noise:
                            likli=get_likelihood(uone,uone,s1,s2,rho,n,X,y,rhos,users)
                            
                            key = '-'.join([str(i) for i in[uone,utwo,rho,s1,s2,n]])
                            to_save[key]=likli

    return to_save


if __name__=="__main__":
    
    
    
    to_save = run()
    
    with open('grid_search.pkl','wb) as f:
              pickle.dump(to_save,f)
    
    print('finished')
