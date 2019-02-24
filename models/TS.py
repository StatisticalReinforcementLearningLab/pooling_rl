import bandit
from numpy.random import uniform
from numpy.linalg import solve
import numpy as np
import state_params
import random
from scipy.stats import norm
from scipy.linalg import block_diag
import standard_bandits as do_work

def gen_nextdosage(x,a):
    anti_sed = int(uniform() < 0.3)
    event = (anti_sed+a)>0
    x_next = 0.9*x + event
    
    return x_next

def get_probs(batch,init):
    return [b[init.prob_index] for b in batch if b[init.avail_index]==1]

def prob_cal_ts(z,x,mu,Sigma,global_params):
    pos_mean = np.dot(z,mu)
    pos_var = np.dot(np.dot(np.transpose(z),Sigma),z)
    pos_var = max(0,pos_var)

  
 
  
    # probability
    pit_zero = norm.cdf((pos_mean)/(pos_var**.5))
  
    # clipping
    prob =  min(bandit.py_c_func(global_params.pi_max, max(bandit.py_c_func(global_params.pi_min, pit_zero))))
  
    return prob

def make_batch(t,Z,X,I,A,prob,R):
    temp = [t]
    temp.extend(Z)
    temp.extend([X,I,A,prob,R])
    return temp

def policy_update_ts(global_params,batch,  mu_1,Sigma_1,mu_2,Sigma_2):
    return txt_effect_update(batch, global_params, mu_1,Sigma_1,mu_2,Sigma_2)


def policy_update_ts_new(context,steps,probs,actions,o_sigma,  mu_1,Sigma_1,mu_2,Sigma_2):
    return txt_effect_update_new(context,steps,probs,actions,o_sigma, mu_1,Sigma_1,mu_2,Sigma_2)


def txt_effect_update_new(context,steps,probs,actions,o_sigma, mu_1,Sigma_1,mu_2,Sigma_2):
    #print(init.avail_index)
    #print(len(batch))
 
    #avail <- batch[, input$avail.index]
    

    
    if len(context)==0:
        
        return [mu_2,Sigma_2]
    else:
        #what is this line doing?
        #check get xz matrix, why does it need global params?
        #I think just for indexing?
        #xz = get_xz_matrix(batch,global_params)
        #print(batch)
        #print(len(xz))
        
        #action <- batch[index, input$action.index]
        
        mu_tmp =[m for m in mu_1]
        mu_tmp.extend([m for m in mu_2])
        mu_tmp.extend([m for m in mu_2])
        Sigma_tmp = block_diag(Sigma_1,Sigma_2,Sigma_2)
        
        #actions = get_actions(batch,global_params)
        #probs = get_probs(batch,global_params)
        
        f_one = transform_f1_new(context)
        f_two = transform_f2_new(context)
        
        X_trn = get_X_trn_new(f_one,actions,f_two,probs)
        Y_trn = steps
    #get_Y_trn(batch,global_params)
        
        
        
        
        temp = post_cal_ts(X_trn, Y_trn, o_sigma, mu_tmp, Sigma_tmp)
        
        #print(len(temp))
        #print(len(f_two))
        #print(len(temp))
        #print(len(f_two))
        nm,nS = clip_mean_sigma(temp[0],temp[1],len(f_two[0]))
        
        return [nm,nS]

def get_X_trn_new(F1,actions,F2,probs):
    #cbind(F1, prob * F2, (action-prob) * F2)
    to_return = []
    for i in range(len(F2)):
        #print(F2[i])
        term_two = probs[i]*F2[i]
        term_three = (actions[i]-probs[i])*F2[i]
        row = np.concatenate((np.array(F1[i]),term_two,term_three))
        to_return.append(row)
    return to_return

#what does this stand for? change the name
def txt_effect_update(batch, global_params, mu_1,Sigma_1,mu_2,Sigma_2):
    #print(init.avail_index)
    #print(len(batch))
    avail = [b[global_params.avail_index] for b in batch]
    #avail <- batch[, input$avail.index]
    
    ##how can this ever equal 1?
    index = [int(a==1) for a in avail]
    
    if sum(index)==0:
        
        return [mu_2,Sigma_2]
    else:
        #what is this line doing?
        #check get xz matrix, why does it need global params? 
        #I think just for indexing?
        xz = get_xz_matrix(batch,global_params)
        #print(batch)
        #print(len(xz))
        
        #action <- batch[index, input$action.index]
        
        mu_tmp =[m for m in mu_1]
        mu_tmp.extend([m for m in mu_2])
        mu_tmp.extend([m for m in mu_2])
        Sigma_tmp = block_diag(Sigma_1,Sigma_2,Sigma_2)
        
        actions = get_actions(batch,global_params)
        probs = get_probs(batch,global_params)
        
        f_one = transform_f1(xz)
        f_two = transform_f2(xz)
        
        X_trn = get_X_trn(f_one,actions,f_two)
        Y_trn = get_Y_trn(batch,global_params)
        
        
        
        
        temp = post_cal_ts(X_trn, Y_trn, global_params.sigma, mu_tmp, Sigma_tmp)
        
        #print(len(temp))
        #print(len(f_two))
        #print(len(temp))
        #print(len(f_two))
        nm,nS = clip_mean_sigma(temp[0],temp[1],len(f_two[0])) 
      
        return [nm,nS]
        # return the post dist of txt eff
        #txt.index <- tail(1:ncol(X.trn), ncol(F2)) # interaction terms
        #list(mean = temp$mean[txt.index], var = temp$var[txt.index, txt.index])

def post_cal_ts(X, Y, sigma, mu, Sigma):
    
    inv_Sigma = solve(Sigma,np.eye(len(Sigma[0])))
    #print(inv_Sigma)
    #print(sigma)
    #print(X)
    #print(inv_Sigma)
    #print(np.dot(np.transpose(X),X))
    #print(type(sigma))
    #print((sigma**2)*inv_Sigma)
    term_one = np.dot(np.transpose(X),X)+sigma**2*inv_Sigma
    term_two = np.dot(np.transpose(X),Y)+np.dot(sigma**2*inv_Sigma,mu)
    pos_mean = solve(term_one,term_two)
    pos_var = solve_sigma(sigma,X,inv_Sigma)
    return [pos_mean,pos_var]

def solve_sigma(sigma,X,inv_sigma):
    term_one = np.dot(np.transpose(X),X)+sigma**2*inv_sigma
    temp = np.multiply(sigma**2,solve(term_one,np.eye(len(term_one))))
    return temp

def clip_mean_sigma(mean,sigma,tlen):
    new_mean = mean[-tlen:]
    #print(sigma)
    start = len(sigma[0])-tlen
    new_sigma = sigma[start::,start:]
    return new_mean,new_sigma

def get_X_trn(F1,actions,F2):
    to_return = []
    for i in range(len(F2)):
        a = np.multiply(actions[i],F2[i])
        row = np.concatenate((np.array(F1[i]),a))
        to_return.append(row)
    return to_return

def get_Y_trn(batch,init):
      return [b[init.reward_index] for b in batch if b[init.avail_index]==1]  
    
def get_actions(batch,init):
    return [b[init.action_index] for b in batch if b[init.avail_index]==1]

def transform_f2(xz):
    return [[1,row[-1],row[0]] for row in xz]

def transform_f1(xz):
    return [[1,row[2],row[0],row[1]] for row in xz]

def transform_f2_new(xz):
    return [np.array([1]+[i for i in row]) for row in xz]

def transform_f1_new(xz):
    return [np.array([1]+[i for i in row]) for row in xz]


def get_xz_matrix(batch,init):
    
   
        new_matrix = [[b[init.x_index]]+b[init.z_index[0]:init.z_index[-1]+1] for b in batch if b[init.avail_index]==1]
        return new_matrix
    
   
