import bandit
import bandit
from numpy.random import uniform
from numpy.linalg import solve
import numpy as np
import state_params
import random
from scipy.stats import norm
from scipy.linalg import block_diag
from psi import psi


def gen_nextdosage(x,a):
    anti_sed = int(uniform() < 0.3)
    event = (anti_sed+a)>0
    x_next = 0.9*x + event
    
    return x_next

def get_probs(batch,init):
    return [b[init.prob_index] for b in batch if b[init.avail_index]==1]

def prob_cal(z,x,mu,Sigma,init,eta):
    pos_mean = np.dot(init.feat2_function(z,x),mu)
    pos_var = np.dot(np.dot(np.transpose(init.feat2_function(z,x)),Sigma),init.feat2_function(z,x))
    pos_var = max(0,pos_var)

  
    margin  = eta.eta_f(x)*init.xi
    #print('margin')
    #print(margin)
    # probability
    pit_zero = norm.cdf((pos_mean-margin)/(pos_var**.5))
  
    # clipping
    prob =  min(bandit.py_c_func(init.pi_max, max(bandit.py_c_func(init.pi_min, pit_zero))))
  
    return prob

def post_cal(X, Y, sigma, mu, Sigma):
    
    inv_Sigma = solve(Sigma,np.eye(len(Sigma[0])))
    #print(inv_Sigma)
    term_one = np.dot(np.transpose(X),X)+sigma**2*inv_Sigma
    term_two = np.dot(np.transpose(X),Y)+np.dot(sigma**2*inv_Sigma,mu)
    pos_mean = solve(term_one,term_two)
    pos_var = solve_sigma(sigma,X,inv_Sigma)
    return [pos_mean,pos_var]

def make_batch(t,Z,X,I,A,prob,R):
    temp = [t]
    temp.extend(Z)
    temp.extend([X,I,A,prob,R])
    return temp


     
      
      
    
        


#what does this stand for? change the name
def txt_effect_update(batch,init,mu_1,sigma_1,mu_2,sigma_2):
    #print(init.avail_index)
    #print(len(batch))
    avail = [b[init.avail_index] for b in batch]
    #avail <- batch[, input$avail.index]
    
    ##how can this ever equal 1?
    index = [int(a==1) for a in avail]
    
    if sum(index)==0:
        
        return [mu_2,sigma_2]
    else:
        #what is this line doing?
        xz = get_xz_matrix(batch,init)
        #action <- batch[index, input$action.index]
        #print(type(mu_1))
        mu_tmp =[m for m in mu_1]
        mu_tmp.extend([m for m in mu_2])
        mu_tmp.extend([m for m in mu_2])
        #[mu_1[0]]+ [mu_2[0]]+[mu_2[0]]
        #print(mu_tmp)
        Sigma_tmp = block_diag(sigma_1,sigma_2,sigma_2)
        
        actions = get_actions(batch,init)
        probs = get_probs(batch,init)
        
        f_one = transform_f1(xz,init)
        f_two = transform_f2(xz)
        
        X_trn = get_X_trn(f_one,actions,f_two,probs)
        Y_trn = get_Y_trn(batch,init)
        
       # print((Sigma_tmp.shape))
    #    print(len(mu_tmp))
    #    print(len(X_trn[0]))
    #    print(len(X_trn))
    #    print(len(Y_trn))
        
        
        temp = post_cal(X_trn, Y_trn, init.sigma, mu_tmp, Sigma_tmp)
        nm,nS = clip_mean_sigma(temp[0],temp[1],len(f_two[0])) 
      
        return [nm,nS]
        # return the post dist of txt eff
        #txt.index <- tail(1:ncol(X.trn), ncol(F2)) # interaction terms
        #list(mean = temp$mean[txt.index], var = temp$var[txt.index, txt.index])



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

def get_X_trn(F1,actions,F2,probs):
    #cbind(F1, prob * F2, (action-prob) * F2)
    to_return = []
    for i in range(len(F2)):
        term_two = np.multiply(probs[i],F2[i])
        term_three = np.multiply(actions[i]-probs[i],F2[i])
        row = np.concatenate((np.array(F1[i]),term_two,term_three))
        to_return.append(row)
    return to_return

def get_Y_trn(batch,init):
      return [b[init.reward_index] for b in batch if b[init.avail_index]==1]  
    
def get_Y_trn_gen(batch,init,key):
    return [b[init.reward_index] for b in batch if b[init.avail_index]==key]  
    
def get_actions(batch,init):
    return [b[init.action_index] for b in batch if b[init.avail_index]==1]

def transform_f2(xz):
    return [[1,row[1],row[0]] for row in xz]

def transform_f1(xz,init):
    return [[1]+row[1:len(init.z_index)+1] + [row[0]] for row in xz]
   

def get_xz_matrix(batch,init):
    new_matrix = [[b[init.x_index]]+b[init.z_index[0]:init.z_index[-1]+1] for b in batch if b[init.avail_index]==1]
    return new_matrix

def get_xz_matrix_gen_key(batch,init,key):
    new_matrix = [[b[init.x_index]]+b[init.z_index[0]:init.z_index[-1]+1] for b in batch if b[init.avail_index]==key]
    return new_matrix

def get_xz_matrix_main_effect(batch,init,indices):
    new_matrix = [[batch[i][init.x_index]]+batch[i][init.z_index[0]:init.z_index[-1]+1] for i in indices]
    return new_matrix

def get_unavailable(batch,init):
    return [b for b in batch if b[init.avail_index]==0]


def get_X_trn_unavail_update(xz,init):
    return [[1]+row[1:len(init.z_index)+1] + [row[0]] for row in xz]

def unavail_update(batch,init,mu_0,sigma_0):

    
    # available, no txt, subset
    avail = get_unavailable(batch,init)
 
    
    if(len(avail) == 0):
      
      # return the prior
        return [mu_0,sigma_0]
      
      
    else:
      
        xz = get_xz_matrix_gen_key(batch,init,0)
      
      # forming the prior 
        mu_tmp = mu_0
        sigma_tmp = sigma_0
      
      
        # calculate posterior (batch update)
       
        X_trn = get_X_trn_unavail_update(xz,init)
        Y_trn = get_Y_trn_gen(batch,init,0)
        temp = post_cal(X_trn, Y_trn, init.sigma, mu_tmp, sigma_tmp)
      
      # return the post dist of mian eff
    return temp
 
def get_valid_main(batch,init):
    return [1 for b in batch if b[init.avail_index]==1 and b[init.action_index]==0]
    
def get_valid_main_indices(batch,init):
    return [i for i in range(len(batch)) if batch[i][init.avail_index]==1 and batch[i][init.action_index]==0]
     
def get_Z_trn(batch,init):
    return [b[init.z_index[0]:init.z_index[-1]+1] for b in batch]


def main_effect_update(batch,init,mu_1,sigma_1):
    
    
        
    
    valid_mains = get_valid_main_indices(batch,init)
    
    
    if(len(valid_mains) == 0):
      
     # return the prior
        return [mu_1,sigma_1]
    else:
        xz = get_xz_matrix_main_effect(batch,init,valid_mains)
        mu_tmp = mu_1
        sigma_tmp = sigma_1
        X_trn = get_X_trn_unavail_update(xz,init)
        Y_trn = [batch[i][init.reward_index] for i in valid_mains]
        
        
        temp = post_cal(X_trn, Y_trn, init.sigma, mu_tmp, sigma_tmp)
      
      # return the post dist of main eff
        return temp

def get_smaller_sub_function(x,z,init):
    new_vector = init.feat0_function(z,x)+init.feat1_function(z,x)+init.feat2_function(z,x)
    
    
def apply_outer(data_structure):
    return [np.mean(np.array(row),axis=0) for row in data_structure]
    
##for each row in Z?    
def apply_inner(x,Z,init):
    return [init.feat0_function(z,x)+init.feat1_function(z,x)+init.feat2_function(z,x) for z in Z]
    
    
def get_F_all(X_null,Z_trn,init):
    #print('called F all')
    
    DS = [apply_inner(x,Z_trn,init) for x in X_null]
    #print(np.array(DS).shape)
    #ds = apply_inner(X_null,Z_trn,init)
    m = apply_outer(DS)
    #print(np.array(m).shape)
    return m
  
    #F.all <- t(sapply(X.null, function(x) apply(apply(Z.trn, 1, function(z) c(input$feat0(z, x), input$feat1(z, x), input$feat2(z, x))), 1, mean)))
    
def get_r(f,alpha):
    return np.matmul(f,alpha)

def policy_update( init,batch,mu_0,sigma_0,mu_1,sigma_1,mu_2,sigma_2,proxy=False,etaf=None):
    #the_psi = psi()
    the_psi = init.psi
    txt_est = txt_effect_update(batch,init,mu_1,sigma_1,mu_2,sigma_2)
    if proxy:
        unavail_params =  unavail_update(batch,init,mu_0,sigma_0)
        alpha_0 = unavail_params[0]
        main_params = main_effect_update(batch,init,mu_1,sigma_1)
        alpha_1 = main_params[0]
        alpha_2 = txt_est[0]
        
        p_avail = np.array([b[init.avail_index] for b in batch]).mean()
        X_null = np.arange(0,1/(1-init.lambda_knot),.01)
        Z_trn =  get_Z_trn(batch,init)
        F_all = get_F_all(X_null,Z_trn,init)
        F0 = [f[:len(alpha_0)] for f in F_all]
        F1 = [f[len(alpha_0):len(alpha_0)+len(alpha_1)] for f in F_all]
        F2 = [f[len(alpha_0)+len(alpha_1):] for f in F_all]
           
            
        r0_vec = get_r(np.array(F0),np.array(alpha_0)) 
        r1_vec = get_r(np.array(F1),np.array(alpha_1)) 
        r2_vec = get_r(np.array(F2),np.array(alpha_2)) 
        
        
        psi_mat = init.psi.eval_all_x_all_dim(X_null)
        #print(np.array(psi_mat).shape)
        tx = np.matmul(np.transpose(psi_mat),psi_mat)
        inv_cov = solve(tx+np.random.rand(tx.shape[0],tx.shape[0]),np.eye(tx.shape[0]))
        psi_mat_irs = init.psi.eval_all_x_all_dim(np.transpose(X_null)*init.lambda_knot+1)
        psi_mat_drs = init.psi.eval_all_x_all_dim(np.transpose(X_null)*init.lambda_knot)
        
        
        
        #psi_mat_irs = np.transpose([the_psi.eval_all_x_all_dim(xrow*init.lambda_knot+1) for xrow in X_null])
        #psi_mat_drs = np.transpose([the_psi.eval_all_x_all_dim(xrow*init.lambda_knot)for xrow in X_null])
        psi_mat_bar = init.prob_sedentary*psi_mat_irs+(1-init.prob_sedentary)*psi_mat_drs
        
        theta_zero = np.zeros(psi_mat.shape[1])
        theta_one = np.zeros(psi_mat.shape[1])
        
        
        theta_bar = theta_one*p_avail + (1-p_avail)*theta_zero
        
        Yone_zero = r1_vec+init.gamma_mdp * np.matmul(psi_mat_bar,theta_bar)
        Yone_one = r1_vec+init.gamma_mdp * np.matmul(psi_mat_irs,theta_bar)
        
        #print(Yone_zero.shape)
        #print(Yone_one.shape)
        
        good_indices = set([i for i in range(len(Yone_one)) if Yone_one[i]-Yone_zero[i]>0])
        Y_one = [Yone_one[i] if i in good_indices else Yone_zero[i] for i in range(len(Yone_zero))]
        
        
        
        Y_zero =  r0_vec+init.gamma_mdp * np.matmul(psi_mat_bar,theta_bar)
        
        #print(Y_one-np.matmul(psi_mat,theta_one))
        
        delta = max(abs(Y_one-np.matmul(psi_mat,theta_one)).all(),abs(Y_zero-np.matmul(psi_mat,theta_zero)).all())
        
        threshold = 1e-2
        
        
        iter_max = 100 
        iter_now = 0
        
        
        ##what is going on here?
        while iter_now<iter_max and delta>threshold:
            theta_one = np.matmul(inv_cov,np.matmul(np.transpose(psi_mat),Y_one))
            theta_zero = np.matmul(inv_cov,np.matmul(np.transpose(psi_mat),Y_zero))
            
            theta_bar = theta_one*p_avail + (1-p_avail)*theta_zero
            
            
            ##Bellman Operator
            
            
            Yone_one = r1_vec+init.gamma_mdp * np.matmul(psi_mat_irs,theta_bar)
            Yone_zero = r1_vec+init.gamma_mdp * np.matmul(psi_mat_bar,theta_bar)
            
            good_indices = set([i for i in range(len(Yone_one)) if Yone_one[i]-Yone_zero[i]>0])
            Y_one = [Yone_one[i] if i in good_indices else Yone_zero[i] for i in range(len(Yone_zero))]
            
            Y_zero =  r0_vec+init.gamma_mdp * np.matmul(psi_mat_bar,theta_bar)
            
            delta = max(abs(Y_one-np.matmul(psi_mat,theta_one)).all(),abs(Y_zero-np.matmul(psi_mat,theta_zero)).all())
        
            
            
            iter_now = iter_now+1
        
        
        ##UPDATE eta
        eta_params = {}
        eta_params['p_sed']=init.prob_sedentary
        eta_params['gamma_mdp']=init.gamma_mdp
        eta_params['theta_bar']=theta_bar
        eta_params['lamda']=init.lambda_knot
        eta_params['init_function']=False
        eta_params['psi']=the_psi
        eta_params['weight']=init.weight
        
        return [txt_est[0],txt_est[1],eta_params]
        
        #etaf.update_params(eta_params)
        
        
      
        #return txt_est
        
        #return delta
        #print(r0_vec.shape)
            
        #index0 <- 1:length(alpha0)
        #index1 <- length(alpha0) + 1:length(alpha1)
        #index2 <- tail(1:ncol(F.all), length(alpha2))
        #F0 <- F.all[, index0]
        #F1 <- F.all[, index1]
        #F2 <- F.all[, index2]
    
        
    
    
    return txt_est
   
        
