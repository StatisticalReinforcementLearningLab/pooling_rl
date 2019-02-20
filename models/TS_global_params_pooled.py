import pickle
import participant
import random
import numpy as np
import psi

class TS_global_params:
    
    
    
    '''
    Keeps track of hyper-parameters for any TS procedure. 
    '''
    
    def __init__(self,xi=10,baseline_features=['ltps','ltps'],psi_features=['ltps']):
        self.nums = set([np.float64,int,float])
        self.pi_max = .8
        self.pi_min = .1
        self.sigma = 1
        self.baseline_features = baseline_features
        self.psi_features = psi_features
        self.baseline_indices = None
        self.psi_indices =None
        
        self.xi  = xi
        
        self.update_period=7
        self.gamma_mdp = .9
        self.lambda_knot = .9 
        self.prob_sedentary = .9 
        self.weight = .5 
        #print(self.baseline_features)
        self.z_index = [i+1 for i in range(len(self.baseline_features))]
        self.x_index = len(self.z_index)+1
        self.avail_index = self.x_index+1
        self.action_index = self.avail_index+1
        self.prob_index = self.action_index+1
        self.reward_index = self.prob_index+1
        #2 has to do with random effects, not likely to change soon
        self.theta_dim = 1+len(self.baseline_features) + 2*(1+len(self.psi_features))
        self.mu_theta =np.ones(self.theta_dim)
        self.sigma_theta =self.get_theta(self.theta_dim)
        self.sigma_v = np.array([[1.0e-6,0.0],[0.0,1.0e-6]])
        self.sigma_u = np.array([[22886.50901787, 14992.74343105],
                              [14992.74343105,  9821.60955232]])
        self.rho_term = 1.9999
        self.u1 = 22886.50901787
        self.u2 = 9821.60955232
        self.noise_term= 90800.30211642
        self.cov=np.array([1])
        self.psi = psi.psi()
        self.decision_times = 1
        self.kdim = None

        self.last_global_update_time = None
        
        
        self.user_id_index=None
        self.user_day_index = None
        self.write_directory ='../temp'
            #'../../regal/murphy_lab/pooling/temp_EB'
        self.updated_cov = False
        self.history = None
        
    def feat0_function(self,z,x):
        
        
        temp =  [1]
        temp.extend(z)
        #print(type(x))
        if type(x) in self.nums:
        
            temp.append(x)
        else:
            temp.extend(x)
        return temp

    def feat1_function(self,z,x):
        temp =  [1]
        temp.extend(z)
        if type(x) in self.nums:
        
            temp.append(x)
        else:
            temp.extend(x)
        return temp    
        
        
    def feat2_function(self,z,x):
        temp = [1,z[0]]
        if type(x) in self.nums:
        
            temp.append(x)
        else:
            temp.extend(x)
        
        return temp
            
    def get_mu0(self,z_init):
        return [0 for i in range(len(self.feat0_function(z_init,0)))]
    
    def get_mu1(self,z_init):
        return [0 for i in range(len(self.feat1_function(z_init,0)))]
    
    def get_mu2(self,z_init):
        return [0 for i in range(len(self.feat2_function(z_init,0)))]
    
    def get_asigma(self,adim):
        return np.diag([10 for i in range(adim)])

    
    def update_params(self,pdict):
        self.noise_term=pdict['noise']
        self.sigma_u = pdict['sigma_u']
        self.sigma_v = pdict['sigma_v']
        self.cov = pdict['cov']
        self.updated_cov=True

    def get_theta(self,dim_baseline):
        m = np.eye(dim_baseline)
        #m = np.add(m,.1)
        return m

    def update_cov(self,current_dts):
        cov = np.eye(current_dts)
        cov = np.add(cov,.001)
        self.cov=cov
