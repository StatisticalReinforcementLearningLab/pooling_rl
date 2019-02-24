import pickle
import participant
import random
import numpy as np
import psi

class TS_batch_params:
    
    
    
    '''
    Keeps track of hyper-parameters for any TS procedure. 
    '''
    
    def __init__(self,xi=10,baseline_features=None,psi_features=None,resp_features=None):
        self.nums = set([np.float64,int,float])
        self.pi_max = .8
        self.pi_min = .1
        self.sigma = 1
        self.baseline_features=baseline_features
        self.responsivity_features = resp_features
        self.num_baseline_features = len(baseline_features)
        self.psi_features = psi_features
        self.num_responsivity_features = len(resp_features)
        self.baseline_indices = [i for i in range(self.num_baseline_features)]
        self.psi_indices = psi_features
        self.responsivity_indices = None
        
        self.xi  = xi
        
        self.update_period=7
        self.gamma_mdp = .9
        self.lambda_knot = .9 
        self.prob_sedentary = .9 
        self.weight = .5 
        #print(self.baseline_features)
        #self.z_index = [i+1 for i in range(len(self.baseline_features))]
        #self.x_index = len(self.z_index)+1
        #self.avail_index = self.x_index+1
        #self.action_index = self.avail_index+1
        #self.prob_index = self.action_index+1
        #self.reward_index = self.prob_index+1
        
        #2 has to do with random effects, not likely to change soon
        self.theta_dim =1+self.num_baseline_features + 2*(1+self.num_responsivity_features)
        print(self.theta_dim)

            #90800.30211642
        #tried random
        #95224.65812823
            #50800.30211642
        self.cov=np.array([1])
        #self.psi = psi.psi()
        self.decision_times = 1
        self.kdim = self.theta_dim+2

        self.last_global_update_time = None
        
        
        self.user_id_index=None
        self.user_day_index = None
        self.write_directory ='../temp'
            #'../../regal/murphy_lab/pooling/temp_EB'
        self.updated_cov = False
        self.history = None
        
        self.mus0 = None
        self.sigmas0 =None
        
        self.mus1 = None
        self.sigmas1 =None
        
        self.mus2 = None
        self.sigmas2 = None
        
        self.batch = {}
        self.batch_index = {}
        
        self.etas = {}
        
        self.last_update = {}
    
    
    def update_mus(self,pid,mu_value,which_mu):
        if which_mu==0:
            self.mus0=mu_value
        
        if which_mu==1:
            self.mus1=mu_value
        
        if which_mu==2:
            self.mus2=mu_value

    def update_sigmas(self,pid,sigma_value,which_sigma):
        if which_sigma==0:
            self.sigmas0=sigma_value
        
        if which_sigma==1:
            self.sigmas1=sigma_value

        if which_sigma==2:
                self.sigmas2=sigma_value
    
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
    
    def get_mu1(self,num_baseline_features):
        return [0 for i in range(num_baseline_features+1)]
    
    def get_mu2(self,num_responsivity_features):
        return [0 for i in range(num_responsivity_features+1)]
    
    def get_asigma(self,adim):
        return np.diag([10 for i in range(adim)])
    
    
    def comput_rho(self,sigma_u):
        return (sigma_u[0][1]/( sigma_u[0][0]**.5*sigma_u[1][1]**.5))+1
    
    
    def update_params(self,pdict):
        self.noise_term=pdict['noise']
        self.sigma_u = pdict['sigma_u']
        self.sigma_v = pdict['sigma_v']
        #save rho term too
        self.rho_term = self.comput_rho(pdict['sigma_u'])
        self.cov = pdict['cov']
        self.updated_cov=True

    def get_theta(self,dim_baseline):
        m = 1000*np.eye(dim_baseline)
        #m = np.add(m,.1)
        return m

    def update_cov(self,current_dts):
        cov = np.eye(current_dts)
        #cov = np.add(cov,.001)
        self.cov=cov
