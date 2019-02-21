import pickle
import participant
import random
import numpy as np
import psi

class TS_global_params:
    
    
    
    '''
    Keeps track of hyper-parameters for any TS procedure. 
    '''
    
    def __init__(self,xi=10,baseline_features=None,psi_features=None,resp_features=None):
        self.nums = set([np.float64,int,float])
        self.pi_max = .8
        self.pi_min = .1
        self.sigma = 1
        self.num_baseline_features = baseline_features
        self.psi_features = psi_features
        self.num_responsivity_features = resp_features
        self.baseline_indices = None
        self.psi_indices =None
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
        self.mu_theta =np.ones(self.theta_dim)
        self.sigma_theta =self.get_theta(self.theta_dim)
        #self.sigma_v=.5*np.eye(2)
        self.sigma_v = np.array([[5.21906177e-00, 0.00000000e+00],
                               [0.00000000e+00, 1.00000000e-2]])
        #self.sigma_v = np.array([[1.0e-6,0.0],[0.0,1.0e-6]])
        #u1
        #22886.50901787
        ##u2
        #9821.60955232
        ##off diagonal
        ##14992.74343105
        self.sigma_u =np.array([[2449.49426332e+01,  739.56533143e+01],
                             [ 739.56533143e+01,  223.29672266e+01]])
            #np.array([[200, 150],
            #[150, 100 ]])
        self.rho_term = 1.9999
        self.u1 = 22886.50901787
        self.u2 = 9821.60955232
        #90800.30211642
        self.noise_term=95224.65812823
            #50800.30211642
        self.cov=np.array([1])
        #self.psi = psi.psi()
        self.decision_times = 1
        self.kdim = None

        self.last_global_update_time = None
        
        
        self.user_id_index=None
        self.user_day_index = None
        self.write_directory ='../temp'
            #'../../regal/murphy_lab/pooling/temp_EB'
        self.updated_cov = False
        self.history = None
    
        self.one_hot_indices = {'tod-0-dow-0-wea-0-pre-0-loc-2': 0,
    'tod-0-dow-0-wea-0-pre-0-loc-0': 1,
        'tod-0-dow-1-wea-1-pre-0-loc-0': 2,
            'tod-1-dow-0-wea-0-pre-0-loc-2': 3,
                'tod-0-dow-0-wea-1-pre-1-loc-3': 4,
                    'tod-0-dow-0-wea-1-pre-1-loc-2': 5,
                        'tod-0-dow-1-wea-0-pre-1-loc-1': 6,
                            'tod-0-dow-0-wea-1-pre-0-loc-3': 7,
                                'tod-1-dow-0-wea-1-pre-0-loc-1': 8,
                                    'tod-1-dow-1-wea-0-pre-0-loc-2': 9,
                                        'tod-0-dow-0-wea-1-pre-0-loc-0': 10,
                                            'tod-1-dow-0-wea-0-pre-0-loc-0': 11,
                                                'tod-0-dow-1-wea-0-pre-0-loc-2': 12,
                                                    'tod-1-dow-1-wea-0-pre-1-loc-3': 13,
                                                        'tod-1-dow-0-wea-0-pre-0-loc-1': 14,
                                                            'tod-1-dow-0-wea-1-pre-1-loc-2': 15,
                                                                'tod-0-dow-1-wea-1-pre-0-loc-3': 16,
                                                                    'tod-1-dow-0-wea-0-pre-1-loc-3': 17,
                                                                        'tod-0-dow-0-wea-1-pre-0-loc-2': 18,
                                                                            'tod-0-dow-0-wea-0-pre-1-loc-3': 19,
                                                                                'tod-1-dow-1-wea-1-pre-1-loc-1': 20,
                                                                                    'tod-0-dow-1-wea-0-pre-1-loc-2': 21,
                                                                                        'tod-0-dow-0-wea-0-pre-1-loc-2': 22,
                                                                                            'tod-1-dow-1-wea-1-pre-0-loc-0': 23,
                                                                                                'tod-0-dow-1-wea-0-pre-0-loc-1': 24,
                                                                                                    'tod-0-dow-1-wea-1-pre-0-loc-2': 25,
                                                                                                        'tod-0-dow-1-wea-0-pre-0-loc-0': 26,
                                                                                                            'tod-1-dow-0-wea-1-pre-0-loc-3': 27,
                                                                                                                'tod-1-dow-0-wea-1-pre-1-loc-0': 28,
                                                                                                                    'tod-0-dow-1-wea-1-pre-1-loc-2': 29,
                                                                                                                        'tod-0-dow-1-wea-1-pre-1-loc-1': 30,
                                                                                                                            'tod-1-dow-1-wea-1-pre-0-loc-3': 31,
                                                                                                                                'tod-1-dow-0-wea-0-pre-1-loc-2': 32,
                                                                                                                                    'tod-1-dow-1-wea-0-pre-1-loc-1': 33,
                                                                                                                                        'tod-0-dow-1-wea-1-pre-1-loc-0': 34,
                                                                                                                                            'tod-1-dow-1-wea-0-pre-1-loc-2': 35,
                                                                                                                                                'tod-1-dow-1-wea-1-pre-1-loc-0': 36,
                                                                                                                                                    'tod-0-dow-0-wea-1-pre-1-loc-1': 37,
                                                                                                                                                        'tod-0-dow-1-wea-0-pre-1-loc-0': 38,
                                                                                                                                                            'tod-1-dow-1-wea-0-pre-0-loc-3': 39,
                                                                                                                                                                'tod-0-dow-1-wea-1-pre-0-loc-1': 40,
                                                                                                                                                                    'tod-1-dow-1-wea-1-pre-0-loc-1': 41,
                                                                                                                                                                        'tod-0-dow-0-wea-1-pre-1-loc-0': 42,
                                                                                                                                                                            'tod-1-dow-0-wea-1-pre-0-loc-2': 43,
                                                                                                                                                                                'tod-1-dow-0-wea-1-pre-1-loc-1': 44,
                                                                                                                                                                                    'tod-0-dow-1-wea-0-pre-0-loc-3': 45,
                                                                                                                                                                                        'tod-1-dow-1-wea-1-pre-1-loc-2': 46,
                                                                                                                                                                                            'tod-0-dow-0-wea-1-pre-0-loc-1': 47,
                                                                                                                                                                                                'tod-1-dow-1-wea-1-pre-1-loc-3': 48,
                                                                                                                                                                                                    'tod-1-dow-0-wea-1-pre-1-loc-3': 49,
                                                                                                                                                                                                        'tod-0-dow-1-wea-1-pre-1-loc-3': 50,
                                                                                                                                                                                                            'tod-1-dow-1-wea-0-pre-0-loc-0': 51,
                                                                                                                                                                                                                'tod-0-dow-0-wea-0-pre-0-loc-1': 52,
                                                                                                                                                                                                                    'tod-1-dow-0-wea-0-pre-1-loc-0': 53,
                                                                                                                                                                                                                        'tod-1-dow-0-wea-0-pre-1-loc-1': 54,
                                                                                                                                                                                                                            'tod-0-dow-0-wea-0-pre-1-loc-1': 55,
                                                                                                                                                                                                                                'tod-1-dow-1-wea-0-pre-0-loc-1': 56,
                                                                                                                                                                                                                                    'tod-0-dow-1-wea-0-pre-1-loc-3': 57,
                                                                                                                                                                                                                                        'tod-1-dow-1-wea-0-pre-1-loc-0': 58,
                                                                                                                                                                                                                                            'tod-0-dow-0-wea-0-pre-0-loc-3': 59,
                                                                                                                                                                                                                                                'tod-1-dow-0-wea-0-pre-0-loc-3': 60,
                                                                                                                                                                                                                                                    'tod-1-dow-1-wea-1-pre-0-loc-2': 61,
                                                                                                                                                                                                                                                        'tod-0-dow-0-wea-0-pre-1-loc-0': 62,
                                                                                                                                                                                                                                                            'tod-1-dow-0-wea-1-pre-0-loc-0': 63}
    
    
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
