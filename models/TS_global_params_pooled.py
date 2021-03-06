import pickle
import participant
import random
import numpy as np
import psi

class TS_global_params:
    
    
    
    '''
    Keeps track of hyper-parameters for any TS procedure. 
    '''
    
    def __init__(self,xi=10,baseline_features=None,psi_features=None,responsivity_keys=None,uparams = None):
        self.nums = set([np.float64,int,float])
        self.pi_max = 0.8
        self.pi_min = 0.1
        self.sigma =1.15            #6**.5
        self.baseline_features=baseline_features
        
        #self.baseline_features = baseline_keys
        #self.responsivity_features = responsivity_keys
        self.responsivity_keys = responsivity_keys
        self.num_baseline_features = len(baseline_features)
        self.psi_features = psi_features
        self.num_responsivity_features = len(responsivity_keys)
        #self.baseline_indices = [i for i in range(self.num_baseline_features)]
        self.psi_indices = psi_features
        #self.responsivity_indices = None
        
        self.xi  = xi
        
        self.update_period=7
        self.gamma_mdp = .9
        self.lambda_knot = .9 
        self.prob_sedentary = .9 
        self.weight = .5
        
        self.inv_term = None
        self.to_save_params = {}
        
        #print(self.baseline_features)
        #self.z_index = [i+1 for i in range(len(self.baseline_features))]
        #self.x_index = len(self.z_index)+1
        #self.avail_index = self.x_index+1
        #self.action_index = self.avail_index+1
        #self.prob_index = self.action_index+1
        #self.reward_index = self.prob_index+1
        
        #2 has to do with random effects, not likely to change soon
        self.theta_dim =1+self.num_baseline_features + 2*(1+self.num_responsivity_features)
        self.baseline_indices =  [i for i in range(self.theta_dim)]
        #print(self.theta_dim)
        self.mu_theta =np.zeros(self.theta_dim)
        self.mu_theta[0]=4.6
        self.sigma_theta =self.get_theta(self.theta_dim)
        self.lr = 0.0001
        #self.sigma_v=.5*np.eye(2)
        
        #self.sigma_v =np.array([[4492.02905157 ,   0.0        ],[   0.0 ,        2027.39758508]])
            #np.array([[84268.04299068,     0.0        ],[    0.0,         13464.64044625]])
        #self.sigma_v = np.array([[5.21906177e-00, 0.00000000e+00],
        # [0.00000000e+00, 1.00000000e-2]])
        #self.sigma_v = np.array([[1.0e-6,0.0],[0.0,1.0e-6]])
        #u1
        #22886.50901787
        ##u2
        #9821.60955232
        ##off diagonal
        ##14992.74343105
        #self.sigma_u =np.array([[2449.49426332e+01,  739.56533143e+01],[ 739.56533143e+01,  223.29672266e+01]])
        
        #continuous
        #self.sigma_u =np.array([[ 0.28800755, -0.17554317],
        #                        [-0.17554317,  0.5466027 ]])
        ##non continuous
        #self.sigma_u =np.array([[2.17412222, 0.61305586],
        #  [0.61305586, 1.39429461]])
          
          #self.sigma_u =np.array([[0.4227,-0.1540 ],[-0.1540,   0.5397]])
        self.sigma_u =np.array([[0.06449696, 0.01502549 ],[ 0.01502549,    0.00896479]])
            #np.array([[0.0710, 0.0189 ],[ 0.0189,    0.0061]])
            #np.array([[0.0710, 0.0189 ],[ 0.0189,    0.0061]])
            #np.array([[0.1866738675,0.22266198246 ],[ 0.22266198246,    0.23096490627]])
        #best so far
            #np.array([[0.06449696, 0.01502549 ],[ 0.01502549,    0.00896479]])
            #np.array([[0.8629,-0.0971 ],[-0.0971,   0.7548]])
            #np.array([[0.07381617,,-0.10 ],[-0.10,   0.29114969]])
            #np.array([[0.04970398,   0.01142578 ],[   0.01142578,   0.0042747]])
     
            #np.array([[0.0984,-0.1540 ],[-0.1540,   0.40]])
            #np.array([[0.1800,-0.1977, ],[-0.1977,   0.4234]])
            #np.array([[0.28613145, 0.0636587 ],[0.0636587 , 0.02591053]])
            #np.array([[0.33,0.0486 ],[0.0486 , 0.1]])
            #np.array([[0.28613145, 0.0636587 ],[0.0636587 , 0.02591053]])
            #np.array([[0.00001,0.000001 ],[0.000001 , 0.00001]])
            #np.array([[0.28613145, 0.0636587 ],[0.0636587 , 0.02591053]])
            #np.array([[ 0.27359038, 0.3575237184874702],
            #[0.3575237184874702, 0.5143725]])

##old
#np.array([[ 1.47652434, 0.20616501,],
#                            [0.20616501,  1.15894301]])
        #self.sigma_u =np.array([[200, 150],[150, 100 ]])
        #continuous
        #self.rho_term =0.5575684246756394
        #non continuous
        self.rho_term =1.6248689729968946
            #0.8796
            #0.3123676555453571
            #0.2250
            #1.7838581064765946
            #0.2250
            #-0.21614189352340551
            #0.2250
            #0.2838
            #0.436
            #1.5
        #1.07393274299268683
            #1.7393274299268683
            #1.3521119759922764
            #1.1576026856712
        #continuous
        #self.u1 = 0.28800755
        #self.u1 =1.47652434
        self.u1 =0.06449696
            #0.8629
            #0.07381617
            #0.0984
            #0.04970398
            #0.0984
            #0.04970398
            #0.0984
            #0.1800
            #0.4227
            #0.28613145
            #0.28613145
        #continuous
        #self.u2 = 0.5466027
        #self.u2 =1.15894301
        self.u2 =0.00896479
            #0.7548
            #0.29114969
            #0.4013
            #0.0042747
            #0.4013
            #0.0042747
            #0.4013
            #0.4234
            #0.36
            #0.5397
            #0.02591053
            #0.02591053
        #90800.30211642
        #continuous
        #self.noise_term=6.32098482
        self.noise_term =1.3305
            #2
            #1.3305
            #1.3305
            #1.3305
            #1.5652649189490888
            #5.7
            #1.15
            #**2
        self.o_noise_term =1.3305
    #2
            #1.3305
            #1.3305
            #3
            #1.3040
            #1.15
        #most recent learned
        #7.61294834
            #7.50134618
            #7.50134618
        #7.49989571
            #90800.30211642
        #tried random
        #95224.65812823
            #50800.30211642
        self.cov=np.array([1])
        #self.psi = psi.psi()
        self.decision_times = 1
        self.kdim = self.theta_dim+2

        self.last_global_update_time = None
        
        self.standardize=False
        
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
        if uparams is not None:
            self.init_u_params(uparams)
    
    def init_u_params(self,uparams):
        self.u1 = uparams[0]
        self.u2 = uparams[1]
        self.u3 = uparams[2]
        self.u4 = uparams[3]
    
        self.r12 = uparams[4]
        self.r13 = uparams[5]
        self.r14 = uparams[6]
        self.r23 = uparams[7]
        self.r24 = uparams[8]
        self.r34 = uparams[9]
        
        self.init_u1 = uparams[0]
        self.init_u2 = uparams[1]
        self.init_u3 = uparams[2]
        self.init_u4 = uparams[3]
        
        self.init_r12 = uparams[4]
        self.init_r13 = uparams[5]
        self.init_r14 = uparams[6]
        self.init_r23 = uparams[7]
        self.init_r24 = uparams[8]
        self.init_r34 = uparams[9]
    
    def update_uparams(self,uparams):
        self.u1 = uparams[0]
        self.u2 = uparams[1]
        self.u3 = uparams[2]
        self.u4 = uparams[3]
        
        self.r12 = uparams[4]
        self.r13 = uparams[5]
        self.r14 = uparams[6]
        self.r23 = uparams[7]
        self.r24 = uparams[8]
        self.r34 = uparams[9]
    
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
        return np.diag([1 for i in range(adim)])
    
    
    def comput_rho(self,sigma_u):
        t =sigma_u[0][0]**.5 * sigma_u[1][1]**.5
        r = (sigma_u[0][1]+t)/t
        return r
    
    
    def update_params(self,pdict):
        self.noise_term=pdict['noise']
        self.sigma_u = pdict['sigma_u']
        #self.sigma_v = pdict['sigma_v']
        #save rho term too
        self.rho_term = self.comput_rho(pdict['sigma_u'])
        self.cov = pdict['cov']
        self.updated_cov=True
    
    def update_params_more(self,pdict):
        self.noise_term=pdict['noise']
        
        #self.sigma_v = pdict['sigma_v']
        #save rho term too
        
        self.cov = pdict['cov']
        self.update_uparams(pdict['uparams'])
        self.updated_cov=True
    

    def get_theta(self,dim_baseline):
        m = 1*np.eye(dim_baseline)
        #m = np.add(m,.1)
        return m

    def update_cov(self,current_dts):
        cov = np.eye(current_dts)
        #cov = np.add(cov,.001)
        self.cov=cov


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
