import numpy as np
import bandit

class eta:
    
    def get_vec_indices(self,pZ):
        return [i for i in range(2,pZ+1)]
    
    
    
    def eta_init(self,x):
        return 0
    
    def eta_function(self,x):
        
        
        
        eta_hat = (1-self.p_sed)*np.dot(np.transpose(self.theta_bar),psi(self.lambda*x))-psi(self.lambda*x+1)*(1-self.gamma_mdp)
        
        return self.weight*eta_hat+(1-self.weight)*(self.eta_init(x))
    
    def update_vars(self,state_params):
        
        
         eta.fn = function(x) {
      
      eta.hat <- c((1-input$p.sed)* t(theta.bar)%*%(psi(input$lambda*x)-psi(input$lambda*x+1)) * (1-input$gamma.mdp))
      input$weight * eta.hat + (1-input$weight) * input$eta.init(x)
      
      
    }
        
    def __init__(self,xi=10):
        
        self.pZ = 2
        self.pi_max = .8
        self.pi_min = .1


        self.sigma = 1


        self.xi  = xi
        
        self.gamma_mdp = .9
        self.lambda_knot = .9 
        self.prob_sedentary = .9 
        self.weight = .5 
        self.z_init = [0 for i in range(self.pZ)]
        self.z_index = [i+1 for i in range(len(self.z_init))]
        self.x_index = self.z_index[-1]+1
        self.avail_index = self.x_index+1
        self.action_index = self.avail_index+1
        self.prob_index = self.action_index+1
        self.reward_index = self.prob_index+1
        
        
        self.mu_0 = [0 for i in range(len(self.feat0_function(self.z_init,0)))]
        self.mu_1 = [0 for i in range(len(self.feat1_function(self.z_init,0)))]
        self.mu_2 = [0 for i in range(len(self.feat2_function(self.z_init,0)))]
        
        self.sigma_0 = np.diag([10 for i in range(len(self.mu_0))])
        self.sigma_1 = np.diag([10 for i in range(len(self.mu_1))])
        self.sigma_2 = np.diag([10 for i in range(len(self.mu_2))])
        
        
        
        