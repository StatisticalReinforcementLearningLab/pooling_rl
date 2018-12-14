import numpy as np
import bandit

class eta:
    
    def __init__(self):
        
        self.p_sed = 0 
        self.theta_bar = 0 
        self.lamda =0 
        self.gamma_mdp = 0 
    
    
    #bsb <- create.bspline.basis (range=c(0, 1/(1-input$lambda)), nbasis=50, norder = 4)
    #psi = function(x) c(eval.basis(x, bsb))
    def psi(self,x):
         c(eval.basis(x, bsb))
    
    def eta_init(self,x):
        return 0
    
    
    def eta_function(self,x):
        
       
    
        
        eta_hat = (1-self.p_sed)*np.dot(np.transpose(self.theta_bar),psi(self.lamda*x))-psi(self.lamda*x+1)*(1-self.gamma_mdp)
        
        return self.weight*eta_hat+(1-self.weight)*(self.eta_init(x))
    
    def update_vars(self,state_params):
        self.p_sed = state_params['p_sed']
        
        self.theta_bar = state_params['theta_bar']
        self.lamda = state_params['lamda']
        self.gamma_mdp = state_params['gamma_mdp']
        
       
        
   
        
        
        
        