import numpy as np
import bandit

class eta:
    
    def __init__(self):
        
        self.p_sed = 0 
        self.theta_bar = 0 
        self.lamda =0 
        self.gamma_mdp = 0 
        self.weight = 0 
        self.function_init = True
        self.psi = None
    
    #bsb <- create.bspline.basis (range=c(0, 1/(1-input$lambda)), nbasis=50, norder = 4)
    #psi = function(x) c(eval.basis(x, bsb))
    def psi(self,x):
         c(eval.basis(x, bsb))
    
    def eta_init(self,x):
        
        
        return 0
    
    
    def eta_function(self,x):
        #print('called eta function')
       
        #print(self.psi.simple_psi(self.lamda*x))
        #print((1-self.p_sed)*np.dot(np.transpose(self.theta_bar),self.psi.eval_all_x_all_dim(self.lamda*x)-self.psi.eval_all_x_all_dim(self.lamda*x+1))*(1-self.gamma_mdp))
        
        #eta_hat = (1-self.p_sed)*np.dot(np.transpose(self.theta_bar),self.psi.simple_psi(self.lamda*x))-self.psi.simple_psi(self.lamda*x+1)*(1-self.gamma_mdp)
        
        
        eta_hat = (1-self.p_sed)*np.dot(np.transpose(self.theta_bar),self.psi.eval_all_x_all_dim(self.lamda*x)-self.psi.eval_all_x_all_dim(self.lamda*x+1))*(1-self.gamma_mdp)
        
        return self.weight*eta_hat+(1-self.weight)*(self.eta_init(x))
    
    def update_params(self,state_params):
        self.p_sed = state_params['p_sed']
        
        self.theta_bar = state_params['theta_bar']
        self.lamda = state_params['lamda']
        self.gamma_mdp = state_params['gamma_mdp']
        
        self.function_init = state_params['init_function']
        self.psi = state_params['psi']
        self.weight = state_params['weight']
   
        
        
        
        