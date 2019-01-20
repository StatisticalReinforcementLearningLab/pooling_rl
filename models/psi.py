import numpy as np
from scipy.interpolate import BSpline
import time 

class psi:
    
    def __init__(self,xmin=0,xmax=10,num_dim=50,degree=4):
        self.xmin = xmin
        self.xmax = xmax
        self.num_dim = num_dim 
        self.degree = degree 
        self.bs = {}
        self.init_bs()
        
        
    def init_bs(self):
        
        for i in range(self.num_dim):
            knots = np.arange(self.xmin,self.xmax,(self.xmax-self.xmin)/self.num_dim)
            
            c = np.zeros(self.num_dim)
            c.put(i,1)
            b = BSpline(knots,c,self.degree)
            self.bs[i]=b
    
    
    def eval_function_index_short(self,x,index):
        
        return self.bs[index](x)
    
    
    def eval_function_index(self,x,index):
        knots = np.arange(self.xmin,self.xmax,(self.xmax-self.xmin)/self.num_dim)
            
        c = np.zeros(self.num_dim)
        c.put(index,1)
        b = BSpline(knots,c,self.degree)
        return b(x)
    
    def eval_all_x_all_dim(self,x):
        psi_mat = [[] for i in range(self.num_dim)]
        #psi_mat = [[] for i in range(len(x))]
        #start = time.time()
        
        for i in range(self.num_dim):
            psi_mat[i]=self.eval_function_index(np.array(x).astype(np.float),i)
        
        #for xi in range(len(x)):
            #psi_mat[xi]=[self.eval_function_index(xi,i) for i in range(self.num_dim)]
        #end = time.time()
        #print(end-start)
        return np.transpose(np.array(psi_mat))
    
    
    def simple_psi(self,x):
        psi_mat = [[] for i in range(self.num_dim)]
        #psi_mat = [[] for i in range(len(x))]
        #start = time.time()
        
        for i in range(self.num_dim):
            psi_mat[i]=self.eval_function_index(np.array(x).astype(np.float),i)
        
        #for xi in range(len(x)):
            #psi_mat[xi]=[self.eval_function_index(xi,i) for i in range(self.num_dim)]
        #end = time.time()
        #print(end-start)
        return np.array(psi_mat).sum()    

    #np.transpose(np.array(psi_mat))

        
   
        
        
        
        