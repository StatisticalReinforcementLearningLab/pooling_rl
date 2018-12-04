import numpy as np

def feat2_function(z,x):
    return [1,z[0],x]

def function_zero(x):
    return 0


def feat0_function(z,x):
    temp =  [1]
    temp.extend(z)
    temp.append(x)
    return temp

def feat1_function(z,x):
    temp =  [1]
    temp.extend(z)
    temp.append(x)
    return temp

def init_mu(val,length):
    return [val]* length


def calculate_prob(z, x, mu, Sigma, eta, input):
    
    
    pos_mean = np.dot(feat2_function(z,x),mu)
    #pos.var <- c(t(input$feat2(z, x)) %*% Sigma %*% input$feat2(z, x))
    pos_var = np.dot(np.dot(np.transpose(feat2_function(z,x)),Sigma),feat2_function(z,x))
    pos_var = max(0,pos_var)
    
    #margin = eta (x) * input$xi
    
    print(pos_var)

