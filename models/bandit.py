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

