import numpy as np
with open('1.in') as f:
    Input0 = []
    for line in f: # read rest of lines
        Input0.append([int(x) for x in line.split()])
    Input = np.array(Input0)
    num_rows, num_cols = Input.shape
    T = Input[:, num_cols-1] #target variable
    T = T.reshape(-1,1)
    x = np.delete(Input, num_cols-1, 1) #input variables
    print(x)
    #P = design matrix
    #optimized weights = (P^tranpose * P)^{-1} * P^tranpose * T

    #Gradient descent
    #w[i] = ith weight
    #a = learning rate
    #D = a*(t-w^tranpose * x)*x
    #w[i] = w[i]+D
