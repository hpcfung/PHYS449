import torch
import numpy as np

#print(torch.cuda.is_available())
Input = np.loadtxt("even_mnist.csv")
training_data = Input[3000:29493,:]
test_data = Input[0:3000,:]
print(Input.shape)
print(training_data.shape)
print(test_data.shape)
