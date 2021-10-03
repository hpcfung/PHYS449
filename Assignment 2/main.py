#references:
#https://pytorch.org/tutorials/beginner/basics/intro.html
#https://visualstudiomagazine.com/articles/2020/09/10/pytorch-dataloader.aspx

import torch as T
import numpy as np

#print(torch.cuda.is_available())
#Input = np.loadtxt("even_mnist.csv")
#training_data = Input[3000:29493,:]
#test_data = Input[0:3000,:]

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class DigitsDataset(T.utils.data.Dataset):

  def __init__(self, src_file, TEST):
    if TEST == True:
      x_tmp = np.loadtxt(src_file, max_rows=3000, usecols=range(0, 196))
      y_tmp = np.loadtxt(src_file, max_rows=3000, usecols=196)
    else:
      x_tmp = np.loadtxt(src_file, skiprows=3000, usecols=range(0, 196))
      y_tmp = np.loadtxt(src_file, skiprows=3000, usecols=196)

    self.x_data = T.tensor(x_tmp).to(device)
    self.y_data = T.tensor(y_tmp).to(device)

  def __len__(self):
    return len(self.x_data)  # required

  def __getitem__(self, idx):
    if T.is_tensor(idx):
      idx = idx.tolist()
    img = self.x_data[idx, 0:196]
    dgt = self.y_data[idx]
    sample = \
      { 'image' : img, 'digit' : dgt }
    return sample

train_dataset = DigitsDataset("even_mnist.csv", TEST=False)
test_dataset = DigitsDataset("even_mnist.csv", TEST=True)
train_dataloader = T.utils.data.DataLoader(train_dataset, batch_size=26492, shuffle=True)

print(train_dataset.__len__())
print(test_dataset.__len__())
