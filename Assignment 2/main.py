#references:
#https://pytorch.org/tutorials/beginner/basics/intro.html
#https://visualstudiomagazine.com/articles/2020/09/10/pytorch-dataloader.aspx

import torch as T
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

#print(torch.cuda.is_available())
#Input = np.loadtxt("even_mnist.csv")
#training_data = Input[3000:29493,:]
#test_data = Input[0:3000,:]

#device = T.device("cuda" if T.cuda.is_available() else "cpu")
device = T.device("cpu")

hidden_layer_width = 200
learning_rate = 1e-3
batch_size = 26492
epochs = 5

class DigitsDataset(T.utils.data.Dataset):

  def __init__(self, src_file, TEST):
    if TEST == True:            #use random split?
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
    #if T.is_tensor(idx):
    #  idx = idx.tolist()
    img = self.x_data[idx, 0:196]
    dgt = T.div(self.y_data[idx], 2, rounding_mode='floor')
    #sample = \
    #  { 'image' : img, 'digit' : dgt }
    return img.float(), dgt.long()

train_dataset = DigitsDataset("even_mnist.csv", TEST=False)
test_dataset = DigitsDataset("even_mnist.csv", TEST=True)

train_dataloader = T.utils.data.DataLoader(train_dataset, batch_size=26492, shuffle=True)
test_dataloader = T.utils.data.DataLoader(test_dataset, batch_size=3000, shuffle=True)

class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(14 * 14, hidden_layer_width),
      nn.ReLU(),
      nn.Linear(hidden_layer_width, hidden_layer_width),
      nn.ReLU(),
      nn.Linear(hidden_layer_width, 5),
    )

  def forward(self, x):
    logits = self.linear_relu_stack(x)
    return logits

model = NeuralNetwork().to(device)

#X = T.rand(1, 196, device=device)
#logits = model(X)
#pred_probab = nn.Softmax(dim=1)(logits)
#y_pred = pred_probab.argmax(1)
#print(f"Predicted class: {y_pred}")




def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  for batch, (X, y) in enumerate(dataloader):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0

  with T.no_grad():
    for X, y in dataloader:
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(T.float).sum().item()

  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


loss_fn = nn.CrossEntropyLoss()
optimizer = T.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

def prediction(input_tensor):
  logits = model(input_tensor)
  pred_probab = nn.Softmax()(logits) #dim=1
  y_pred = pred_probab.argmax()*2 #1
  return y_pred.tolist()

Input = np.loadtxt("even_mnist.csv")

figure = plt.figure(figsize=(8, 8))
cols, rows = 5, 3
for n in range(15):
    sample = Input[n, 0:197]
    picture_as_array = []
    picture = np.zeros([14, 14], int)
    for i in range(196):
        picture[i // 14, i % 14] = sample[i]
        picture_as_array.append(sample[i])
    figure.add_subplot(rows, cols, n+1)
    picture_as_tensor = T.tensor(np.array(picture_as_array)).to(device)
    title = str(prediction(picture_as_tensor.float()))+" (actual: "+str(round(sample[196]))+")"
    plt.title(title)
    plt.axis("off")
    plt.imshow(np.array(picture), cmap="gray")
plt.show()
