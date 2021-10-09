#references:
#https://pytorch.org/tutorials/beginner/basics/intro.html
#https://visualstudiomagazine.com/articles/2020/09/10/pytorch-dataloader.aspx

import numpy as np
import torch as T
from torch import nn
import json
import sys

if __name__ == '__main__':

    g = open(sys.argv[1], )
    data = json.load(g)
    print(data)
    hidden_layer_width1 = data['hidden_layer_width1']
    hidden_layer_width2 = data['hidden_layer_width2']
    learning_rate = data['learning_rate']
    batch_size = data['batch_size']
    epochs = data['epochs']
    g.close()

    device = T.device("cpu")


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
            return len(self.x_data)

        def __getitem__(self, idx):
            img = self.x_data[idx, 0:196]
            dgt = T.div(self.y_data[idx], 2, rounding_mode='floor')
            return img.float(), dgt.long()


    train_dataset = DigitsDataset("even_mnist.csv", TEST=False)
    test_dataset = DigitsDataset("even_mnist.csv", TEST=True)

    train_dataloader = T.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = T.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(14 * 14, hidden_layer_width1),
                nn.ReLU(),
                nn.Linear(hidden_layer_width1, hidden_layer_width2),
                nn.ReLU(),
                nn.Linear(hidden_layer_width2, 5),
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to(device)

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
                loss = loss.item()
                current = batch * len(X)
                print(f"[samples processed: {current:>5d}/{size:>5d}]"  #Training loss: {loss:>7f}  
                      f"[mini-batch processed: {batch}]")
                test_loop(train_dataloader, model, loss_fn, "Training")
                test_loop(test_dataloader, model, loss_fn, "Test")
                print()


    def test_loop(dataloader, model, loss_fn, train_or_test):
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
        print(f"{train_or_test} accuracy: {(100 * correct):>0.1f}%   {train_or_test} loss: {test_loss:>8f}")

    def test_report(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with T.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(T.float).sum().item()

        test_loss /= num_batches
        print(f"Size of testing set: {size}")
        print(f"Correctly labelled: {round(correct)}   Incorrectly labelled: {size-round(correct)}")
        correct /= size
        print(f"Test accuracy: {(100 * correct):>0.1f}%   Test loss: {test_loss:>8f} \n")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = T.optim.SGD(model.parameters(), lr=learning_rate)
    print()

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
    print("Training complete")
    print(f"-------------------------------------------------------------------------------")
    print(f"Final test")
    print()
    test_report(test_dataloader, model, loss_fn)
