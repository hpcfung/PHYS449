import torch
import matplotlib.pyplot as plt
import numpy as np

#print(torch.cuda.is_available())
Input = np.loadtxt("even_mnist.csv")

figure = plt.figure(figsize=(8, 8))
cols, rows = 5, 3
for n in range(15):
    sample = Input[n, 0:197]
    picture = np.zeros([14, 14], int)
    for i in range(196):
        picture[i // 14, i % 14] = sample[i]
    figure.add_subplot(rows, cols, n+1)
    plt.axis("off")
    plt.imshow(np.array(picture), cmap="gray")
plt.show()
