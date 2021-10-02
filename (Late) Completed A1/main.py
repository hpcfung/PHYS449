import numpy as np
import matplotlib.pyplot as plt
import json

val = input("Input file name: ")

Input = np.loadtxt(val+".in")
num_rows, num_cols = Input.shape
X = Input[:,0:num_cols-1] #input variables
T = Input[:,num_cols-1]
b = np.ones((num_rows, 1))
P = np.hstack((b, X)) #design matrix
psuedo_inverse = np.matmul(np.linalg.inv(np.matmul(np.transpose(P),P)),np.transpose(P))
w = np.matmul(psuedo_inverse,T)
print(f"Analytical weights: {w}")

def analytic_plot(): #can be used to plot the fitted hyperplane if the input dimension is 2
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xdata = Input[:, 0]
    ydata = Input[:, 1]
    zdata = Input[:, 2]

    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');  # alpha = 1
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');

    def f(x, y):
        return w[0] + w[1] * x + w[2] * y

    xa = np.linspace(10, 33, 300)
    ya = np.linspace(-10, 65, 500)

    X, Y = np.meshgrid(xa, ya)
    Z = f(X, Y)
    ax.contour3D(X, Y, Z, 50, cmap='binary')

g = open(val+'.json',)
data = json.load(g)
print(data)
alpha = data['learning rate']
N = data['num iter']
g.close()

W = np.random.multivariate_normal(np.zeros(num_cols,float),np.identity(num_cols))

print(f"initial weights: {W}")

def update_per_sample(lastW,I,J):
    prediction = np.dot(lastW,P[J,:])
    return alpha*(Input[J,num_cols-1] - prediction)*P[J,I]

xline = [W[0]]
yline = [W[1]]
zline = [W[2]]
xline1 = [W[0]]
yline1 = [W[1]]
zline1 = [W[2]]

for k in range(N):
    W_last_iteration = W
    for i in range(num_cols): #loop over weights
        delta = 0
        for j in range(num_rows): #loop over samples
            delta = delta + update_per_sample(W_last_iteration,i,j)
        W[i] = W[i] + delta
    xline1.append(W[0])
    yline1.append(W[1])
    zline1.append(W[2])
    if k + 1 < 20:
        print(f"iteration {k + 1}: {W}")
        xline.append(W[0])
        yline.append(W[1])
        zline.append(W[2])
    else:
        if k + 1 < 10000:
            if (k + 1) % 1000 == 0:
                print(f"iteration {k + 1}: {W}")
                xline.append(W[0])
                yline.append(W[1])
                zline.append(W[2])
        else:
            if (k + 1) % 10000 == 0:
                print(f"iteration {k + 1}: {W}")
                xline.append(W[0])
                yline.append(W[1])
                zline.append(W[2])

def gd_plot(): #can be used to plot the history of gradient descent if the input dimension is 2
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot3D(xline1, yline1, zline1, 'gray')
    ax.scatter3D(xline, yline, zline, cmap='Greens')
    ax.scatter3D([w[0]], [w[1]], [w[2]], cmap='Red')
    ax.set_xlabel('bias')
    ax.set_ylabel('w1')
    ax.set_zlabel('w2')

f = open(val+".out","w")
for i in w:
     k = "{:.4f}".format(i)
     f.write(k)
     f.write("\n")
f.write("\n")
for i in W:
     k = "{:.4f}".format(i)
     f.write(k)
     f.write("\n")
f.close()

#plt.show()
