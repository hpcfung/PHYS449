import numpy as np
from pylab import plot, show
import matplotlib.pyplot as plt

initial_x = 1
initial_y = 0
T = 10
N = 100

lower_bound = -1.3
upper_bound = 1.3
plot_grid_size = 10

time_step = T/N

def u(x,y):
    return -y/np.sqrt(x**2 + y**2)

def v(x,y):
    return x/np.sqrt(x**2 + y**2)

def plot_vector_field():
    X,Y = np.meshgrid(np.linspace(lower_bound,upper_bound,plot_grid_size),
                      np.linspace(lower_bound,upper_bound,plot_grid_size))
    U,V = np.meshgrid(np.linspace(lower_bound, upper_bound, plot_grid_size),
                       np.linspace(lower_bound, upper_bound, plot_grid_size))
    for i in range(plot_grid_size):
        for j in range(plot_grid_size):
            U[i,j] = u(X[i,j],Y[i,j])
            V[i,j] = v(X[i,j],Y[i,j])

    plt.quiver(X,Y,U,V)

def euler_met(x0,y0,Delta_t):
    return x0+u(x0,y0)*Delta_t, y0+v(x0,y0)*Delta_t

def rk2(x0,y0,Delta_t):
    k1x = u(x0,y0)*Delta_t
    k1y = v(x0,y0)*Delta_t
    k2x = u(x0+k1x/2,y0+k1y/2)*Delta_t
    k2y = v(x0+k1x/2,y0+k1y/2)*Delta_t
    return x0+k2x, y0+k2y

def rk4(x0,y0,Delta_t):
    k1x = u(x0, y0) * Delta_t
    k1y = v(x0, y0) * Delta_t
    k2x = u(x0 + k1x / 2, y0 + k1y / 2) * Delta_t
    k2y = v(x0 + k1x / 2, y0 + k1y / 2) * Delta_t
    k3x = u(x0 + k2x / 2, y0 + k2y / 2) * Delta_t
    k3y = v(x0 + k2x / 2, y0 + k2y / 2) * Delta_t
    k4x = u(x0 + k3x, y0 + k3y) * Delta_t
    k4y = v(x0 + k3x, y0 + k3y) * Delta_t
    return x0+(k1x+2*k2x+2*k3x+k4x)/6, y0+(k1y+2*k2y+2*k3y+k4y)/6
#no x dep, so that term in the rk method should not matter?
#just imagine setting output var from 1 to 2?

x_points_ana = [initial_x] #analytical sol
y_points_ana = [initial_y]
x_points_euler = [initial_x]
y_points_euler = [initial_y]
x_points_rk2 = [initial_x]
y_points_rk2 = [initial_y]
x_points_rk4 = [initial_x]
y_points_rk4 = [initial_y]
t_points = [0]

t = 0
for k in range(N):
    t = t + time_step
    new_x_ana = np.cos(t)
    new_y_ana = np.sin(t)
    x_points_ana.append(new_x_ana)
    y_points_ana.append(new_y_ana)

    last_x_euler = x_points_euler[k]
    last_y_euler = y_points_euler[k]
    new_x_euler, new_y_euler = euler_met(last_x_euler,last_y_euler,time_step)
    x_points_euler.append(new_x_euler)
    y_points_euler.append(new_y_euler)

    last_x_rk2 = x_points_rk2[k]
    last_y_rk2 = y_points_rk2[k]
    new_x_rk2, new_y_rk2 = rk2(last_x_rk2, last_y_rk2, time_step)
    x_points_rk2.append(new_x_rk2)
    y_points_rk2.append(new_y_rk2)

    last_x_rk4 = x_points_rk4[k]
    last_y_rk4 = y_points_rk4[k]
    new_x_rk4, new_y_rk4 = rk4(last_x_rk4, last_y_rk4, time_step)
    x_points_rk4.append(new_x_rk4)
    y_points_rk4.append(new_y_rk4)

    t_points.append(t)

    if k%20 == 0:
        label = f"t = {t}"
        plt.annotate(label,  # this is the text
                     xy=(new_x_ana, new_y_ana),  # these are the coordinates to position the label
                     xytext=(new_x_ana - 0.5, new_y_ana + 0.5),
                     arrowprops=dict(arrowstyle="->"))
        plt.annotate(label,  # this is the text
                     xy=(new_x_euler, new_y_euler),  # these are the coordinates to position the label
                     xytext=(new_x_euler-0.5, new_y_euler+0.5),
                     arrowprops=dict( arrowstyle="->"))
        plt.annotate(label,  # this is the text
                     xy=(new_x_rk2, new_y_rk2),  # these are the coordinates to position the label
                     xytext=(new_x_rk2 - 0.5, new_y_rk2 + 0.5),
                     arrowprops=dict(arrowstyle="->"))
        plt.annotate(label,  # this is the text
                     xy=(new_x_rk4, new_y_rk4),  # these are the coordinates to position the label
                     xytext=(new_x_rk4 - 0.5, new_y_rk4 + 0.5),
                     arrowprops=dict(arrowstyle="->"))



#pytorch seems to be able to do do backprop automatically even when weight sharing is used
#initial hidden and cell state default to zeros if (h_0, c_0) is not provided
#no need to specify run LSTM for how many times?
#Since: For each element in the input sequence, each layer computes the following function:

plot_vector_field()
plot(x_points_ana, y_points_ana, linestyle='-', marker='o', color='k', linewidth=2)
plot(x_points_euler, y_points_euler, linestyle='-', marker='o', color='b')
plot(x_points_rk2, y_points_rk2, linestyle='-', marker='o', color='g')
plot(x_points_rk4, y_points_rk4, linestyle='-', marker='o', color='r')
show()
