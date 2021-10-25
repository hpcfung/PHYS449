import numpy as np
from pylab import plot, show
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == '__main__':
    initial_x = 0.5
    initial_y = 0.6
    T = 10
    N = 1000

    IC_lower_bound = -1
    IC_upper_bound = 1

    scale_factor = 3

    lower_bound = IC_lower_bound*scale_factor
    upper_bound = IC_upper_bound*scale_factor
    plot_grid_size = 30

    num_DE = 5

    time_step = T / N

    pp = PdfPages('numerical_DEs.pdf')

    def u0(x, y):
        return -y / np.sqrt(x ** 2 + y ** 2)

    def v0(x, y):
        return x / np.sqrt(x ** 2 + y ** 2)


    def u1(x, y):
        return np.cos(x*y)

    def v1(x, y):
        return -np.sin(np.exp(x-y))


    def u2(x, y):
        return 1/(y**4+5)

    def v2(x, y):
        return np.sin(3*y)/2


    def u3(x, y):
        return -np.exp(-x**2)/2

    def v3(x, y):
        if y < 1e-8 and y > -1e-8:
            return -1/5
        else:
            return -np.sin(y)/(5*y)


    def u4(x, y):
        return y

    def v4(x, y):
        return -x-x**3-2*y


    def u5(x, y):
        return 2*x-2*x*y

    def v5(x, y):
        return -y+x*y

    def plot_vector_field(u, v):
        X, Y = np.meshgrid(np.linspace(lower_bound, upper_bound, plot_grid_size),
                           np.linspace(lower_bound, upper_bound, plot_grid_size))
        U, V = np.meshgrid(np.linspace(lower_bound, upper_bound, plot_grid_size),
                           np.linspace(lower_bound, upper_bound, plot_grid_size))
        for i in range(plot_grid_size):
            for j in range(plot_grid_size):
                U[i, j] = u(X[i, j], Y[i, j])
                V[i, j] = v(X[i, j], Y[i, j])

        plt.quiver(X, Y, U, V)

    def rk2(x0, y0, Delta_t, u, v):
        k1x = u(x0, y0) * Delta_t
        k1y = v(x0, y0) * Delta_t
        k2x = u(x0 + k1x / 2, y0 + k1y / 2) * Delta_t
        k2y = v(x0 + k1x / 2, y0 + k1y / 2) * Delta_t
        return x0 + k2x, y0 + k2y


    def rk4(x0, y0, Delta_t, u, v):
        k1x = u(x0, y0) * Delta_t
        k1y = v(x0, y0) * Delta_t
        k2x = u(x0 + k1x / 2, y0 + k1y / 2) * Delta_t
        k2y = v(x0 + k1x / 2, y0 + k1y / 2) * Delta_t
        k3x = u(x0 + k2x / 2, y0 + k2y / 2) * Delta_t
        k3y = v(x0 + k2x / 2, y0 + k2y / 2) * Delta_t
        k4x = u(x0 + k3x, y0 + k3y) * Delta_t
        k4y = v(x0 + k3x, y0 + k3y) * Delta_t
        return x0 + (k1x + 2 * k2x + 2 * k3x + k4x) / 6, y0 + (k1y + 2 * k2y + 2 * k3y + k4y) / 6

    # no x dep, so that term in the rk method should not matter?
    # just imagine setting output var from 1 to 2?

    def solve_DE(u, v, index):
        x_points_rk2 = [initial_x]
        y_points_rk2 = [initial_y]
        x_points_rk4 = [initial_x]
        y_points_rk4 = [initial_y]
        t_points = [0]

        t = 0
        for k in range(N):
            t = t + time_step

            last_x_rk2 = x_points_rk2[k]
            last_y_rk2 = y_points_rk2[k]
            new_x_rk2, new_y_rk2 = rk2(last_x_rk2, last_y_rk2, time_step, u, v)
            x_points_rk2.append(new_x_rk2)
            y_points_rk2.append(new_y_rk2)

            last_x_rk4 = x_points_rk4[k]
            last_y_rk4 = y_points_rk4[k]
            new_x_rk4, new_y_rk4 = rk4(last_x_rk4, last_y_rk4, time_step, u, v)
            x_points_rk4.append(new_x_rk4)
            y_points_rk4.append(new_y_rk4)

            t_points.append(t)

            if k % 100 == 0:
                label = f"t = {t}"
                plt.annotate(label,  # this is the text
                             xy=(new_x_rk2, new_y_rk2),  # these are the coordinates to position the label
                             xytext=(new_x_rk2 - 0.5, new_y_rk2 + 0.5),
                             arrowprops=dict(arrowstyle="->"))
                plt.annotate(label,  # this is the text
                             xy=(new_x_rk4, new_y_rk4),  # these are the coordinates to position the label
                             xytext=(new_x_rk4 - 0.5, new_y_rk4 + 0.5),
                             arrowprops=dict(arrowstyle="->"))

        plot_vector_field(u, v)
        plot(x_points_rk2, y_points_rk2, linestyle='-', color='g', label="rk2")
        plot(x_points_rk4, y_points_rk4, linestyle='-', color='r', label="rk4")
        plot(initial_x,initial_y, marker='o', color='b')
        name = str(index)
        plt.title("DE "+name)
        #plt.savefig('plot_of_DE_'+name+'.pdf')
        plt.legend(loc='lower right')
        pp.savefig()
        plt.clf()
        #show()

    u_list = [u1, u2, u3, u4, u5]
    v_list = [v1, v2, v3, v4, v5]

    for I in range(num_DE):
        u = u_list[I]
        v = v_list[I]
        solve_DE(u, v, I+1)

    pp.close()
