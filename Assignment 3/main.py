import numpy as np
from pylab import plot, show
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages #remove later

import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    #np.random.seed(0)

    initial_x = 0.5
    initial_y = 0.6
    T = 7
    solution_steps = 500
    step_size_in_training_data = 25
    training_sol_steps = 20 #when mmultiplied, these 2 should give sol steps

    IC_lower_bound = -1
    IC_upper_bound = 1

    IC_factor = 2
    scale_factor = 3

    lower_bound = IC_lower_bound*scale_factor
    upper_bound = IC_upper_bound*scale_factor
    plot_grid_size = 20
    IC_grid_size = 8

    num_sol = IC_grid_size**2

    num_DE = 4

    time_step = T / solution_steps

    pp = PdfPages('MLP_DE_solver.pdf')

    sol_set = np.zeros((num_sol,training_sol_steps,2), dtype=np.float32)
    IC_set = np.zeros((num_sol,2), dtype=np.float32)

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
        return np.sin(4*y)/2+0.05


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
        return -x-x**3-0.5*y

    def u5(x, y):
        return np.sin(np.pi*x) + np.sin(np.pi*y)

    def v5(x, y):
        return np.cos(np.pi*y)


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

    def solve_DE(u, v, init_x, init_y, index):
        x_points_rk4 = [init_x]
        y_points_rk4 = [init_y]

        tmp = 0
        for k in range(solution_steps+1):
            last_x_rk4 = x_points_rk4[k]
            last_y_rk4 = y_points_rk4[k]
            new_x_rk4, new_y_rk4 = rk4(last_x_rk4, last_y_rk4, time_step, u, v)
            x_points_rk4.append(new_x_rk4)
            y_points_rk4.append(new_y_rk4)

            if k%step_size_in_training_data==0 and k!=0:
                #print(tmp)
                sol_set[index, tmp, 0] = np.float32(new_x_rk4)
                sol_set[index, tmp, 1] = np.float32(new_y_rk4)
                tmp+=1

        #plot(x_points_rk4, y_points_rk4, linestyle='--')
        #plot(init_x,init_y, marker='o', color='b')
        #show()

    #prepare dataset
    u_list = [u1, u2, u3, u4, u0, u5]
    v_list = [v1, v2, v3, v4, v0, v5]

    I = 4
    u = u_list[I]
    v = v_list[I]
    plot_vector_field(u, v)
    counter = 0
    for J in np.linspace(IC_lower_bound*IC_factor, IC_upper_bound*IC_factor, IC_grid_size):
        for K in np.linspace(IC_lower_bound*IC_factor, IC_upper_bound*IC_factor, IC_grid_size):
            #print(f"{counter}: ({J},{K})")
            IC_set[counter, 0] = np.float32(J)
            IC_set[counter, 1] = np.float32(K)
            solve_DE(u, v, J, K, counter)
            counter += 1

    print("training set generated")

    device = torch.device("cpu")
    training_sol = torch.tensor(sol_set).to(device)
    training_IC = torch.tensor(IC_set).to(device)

    #training_input = np.zeros((training_sol_steps,1,1), dtype = np.float32)
    #training_input_tensor = torch.tensor(training_input).to(device)

    N = 2000 #1000
    learning_rate = 1e-3 #2e-3
    beta = (0.9,0.99)
    hidden_layer_width1 = 100
    hidden_layer_width2 = 100
    hidden_layer_width3 = 100

    class NeuralNetwork(nn.Module):

        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(2, hidden_layer_width1),
                nn.LeakyReLU(),
                nn.Linear(hidden_layer_width1, hidden_layer_width2),
                nn.LeakyReLU(),
                nn.Linear(hidden_layer_width2, hidden_layer_width3),
                nn.LeakyReLU(),
                nn.Linear(hidden_layer_width3, training_sol_steps*2),
            )

        def forward(self, ICs):
            raw_output = self.linear_relu_stack(ICs)
            pred_x_list, pred_y_list = torch.split(raw_output, training_sol_steps)
            return torch.stack([pred_x_list,pred_y_list],dim=-1)


    model = NeuralNetwork().to(device)
    loss_fcn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=beta)

    for epoch in range(N):
        optimizer.zero_grad()

        loss = 0
        for J in range(num_sol):
            output_seq = model(training_IC[J,:])
            #print(output_seq.size())
            #S = training_sol[J,:,0,:]
            #print(f"output_seq size = {output_seq.size()}")
            #print(f"training_sol size = {training_sol[J,:,:].size()}")
            loss += loss_fcn(output_seq, training_sol[J,:,:])
        loss = loss/num_sol
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == (N-1):
            print(f"epoch = {epoch}   loss = {loss}")

    print("training complete")


    def testing(c):
        test_IC = np.random.uniform(low=IC_lower_bound, high=IC_upper_bound, size=2)
        test_input = torch.tensor(np.array(test_IC, dtype = np.float32)).to(device)
        output_seq = model(test_input)

        output_seq_cpu = output_seq.cpu()
        output_SEQ = output_seq_cpu.detach().numpy()

        output_x = np.insert(output_SEQ[:,0],0,test_IC[0]) #add the IC to the plot
        output_y = np.insert(output_SEQ[:,1],0,test_IC[1])

        plot(output_x, output_y, linestyle='-', marker='o', markersize=2, color = c)
        plot(test_IC[0],test_IC[1], marker='o', color = c)
        #plt.legend(loc='lower right')

    def color_code(num):
        if num%7==0:
            return 'b'
        if num%7==1:
            return 'g'
        if num%7==2:
            return 'r'
        if num%7==3:
            return 'c'
        if num%7==4:
            return 'm'
        if num%7==5:
            return 'y'
        if num%7==6:
            return 'k'

    for J in range(7): #7
        testing(color_code(J))

    name = str(N)
    plt.title("DE with " + name + " iterations")

    torch.save(model.state_dict(), name+"iterations.pt")

    pp.savefig()
    show()
    pp.close()

#problem: sol does not start at IC
