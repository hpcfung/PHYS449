import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import json, argparse, sys

if __name__ == '__main__':
    device = torch.device("cpu")

    # Command line arguments
    parser = argparse.ArgumentParser(description='ODE Solver')
    parser.add_argument('--param', type=str, default='param/param.json',
                        metavar='param.json', help='file name for json attributes')
    parser.add_argument('-v', type=int, default=1,
                        metavar='N', help='verbosity (default: 1)')
    parser.add_argument('--res-path', type=str, default='plots',
                        metavar='results', help='path to save the test plots at')
    parser.add_argument('--x-field', type=str, default='-y/np.sqrt(x**2 + y**2)',
                        metavar='x**2', help='expression of the x-component of the vector field')
    parser.add_argument('--y-field', type=str, default='x/np.sqrt(x**2 + y**2)',
                        metavar='y**2', help='expression of the y-component of the vector field')
    parser.add_argument('--lb', type=float, default=-1.0,
                        metavar='LB', help='lower bound for initial conditions')
    parser.add_argument('--ub', type=float, default=1.0,
                        metavar='UB', help='upper bound for initial conditions')
    parser.add_argument('--n-tests', type=int, default=3,
                        metavar='N_TESTS', help='number of test trajectories to plot')

    args = parser.parse_args()

    if args.v > 2 or args.v < 1:
        print(f"Warning, verbosity can only be 0 or 1, entered {args.v}")
        print("Program exited")
        sys.exit()

    verbosity = args.v
    param_path = args.param

    _x_field = args.x_field
    _y_field = args.y_field
    u = lambda x, y: eval(_x_field)
    v = lambda x, y: eval(_y_field)

    IC_lower_bound = args.lb
    IC_upper_bound = args.ub

    res_path = args.res_path
    NUM_test = args.n_tests

    # read json file
    g = open(param_path)
    data = json.load(g)

    if verbosity == 2:
        print(f"json parameters")
        print(f"ode: {data['ode']}")
        print(f"scale factors: {data['scale factors']}")
        print(f"grid sizes: {data['grid sizes']}")
        print(f"optim: {data['optim']}")
        print(f"model: {data['model']}")
        print()

    # ode parameters
    T = data['ode']['T']
    solution_steps = data['ode']['solution steps']
    step_size_in_training_data = data['ode']['step size in training data']

    training_sol_steps = int(solution_steps/step_size_in_training_data)
    time_step = T / solution_steps

    # scale factors
    IC_factor = data['scale factors']['IC bounds scale factor']
    scale_factor = data['scale factors']['plot scale factor']

    lower_bound = IC_lower_bound*scale_factor
    upper_bound = IC_upper_bound*scale_factor

    # grid sizes
    plot_grid_size = data['grid sizes']['plot grid size']
    IC_grid_size = data['grid sizes']['IC grid size']

    num_sol = IC_grid_size**2

    # optim parameters
    N = data['optim']['number of epochs'] 
    learning_rate = data['optim']['learning rate']  

    # model parameters
    hidden_layer_width1 = data['model']['hidden layer 1 width']
    hidden_layer_width2 = data['model']['hidden layer 2 width']
    hidden_layer_width3 = data['model']['hidden layer 3 width']

    # testing parameters
    num_test = data['testing']['ICs tested']
    if verbosity==2:
        TEST_IC = np.random.uniform(low=IC_lower_bound, high=IC_upper_bound, size=(num_test,2))
        TEST_input = torch.tensor(np.array(TEST_IC, dtype=np.float32)).to(device)

    # initialize solutions and initial conditions arrays
    sol_set = np.zeros((num_sol,training_sol_steps,2), dtype=np.float32)
    IC_set = np.zeros((num_sol,2), dtype=np.float32)

    # if verbosity = 2, we also calculate the test error
    # initialize test solutions
    if verbosity==2:
        Test_set = np.zeros((num_test,training_sol_steps,2), dtype=np.float32)

    # plots vector field and prepares vector field for generating the training set
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

    # one iteration of 4th order Runga-Kutta for generating the training set
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

    # generates training solution for the given initial condition
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
                sol_set[index, tmp, 0] = np.float32(new_x_rk4)
                sol_set[index, tmp, 1] = np.float32(new_y_rk4)
                tmp+=1

    # generates testing solution
    def gen_test(u, v, init_x, init_y, index):
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
                Test_set[index, tmp, 0] = np.float32(new_x_rk4)
                Test_set[index, tmp, 1] = np.float32(new_y_rk4)
                tmp+=1


    # generates training set
    plot_vector_field(u, v)
    counter = 0
    for J in np.linspace(IC_lower_bound*IC_factor, IC_upper_bound*IC_factor, IC_grid_size):
        for K in np.linspace(IC_lower_bound*IC_factor, IC_upper_bound*IC_factor, IC_grid_size):
            IC_set[counter, 0] = np.float32(J)
            IC_set[counter, 1] = np.float32(K)
            solve_DE(u, v, J, K, counter)
            counter += 1
        print(f"{counter} training sequences generated")

    print("Training set generated\n")

    # if verbosity = 2, this generates the testing set
    if verbosity == 2:
        TEST_set = torch.tensor(Test_set).to(device)
        for J in range(num_test):
            gen_test(u, v, TEST_IC[J,0], TEST_IC[J,1], J)
        print("Testing set generated\n")

    # convert the array into a tensor
    training_sol = torch.tensor(sol_set).to(device)
    training_IC = torch.tensor(IC_set).to(device)

    # defines neural network used
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


    #initializing the model, loss function, and optimizer
    model = NeuralNetwork().to(device)
    loss_fcn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # if verbosity = 2, we also calculate the test error
    if verbosity==2:
        def test_loss():
            loss = 0
            for J in range(num_test):
                output_seq = model(TEST_input[J, :])
                loss += loss_fcn(output_seq, TEST_set[J, :, :])
            return loss / num_test

    # training loop
    for epoch in range(N):
        optimizer.zero_grad()

        loss = 0
        for J in range(num_sol):
            output_seq = model(training_IC[J,:])
            loss += loss_fcn(output_seq, training_sol[J,:,:])
        loss = loss/num_sol
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == (N-1):
            if verbosity == 2:
                print(f"epoch = {epoch}   training loss = {loss}   testing loss = {test_loss()}")
            else:
                print(f"epoch = {epoch}   training loss = {loss}")

    print("Training complete")

    # generates test plot
    def testing(c):
        test_IC = np.random.uniform(low=IC_lower_bound, high=IC_upper_bound, size=2)
        test_input = torch.tensor(np.array(test_IC, dtype = np.float32)).to(device)
        output_seq = model(test_input)

        output_seq_cpu = output_seq.cpu()
        output_SEQ = output_seq_cpu.detach().numpy()

        output_x = np.insert(output_SEQ[:,0],0,test_IC[0]) #add the IC to the plot
        output_y = np.insert(output_SEQ[:,1],0,test_IC[1])

        plt.plot(output_x, output_y, linestyle='-', marker='o', markersize=2, color = c)
        plt.plot(test_IC[0],test_IC[1], marker='o', color = c)

    color_code = ['b','g','r','c','m','y','k'] # different colors for different test solutions
    for J in range(NUM_test):
        testing(color_code[J%7])

    name = str(N)
    plt.title("Neural DE solutions, " + name + " epochs")
    plt.savefig(res_path+'.pdf')

    plt.show()
