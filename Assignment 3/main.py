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
    NUM_plot = args.n_tests


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

    dt = T / solution_steps

    # scale factors
    training_factor = data['scale factors']['training grid scale factor']
    scale_factor = data['scale factors']['plot scale factor']

    lower_bound = IC_lower_bound*scale_factor
    upper_bound = IC_upper_bound*scale_factor
    train_lb = IC_lower_bound*training_factor
    train_ub = IC_upper_bound*training_factor

    # grid sizes
    plot_grid_size = data['grid sizes']['plot grid size']
    training_grid_size = data['grid sizes']['training grid size']

    num_train = training_grid_size**2

    # optim parameters
    N = data['optim']['number of epochs']
    learning_rate = data['optim']['learning rate']
    batch_size = data['optim']['batch size']

    # model parameters
    hidden_layer_width1 = data['model']['hidden layer 1 width']
    hidden_layer_width2 = data['model']['hidden layer 2 width']
    hidden_layer_width3 = data['model']['hidden layer 3 width']

    # testing parameters
    num_test = data['testing']['number of points tested']

    # given a point on the vector field, the neural network tells us what the next point should be
    # ie x0 + u*dt, y0 + v*dt
    # however for numerical stability, we do not use the neural network to output this directly
    # since x0 + u*dt is very close to x0
    # instead we focus on dx = u*dt
    # in fact since u*dt is very small, we focus on u
    # we scale the output of the neural network by dt
    # instead of using the neural network to predict u*dt directly
    def train_next_step(x0,y0):
        return np.float32(u(x0,y0)), np.float32(v(x0,y0))

    # create a list of ICs and their corresponding "next point" for use in training
    IC_set = np.zeros((num_train, 2), dtype=np.float32)
    out_set = np.zeros((num_train, 2), dtype=np.float32)

    counter = 0
    for J in np.linspace(train_lb, train_ub, training_grid_size):
        for K in np.linspace(train_lb, train_ub, training_grid_size):
            IC_set[counter, 0] = np.float32(J)
            IC_set[counter, 1] = np.float32(K)
            out_set[counter, 0], out_set[counter, 1] = train_next_step(J,K)
            counter += 1

    # if verbosity = 2, we also prepare a test dataset
    if verbosity == 2:
        _Test_set = np.random.uniform(low=IC_lower_bound, high=IC_upper_bound, size=(num_test,2))
        Test_set = np.array(_Test_set, dtype=np.float32)
        Test_out_set = np.zeros((num_test,2))
        TEST_set = torch.tensor(Test_set).to(device)
        for J in range(num_test):
            Test_out_set[J, 0], Test_out_set[J, 1] = train_next_step(Test_set[J,0],Test_set[J,1])
        TEST_out_set = torch.tensor(Test_out_set).to(device)
        print("Testing set generated\n")


    # plots vector field
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

    plot_vector_field(u, v)

    # defines training dataset
    class TrainingDataset(torch.utils.data.Dataset):

        def __init__(self):
            self.input_data = torch.tensor(IC_set).to(device)
            self.output_data = torch.tensor(out_set).to(device)

        def __len__(self):
            return num_train

        def __getitem__(self, idx):
            return self.input_data[idx,:], self.output_data[idx,:]

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
                nn.Linear(hidden_layer_width3, 2),
            )

        def forward(self, ICs):
            return self.linear_relu_stack(ICs)

    # initialize the training dataset and dataloader
    train_dataset = TrainingDataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # initializing the model, loss function, and optimizer
    model = NeuralNetwork().to(device)
    loss_fcn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # we only calculate the test loss if verbosity = 2
    if verbosity == 2:
        def test_loss():
            Loss = 0
            for K in range(num_test):
                predicted_step = model(TEST_set[K,:])
                Loss += loss_fcn(predicted_step, TEST_out_set[K,:])
            return Loss/num_test

    # defines training loop
    def train_loop(_epoch): 
        for batch, (_x, _y) in enumerate(train_dataloader):
            predicted_step = model(_x)
            loss = loss_fcn(predicted_step, _y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbosity == 2:
                if batch == 0:
                    print(f"epoch {epoch + 1}   training loss = {loss}   testing loss = {test_loss()}")
            else:
                if _epoch % 5 == 0:
                    if batch == 0:
                        print(f"epoch {epoch + 1}   training loss = {loss}")


    # training
    for epoch in range(N):
        train_loop(epoch)
    print("Training complete")


    # generates test plot
    def testing(c):
        test_IC = np.random.uniform(low=IC_lower_bound, high=IC_upper_bound, size=2)
        plot_x = [test_IC[0]]
        plot_y = [test_IC[1]]
        test_input = torch.tensor(np.array(test_IC, dtype=np.float32)).to(device)

        for I in range(solution_steps):
            velocity = model(test_input)
            test_output = test_input + velocity*dt
            np_add_tuple = test_output.cpu().detach().numpy()
            plot_x.append(np_add_tuple[0])
            plot_y.append(np_add_tuple[1])
            test_input = test_output

        plt.plot(plot_x, plot_y, color=c)
        plt.plot(test_IC[0], test_IC[1], marker='o', color=c)


    color_code = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # different colors for different test solutions
    for J in range(NUM_plot):
        testing(color_code[J % 7])


    name = str(N)
    plt.title("Neural DE solutions, " + name + " epochs")
    plt.savefig(res_path + '.pdf')
    print()
    print(f"Test plot generated")
    plt.show()
