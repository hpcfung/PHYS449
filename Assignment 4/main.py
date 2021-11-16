import numpy as np
import random, argparse, json
import matplotlib.pyplot as plt

# The dataset is in terms of + and -, we convert it into +1 and -1
def translate_type(_X):
    if _X == '+':
        return 1.0
    else:
        return -1.0

# Omega is the set of all possible inputs to the Boltzmann machine
# ie all possible states of the system
# each row represents a possible state
# so all possible states can be accessed by its row number in Omega
# eg 0 to 15 for the given dataset
# Omega is initialized using the following algorithm
# which is based on the enumeration of binary numbers
def edit_one_column(_J):
    cycle_length = 2 ** (mod_size - _J)
    half_cycle_length = 2 ** (mod_size - _J - 1)
    num_cycles = 2 ** _J
    for I0 in range(num_cycles):
        for I1 in range(half_cycle_length):
            Omega[I0 * cycle_length + I1, _J] = -1.0
        for I1 in range(half_cycle_length):
            Omega[I0 * cycle_length + half_cycle_length + I1, _J] = 1.0

def initialize_Omega():
    for _I in range(mod_size):
        edit_one_column(_I)

# to create the distribution for the training dataset
# the program needs to read the training data and add one to the corresponding entry in Omega
# the following algorithm converts an array of +1 and -1 to its corresponding index in Omega
def con_to_digits(input):
    if input > 0:
        return 1
    else:
        return 0

# the distribution is basically a histogram
def initialize_distribution():
    for _I in range(num_samples):
        address = 0
        for _J in range(mod_size):
            address += con_to_digits(train_data[_I, mod_size - 1 - _J]) * (2 ** _J)
        distribution[address] += 1

# calculates the ppositive phase by going over the training data
def calc_pos_phase(idx):
    corr = 0
    if idx == mod_size - 1:
        for I in range(num_samples):
            corr += train_data[I, idx] * train_data[I, 0]
    else:
        for I in range(num_samples):
            corr += train_data[I, idx] * train_data[I, idx + 1]

    return corr / num_samples

# calculates the energy for a given state (an array of +1 and -1)
def energy(state):
    E = 0
    for _I in range(mod_size - 1):
        E -= weights[_I] * state[_I] * state[_I + 1]

    E -= weights[mod_size - 1] * state[mod_size - 1] * state[0]
    return E

# Metropolis-Hastings Monte-Carlo Markov Chain
def next_state(x):
    y_index = random.sample(range(state_space_size), k=1)[0]
    y = Omega[y_index, :]
    E_x = energy(x)
    E_y = energy(y)
    if E_y < E_x:
        return y
    else:
        random_num = np.random.uniform(low=0.0, high=1.0)
        if random_num < np.exp(E_x - E_y):
            return y
        else:
            return x

# runs one Monte-Carlo simulation
def MC_run():
    initial_index = random.sample(range(state_space_size), k=1)[0]
    current_state = Omega[initial_index, :]
    for L in range(MC_run_iterations):
        current_state = next_state(current_state)
        # print(L, current_state)
    return current_state

# by performing many Monte-Carlo simulations
# this calculates the negative phase (the expectation value)
# in verbose mode, it also creates a distribution of p_lambda from the Monte-Carlo simulations
def MC_neg_phase():
    tmp_neg_phase = np.zeros(mod_size)
    for I in range(MC_trial_num):
        MC_state = MC_run()
        for P in range(mod_size):
            tmp_neg_phase[P] += MC_state[P] * MC_state[(P + 1) % mod_size]
        if v == 2:
            address = 0
            for _J in range(mod_size):
                address += con_to_digits(MC_state[mod_size - 1 - _J]) * (2 ** _J)
            MC_distribution[address] += 1

    return tmp_neg_phase / MC_trial_num

# one iteration of gradient descent
def training_loop():
    global weights
    NEG = MC_neg_phase()
    step = pos_phase - NEG
    weights = weights + step * learn_rate - 2*regularizer*weights

# calculates the Kullback-Leibler divergence
# def KL_div():
#     # Hp is the entropy of p_D, the dataset probability distribution
#     Hp = 0
#     # cross-entropy of p_D and p_lambda
#     cross_entropy = 0
#     for S in range(state_space_size):
#         Hp -= train_prob[S]*np.log(train_prob[S])
#         if MC_prob[S] > 0:
#             cross_entropy -= train_prob[S] * np.log(MC_prob[S])
#     return cross_entropy-Hp
def KL_div():
    div = 0
    for S in range(state_space_size):
        # when MC_prob[S] = 0, this expression becomes undefined
        # however as an approximation, we can ignore this term
        # as explained in README.md
        if MC_prob[S] > 0:
            div += train_prob[S]*np.log(train_prob[S]/MC_prob[S])
    return div


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(
        description='Implement and train a fully visible Boltzmann machine on'
                    'data gathered from a 1-D classical Ising chain and use it'
                    ' to predict the model couplers.'
                    ' The hyperparameters used are in the param.json file, and consist of the following: '
                    'Number of gd iterations: number of iterations of gradient descent during training; '
                    'learning rate: scale factor for each step of the gradient descent; '
                    'Number of iterations per MC simulation: number of steps in each Monte-Carlo simulation; '
                    'Number of MC simulations: total number of Monte-Carlo simulations performed; '
                    'verbosity: when the verbosity is 2 (the value in param.json), the program tracks '
                    'the KL divergence of the training dataset with respect to your generative model during '
                    'the training and save a plot of its values versus training epochs')
    parser.add_argument('Data_path', type=str, help='path to the training data')

    args = parser.parse_args()
    data_path = args.Data_path

    # read json file
    data = json.load(open('param.json'))

    N = data['Number of gd iterations']
    learn_rate = data['learning rate']
    regularizer = data['regularizer']

    # how many transitions we do in each MC simulation
    MC_run_iterations = data['Number of iterations per MC simulation']
    # how many MC simulations we do
    MC_trial_num = data['Number of MC simulations']

    v = data['verbosity']

    # loads training data
    raw_data = np.loadtxt(data_path,dtype=str)

    # model size
    mod_size = len(raw_data[0])
    # number of samples in the training dataset
    num_samples = raw_data.shape[0]

    # initialize training data
    train_data = np.zeros((num_samples,mod_size))

    for I in range(num_samples):
        for J in range(mod_size):
            train_data[I,J] = translate_type(raw_data[I][J])

    # this is the size of Omega
    # eg 16 for the given dataset
    state_space_size = 2 ** mod_size

    # initializing Omega and the training dataset distribution and probabilities
    Omega = np.zeros((state_space_size, mod_size))

    distribution = np.zeros(state_space_size,dtype=int)
    train_prob = np.zeros(state_space_size)

    initialize_Omega()
    initialize_distribution()
    # training dataset probabilities
    train_prob = distribution/num_samples

    # initializing the weights of the model
    weights = np.random.uniform(low=-1.0,high=1.0,size=mod_size)

    # initializing the positve phase and negative phase
    pos_phase = np.zeros(mod_size)
    neg_phase = np.zeros(mod_size)

    # calculate positive phase
    for J in range(mod_size):
        pos_phase[J] = calc_pos_phase(J)

    # in verbose mode, tracks the KL divergence
    if v == 2:
        KL_history = []

    # training
    print("Training starts")
    couplers = {}
    epochs = []
    for epoch in range(N):
        epochs.append(epoch+1)
        if v == 2:
            MC_distribution = np.zeros(state_space_size, dtype=int)

        training_loop()

        for _idx in range(mod_size):
            couplers[(_idx,_idx+1%mod_size)] = weights[_idx]

        if v == 2:
            # calculates p_lambda using the distribution obtained from the Monte-Carlo simulations
            MC_prob = MC_distribution / MC_trial_num
            KL = KL_div()
            KL_history.append(KL)
            if epoch % 10 == 0:
                print(f"epoch = {epoch + 1}   couplers = {couplers}   KL div = {KL}")
        else:
            if epoch % 10 == 0:
                print(f"epoch = {epoch + 1}   couplers = {couplers}")


    print()
    print("Final coupler values")
    print(couplers)
    print()
    plt.plot(epochs,KL_history)
    plt.title("KL divergence of the training dataset wrt model during training")
    plt.xlabel('Training epochs')
    plt.ylabel('KL divergence')
    plt.savefig('KL_div_during_training.pdf')
    print("Program completed, plot saved as KL_div_during_training.pdf")
