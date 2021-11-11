import numpy as np
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':
    N = 10
    learn_rate = 1 #1e-2

    # how many transitions we do in each MC run
    MC_run_iterations = 100
    # how many MC runs we do
    MC_trial_num = 100

    v = 2

    # suggested = 1000 iterations, 100?
    # 1000, 1000 gives [-0.556  0.468  0.386  0.462], pretty good

    raw_data = np.loadtxt('in.txt',dtype=str)

    mod_size = len(raw_data[0]) # model size
    num_samples = raw_data.shape[0]

    train_data = np.zeros((num_samples,mod_size))

    def translate_type(_X):
        if _X == '+':
            return 1.0
        else:
            return -1.0

    for I in range(num_samples):
        for J in range(mod_size):
            train_data[I,J]= translate_type(raw_data[I][J])

    # this is the size of Omega
    # eg 16 for the given dataset
    state_space_size = 2 ** mod_size

    # Omega is the set of all possible inputs to the Boltzmann machine
    # ie all possible states of the system
    # each row represents a possible state
    # so all possible states can be accessed by its row number in Omega
    # eg 0 to 15 for the given dataset
    Omega = np.zeros((state_space_size, mod_size))

    distribution = np.zeros(state_space_size,dtype=int)
    train_prob = np.zeros(state_space_size)

    def edit_one_column(_J):
        cycle_length = 2**(mod_size-_J)
        half_cycle_length = 2**(mod_size-_J-1)
        num_cycles = 2**_J
        for I0 in range(num_cycles):
            for I1 in range(half_cycle_length):
                Omega[I0*cycle_length+I1,_J] = -1.0
            for I1 in range(half_cycle_length):
                Omega[I0*cycle_length+half_cycle_length+I1, _J] = 1.0

    def initialize_Omega():
        for _I in range(mod_size):
            edit_one_column(_I)

    def con_to_digits(input):
        if input>0:
            return 1
        else:
            return 0

    def initialize_distribution():
        for _I in range(num_samples):
            address = 0
            for _J in range(mod_size):
                address += con_to_digits(train_data[_I,mod_size-1-_J])*(2**_J)
            distribution[address] += 1

    initialize_Omega()
    initialize_distribution()
    train_prob = distribution/num_samples
    model_prob = np.zeros(state_space_size)

    # weights = np.array([-0.97482081,0.89636695,0.86760752,0.88467344])
    weights = np.zeros(mod_size)

    # weights = np.random.uniform(low=-1.0,high=1.0,size=mod_size)

    pos_phase = np.zeros(mod_size)
    neg_phase = np.zeros(mod_size)

    def calc_pos_phase(idx):
        corr = 0
        if idx == mod_size-1:
            for I in range(num_samples):
                corr += train_data[I,idx]*train_data[I,0]
        else:
            for I in range(num_samples):
                corr += train_data[I,idx]*train_data[I,idx+1]

        return corr/num_samples

    # calculate positive phase
    for J in range(mod_size):
        pos_phase[J] = calc_pos_phase(J)

    def energy(state):
        E = 0
        for _I in range(mod_size-1):
            E -= weights[_I]*state[_I]*state[_I+1]

        E -= weights[mod_size-1]*state[mod_size-1]*state[0]
        return E

    def prob(idx):
        return np.exp(-energy(Omega[idx,:]))

    def update_model_prob():
        global model_prob
        part_fcn = 0
        for _J in range(state_space_size):
            unnorm_prob = prob(_J)
            model_prob[_J] = unnorm_prob
            part_fcn += unnorm_prob

        model_prob = model_prob/part_fcn


    def calc_neg_phase(index):
        corr = 0
        for I in range(state_space_size):
            corr += Omega[I, index] * Omega[I, (index+1) % mod_size] * model_prob[I]

        return corr

    # Metropolis-Hastings
    def next_state(x):
        y_index = random.sample(range(state_space_size), k=1)[0]
        y = Omega[y_index,:]
        E_x = energy(x)
        E_y = energy(y)
        if E_y < E_x:
            return y
        else:
            random_num = np.random.uniform(low=0.0, high=1.0)
            if random_num < np.exp(E_x-E_y):
                return y
            else:
                return x

    # running Monte-Carlo until
    def MC_run():
        initial_index = random.sample(range(state_space_size), k=1)[0]
        current_state = Omega[initial_index,:]
        for L in range(MC_run_iterations):
            current_state = next_state(current_state)
            # print(L, current_state)
        return current_state

    def MC_neg_phase():
        tmp_neg_phase = np.zeros(mod_size)
        for I in range(MC_trial_num):
            MC_state = MC_run()
            for P in range(mod_size):
                tmp_neg_phase[P] += MC_state[P] * MC_state[(P+1) % mod_size]
            if v == 2:
                address = 0
                for _J in range(mod_size):
                    address += con_to_digits(MC_state[mod_size - 1 - _J]) * (2 ** _J)
                MC_distribution[address] += 1
            # if I % 100 == 0:
            #     print(I)

        return tmp_neg_phase/MC_trial_num


    def update_neg_phase():
        update_model_prob()
        for J in range(mod_size):
            neg_phase[J] = calc_neg_phase(J)

    def training_loop():
        global weights
        update_neg_phase()
        # print(f"pos_phase = {pos_phase}")
        NEG = MC_neg_phase()
        # print(f"neg_phase = {NEG}")
        step = pos_phase-NEG
        weights = weights + step*learn_rate

    def KL_div():
        sum = 0
        for S in range(state_space_size):
            sum += train_prob[S]*np.log(train_prob[S]/MC_prob[S])
        return sum

    def correct_KL_div():
        update_model_prob()
        sum = 0
        for S in range(state_space_size):
            sum += train_prob[S]*np.log(train_prob[S]/model_prob[S])
        return sum




    # print(pos_phase)
    # print(np.array([-0.53999949,  0.4659995,   0.43599949,  0.4539995]))
    # update_neg_phase()
    # print(neg_phase)
    # print(calc_MC_neg_phase())

    # add regularizer?

    weight1_history = [weights[0]]
    weight2_history = [weights[1]]
    weight3_history = [weights[2]]
    weight4_history = [weights[3]]
    for L in range(N):

        MC_distribution = np.zeros(state_space_size, dtype=int)
        training_loop()
        MC_prob = MC_distribution / MC_trial_num

        weight1_history.append(weights[0])
        weight2_history.append(weights[1])
        weight3_history.append(weights[2])
        weight4_history.append(weights[3])
        # print(f"div = {KL_div()}   correct div = {correct_KL_div()}")
        # print(model_prob)
        # print(MC_prob)


        if L % 10 == 0:
            print(f"N = {L}   {weights}")
            # print()

    # print(f"N = {N-1}   {weights}")  # div = {KL_div}
    # print(f"pos_phase = {pos_phase}")
    # print(f"neg_phase = {neg_phase}")
    plt.plot(weight1_history, label = 'weight1')
    plt.plot(weight2_history, label='weight2')
    plt.plot(weight3_history, label='weight3')
    plt.plot(weight4_history, label='weight4')
    plt.legend(loc="best")
    plt.show()
