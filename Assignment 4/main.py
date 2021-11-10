import numpy as np

if __name__ == '__main__':
    N = 100
    learn_rate = 1 #1e-2

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

    state_space_size = 2 ** mod_size
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
    state_space_prob = np.zeros(state_space_size)

    # weights = np.array([-0.5,0.5,0.5,0.5])
    #weights = np.zeros(mod_size)
    weights = np.random.uniform(low=-1.0,high=1.0,size=mod_size)

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

    def update_state_space_prob():
        global state_space_prob
        part_fcn = 0
        for _J in range(state_space_size):
            unnorm_prob = prob(_J)
            state_space_prob[_J] = unnorm_prob
            part_fcn += unnorm_prob

        state_space_prob = state_space_prob/part_fcn


    def calc_neg_phase(index):
        corr = 0
        if index == mod_size - 1:
            for I in range(state_space_size):
                corr += Omega[I,index]*Omega[I,0]*state_space_prob[I]
        else:
            for I in range(state_space_size):
                corr += Omega[I,index]*Omega[I,index+1]*state_space_prob[I]

        return corr

    def update_neg_phase():
        update_state_space_prob()
        for J in range(mod_size):
            neg_phase[J] = calc_neg_phase(J)

    def training_loop():
        global weights
        update_neg_phase()
        step = pos_phase-neg_phase
        weights = weights + step*learn_rate

    # def KL_div():
    #     sum = 0
    #     for S in range(state_space_size):




    def alt_calc_pos_phase(index):
        corr = 0
        if index == mod_size - 1:
            for I in range(state_space_size):
                corr += Omega[I,index]*Omega[I,0]*train_prob[I]
        else:
            for I in range(state_space_size):
                corr += Omega[I,index]*Omega[I,index+1]*train_prob[I]

        return corr

    alt_pos_phase = np.zeros(mod_size)
    for alpha in range(mod_size):
        alt_pos_phase[alpha]=alt_calc_pos_phase(alpha)

    for L in range(N):
        training_loop()
        if L % 20 == 0:
            print(f"N = {L}   {weights}")  #div = {KL_div}
            print(f"pos_phase = {pos_phase}")
            print(f"alt_pos_phase = {alt_pos_phase}")
            print(f"neg_phase = {neg_phase}")

    print(f"N = {N-1}   {weights}")  # div = {KL_div}
    print(f"pos_phase = {pos_phase}")
    print(f"alt_pos_phase = {alt_pos_phase}")
    print(f"neg_phase = {neg_phase}")
