##Pattern Recognition: Implementation of the "forward part" of forward/backward
##algorithm. Reports observation probability of input sequence.      

import sys

# -------------------------------
# HMM CLASS
# -------------------------------
class HMM:
    def __init__(self):
        pass

    # Reads HMM values from a given file
    def read_hmm_from_file(self, file_name):
        fmm_fd = open(file_name, "r")
        # N - number of states
        # M - number of observation symbols
        # T - number of time steps or length of oberservation sequences
        self.N, self.M, self.T = [int(i) for i in fmm_fd.readline().rstrip().split(" ")]

        # Read state from the 2nd line
        self.states = fmm_fd.readline().rstrip().split(" ")
        # Read vocab from the 3rd line
        self.vocab = fmm_fd.readline().rstrip().split(" ")
        self.vocab_index = {}
        for i in range(len(self.vocab)):
            self.vocab_index[self.vocab[i]] = i

        self.a = []
        B_by_lines = []

        if (fmm_fd.readline().rstrip() == "a:"):
            for i in range(0, self.N):
                self.a.append([float(i) for i in fmm_fd.readline().rstrip().split(" ")])
        if (fmm_fd.readline().rstrip() == "b:"):      
            for i in range(0, self.N):
                B_by_lines.append([float(i) for i in fmm_fd.readline().rstrip().split(" ")])

        self.b = {}
        for i in range(0, self.M):
            tmp_arr = []
            for j in range(0, self.N):
                tmp_arr.append(B_by_lines[j][i])
            self.b[self.vocab[i]] = tmp_arr
        #self.b or obsProb is [observations or M]x[states or N]

        if (fmm_fd.readline().rstrip() == "pi:"):      
            self.pi = [float(i) for i in fmm_fd.readline().rstrip().split(" ")]

def forward(hmm, T, obs):
    a = hmm.a
    b = hmm.b
    pi = hmm.pi
    N = hmm.N
    M = hmm.M

    #(1) : Initialization
    alpha = [[0 for item2 in range(N)] for item in range(T)]
    for state_num in range(N):
        alpha[0][state_num] = pi[state_num] * b[obs[0]][state_num]
                    
    #(2): Induction
    for i in range(1,T):
        for current_state in range(N):
            alpha[i][current_state] = sum(alpha[i-1][prev_state]*a[prev_state][current_state] for prev_state in range(N))* b[obs[i]][current_state]

    return alpha
    #(3): Termination
    #prob = sum(alpha[T-1][states] for states in range(N))
    #return prob

def forward2(STM, obsProb, initial, obs_seq, hmm):
    vocab_index = hmm.vocab_index
    N = hmm.N
    T = len(obs_seq)
    alpha = [[0 for item2 in range(N)] for item in range(T)]
    for state_num in range(N):
        alpha[0][state_num] = initial[state_num] * obsProb[state_num][vocab_index[obs_seq[0]]]
    for i in range(1,T):
        for current_state in range(N):
            alpha[i][current_state] = sum(alpha[i-1][prev_state]*STM[prev_state][current_state] for prev_state in range(N))* obsProb[current_state][vocab_index[obs_seq[i]]]

    return alpha

def backward(hmm, T, obs_seq):
    STM = hmm.a
    obsProb = hmm.b
    initial = hmm.pi
    N = hmm.N

    beta = [[0.0 for state in range(N)] for obs in range(T)]
    for t in range(T-1, -1, -1):
        for stateI in range(N):
            if t == T-1:
                beta[t][stateI] = 1.0
            else:
                for stateJ in range(N):
                    beta_val = beta[t+1][stateJ]
                    a_val = STM[stateI][stateJ]
                    b_val = obsProb[obs_seq[t+1]][stateJ]
                    beta[t][stateI] += beta_val*a_val*b_val
    return beta

def gammaFunc(hmm, alpha, beta):
    N = hmm.N
    gamma = [[0.0 for state in range(N)] for t in range(T)]
    for t in range(T):
        denominator = 0.0
        for stateJ in range(N):
            denominator += alpha[t][stateJ]*beta[t][stateJ]
        for stateI in range(N):
            gamma[t][stateI] = float(alpha[t][stateI]*beta[t][stateI]/denominator)
    return gamma


def epsi(hmm, alpha, beta, obs_seq, old_prob):
    STM = hmm.a
    obsProb = hmm.b
    N = hmm.N
    T = len(obs_seq)

    epsilon = [[[0.0 for j in range(N)] for i in range(N)] for t in range(T)]

    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                epsilon[t][i][j] = float(alpha[t][i]*STM[i][j]*beta[t+1][j]*obsProb[obs_seq[t+1]][j]/old_prob)
    return epsilon
# -------------------------------
# MAIN PROGRAM
# -------------------------------

if __name__ == '__main__':

    # Get names of files
    hmm_fname = sys.argv[1] # Original HMM
    #hmm_fname = "sentence.hmm"
    obser_fname = sys.argv[2] # Observations
    #obser_fname = "example2.obs"
    optimized_fname = sys.argv[3] # Output HMM

    # Initialize new HMM and read it from file
    hmm = HMM()
    hmm.read_hmm_from_file(hmm_fname)

    # Reading observation file
    obser_fd = open(obser_fname, "r")
    num_of_obser_sets = int(obser_fd.readline().rstrip())
    observations = []

    for i in range(num_of_obser_sets):
        observation = {}
        observation["T"] = int(obser_fd.readline().rstrip())
        observation["seq"] = obser_fd.readline().rstrip().split(" ")
        observations.append(observation)

    updated = [[None for params in range(4)] for obs in range(len(observations))]

    counter = 0
    #implement forward algorithm for each observation sequence in the obs file
    for observation in observations:#since number of iterations = 1
        N = hmm.N
        obs_seq = observation["seq"]
        T = len(obs_seq)
        alpha = forward(hmm, observation["T"],obs_seq)
        #print("Alpha {0}".format(alpha))
        old_prob = sum(alpha[T-1][states] for states in range(N))
        #print(old_prob)
        beta = backward(hmm, observation["T"], obs_seq)
        #print("Beta {0}".format(beta))
        epsilon = epsi(hmm, alpha, beta, obs_seq, old_prob)
        #print("Epsilon {0}".format(epsilon))
        gamma = gammaFunc(hmm, alpha, beta)
        #print("Gamma {0}".format(gamma))
        new_pi = gamma[0]
        #print("Pi-Updated: {0}".format(new_pi))
        new_STM = [[0.0 for j in range(N)] for i in range(N)]
        for stateI in range(N):
            denom_gamma = 0.0
            for t in range(T-1):
                denom_gamma += gamma[t][stateI]
            if denom_gamma == 0.0:
                for stateJ in range(N):
                    new_STM[stateI][stateJ] = hmm.a[stateI][stateJ]
            else:
                for stateJ in range(N):
                    for t in range(T-1):
                        new_STM[stateI][stateJ] += epsilon[t][stateI][stateJ]
                    new_STM[stateI][stateJ] /= denom_gamma
                    if new_STM[stateI][stateJ] > 0.99:
                        new_STM[stateI][stateJ] = 1.0
        #print("A-Updated: {0}".format(new_STM))
        new_obsProb = [[0.0 for obs in range(hmm.M)] for state in range(N)]
        for stateI in range(N):
            denom_gamma = 0.0
            for t in range(T):
                denom_gamma += gamma[t][stateI]
            if denom_gamma == 0.0:
                for t in range(hmm.M):
                    new_obsProb[stateI][t] = hmm.b[hmm.vocab[t]][stateI]
            else:
                for k in range(hmm.M):
                    for t_inner in range(T):
                        if hmm.vocab_index[obs_seq[t_inner]] == k:
                            new_obsProb[stateI][k] += gamma[t_inner][stateI]
                    new_obsProb[stateI][k] /= denom_gamma
        #print("B-Updated: {0}".format(new_obsProb))
        new_alpha = forward2(new_STM, new_obsProb, new_pi, obs_seq, hmm)
        new_prob = sum(new_alpha[T-1][states] for states in range(N))
        print(str(old_prob) + " " + str(new_prob))
        updated[counter][0] = new_STM
        updated[counter][1] = new_obsProb
        updated[counter][2] = new_pi
        updated[counter][3] = obs_seq
        counter += 1
    file_out = open(optimized_fname, "wb")
    for sequence in updated:
        file_out.write("{0} {1} {2}\n".format(hmm.N, hmm.M, len(sequence[3])))
        n_states = ""
        for state in hmm.states:
            n_states += str(state) + " "
        file_out.write("{0}\n".format(n_states))
        m_vocab = ""
        for word in hmm.vocab:
            m_vocab += str(word) + " "
        file_out.write("{0}\n".format(m_vocab))
        file_out.write("a:\n")
        for n in sequence[0]:
            output = ""
            for prob in n:
                output += str(prob) + " "
            file_out.write("{0}\n".format(output))
        file_out.write("b:\n")
        for n in sequence[1]:
            output = ""
            for prob in n:
                output += str(prob) + " "
            file_out.write("{0}\n".format(output))
        file_out.write("pi:\n")
        initial_distr = ""
        for prob in sequence[2]:
            initial_distr += str(prob) + " "
        file_out.write("{0}".format(initial_distr))
    file_out.close()
    
