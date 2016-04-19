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
            
    #(3): Termination
    prob = sum(alpha[T-1][states] for states in range(N))
        
    return prob
    
# -------------------------------
# MAIN PROGRAM
# -------------------------------

if __name__ == '__main__':

    # Get names of files
    hmm_fname = sys.argv[1] # Original HMM
    obser_fname = sys.argv[2] # Observations
    #optimized_fname = sys.argv[3] # Output HMM

    # Initialize new HMM and read it from file
    hmm = HMM()
    hmm.read_hmm_from_file(hmm_fname)

    # Reading observation file
    obser_fd = open(obser_fname, "r")
    num_of_obser_sets = int(obser_fd.readline().rstrip())
    observations = []

    for i in range(0, num_of_obser_sets):
        observation = {}
        observation["T"] = int(obser_fd.readline().rstrip())
        observation["seq"] = obser_fd.readline().rstrip().split(" ")
        observations.append(observation)
    
    #implement forward algorithm for each observation sequence in the obs file
    for observation in observations:
        prob = forward(hmm, observation["T"],observation["seq"])
        print(prob)

    
