import sys

hmm_file = str(sys.argv[1])
obs_file = str(sys.argv[2])
#hmm_file = "sentence.hmm"
#obs_file = "example1.obs"

def initModel(model_file):
    model = {}
    model["states"] = {}
    model["vocab"] = {}
    with open(model_file, 'r') as f:
        i = 0
        inner = -1
        N = 0
        M = 0
        for line in f:
            if i == 0:
                j = 0
                for word in line.split():
                    if j == 0:
                        N = int(word)
                    elif j == 1:
                        M = int(word)
                    else:
                        model["T"] = int(word.strip())
                    j += 1
            elif i == 1:
                index = 0
                for word in line.split():
                    model["states"][index] = word.strip()#changed
                    index += 1
            elif i == 2:
                index = 0
                for word in line.split():
                    model["vocab"][word.strip()] = index
                    index += 1
            elif i == 3:
                if inner == -1:
                    model["state_trans"] = [[0.0 for col in range(N)] for row in range(N)]
                    inner += 1
                    continue
                elif inner < N:
                    cols = 0
                    for word in line.split():
                        model["state_trans"][inner][cols] = float(word.strip())
                        cols += 1
                    inner += 1
                    continue
                elif inner == N:
                    model["obsProb"] = [[0.0 for col in range(M)] for row in range(N)]
                    inner += 1
                    continue
                elif inner < (2*N + 1):
                    cols = 0
                    for word in line.split():
                        model["obsProb"][(inner%N) - 1][cols] = float(word.strip())
                        cols += 1
                    inner += 1
                    continue
                elif inner == (2*N + 1):
                    model["initial"] = []
                    inner += 1
                    continue
                else:
                    for word in line.split():
                        model["initial"].append(float(word.strip()))
            i += 1
    return model
#print(initModel("sentence.hmm"))

#Model: (dictionary)
'''
Key                        Value
model["states"]            (dict) of (ints) index-keys for state-values in model -- "SUBJECT", "AUXILIARY", "PREDICATE", "OBJECT"
^changed
model["vocab"]             (dict) of (ints) index-values for each word-key in our vocab -- "kids", "robots", "do", "can", "play", "eat", "chess", "food"
model["T"] = T             (int) time step/length of obs. sequence
model["state_trans"] = a   (2D array) filled with corresponding probabilities of indexes of state-index by state-index
model["obsProb"] = b       (2D array) filled with corresponding probabilities of indexes of state-index by vocab-index
model["initial"] = pi      (array) of probabilities corresponding to indexes of state-index
'''
def statePath(model, obs_seq):
    states = model["states"]
    vocab = model["vocab"]
    T = model["T"]
    A = model["state_trans"]
    B = model["obsProb"]
    pi = model["initial"]
    
    viterbi = [None]*(len(obs_seq) + 1)
    viterbi[0] = 0
    graph = [[0.0 for state in range(len(states.keys()))] for observation in range(len(obs_seq))]
    
    for i in range(len(graph)):
        #print(obs_seq[i])
        obs = vocab[obs_seq[i]]
        for state in range(len(states.keys())):
            if i == 0:
                '''
                graph[0][state] = 0.0
                for prev_state in range(len(states.keys())):
                    next_term = pi[prev_state]*A[prev_state][state]
                    #print(next_term)
                    graph[0][state] += next_term
                    #print(graph[0][state])
                initial_factor = B[state][obs]
                #print(initial_factor)
                graph[0][state] *= initial_factor
                #print(states[state])
                #print(graph[0][state])'''
                graph[0][state] = B[state][obs]*pi[state]
            else:
                terms = []
                for prev_state in range(len(states.keys())):
                    terms.append(graph[i-1][prev_state]*A[prev_state][state])
                #print(terms)
                #print(states[state])
                graph[i][state] = (B[state][obs] * max(terms))
    viterbi[0] = max(graph[-1])
    #print(obs_seq)
    #print(graph)
    for i in range(1, len(viterbi)):
        viterbi[i] = states[graph[i-1].index(max(graph[i-1]))]

    return viterbi

def main(hmm_file, obs_file):
    model = initModel(hmm_file)
    #print(model)
    sequences = []
    with open(obs_file, 'r') as f:
        i = 0
        inner = 0
        for line in f:
            if i == 0:
                sequences = [None]*(int(line.strip()))
                i += 1
            elif i < ((len(sequences))*2 + 1):
                if (inner%2) == 0:
                    sequences[int(inner/2)] = [None]*(int(line.strip()))
                else:
                    j = 0
                    for word in line.split():
                        sequences[int(inner/2)][j] = word.strip()
                        j += 1
                inner += 1
                i += 1
    for sequence in sequences:
        viterbi = statePath(model, sequence)
        output = ""
        for item in viterbi:
            if item == 0:
                output += str(item)
                break
            else:
                output += str(item) + " "
        print(output)

main(hmm_file, obs_file)

























