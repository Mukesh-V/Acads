import numpy as np

def one_hot(i, L):
    encoded = [0 for i in range(L)]
    encoded[i] = 1
    return np.asarray(encoded)
