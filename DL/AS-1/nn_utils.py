import numpy as np
from data_utils import *
from activations import *

def init(m, n, mode=None):
    if mode == 'random':
        return np.random.rand(m, n) * 0.01
    elif mode == 'xavier':
        return np.random.rand(m, n) * np.sqrt(1/n)
    else:
        return np.zeros((m, n))
    
def nn_init(sizes, mode='dense', imode='random'):
    Wb, grads, history = [[], []], [[], []], [[], []]
    if mode == 'dense':
        for i in range(len(sizes)-1):
            Wb[0].append(init(sizes[i+1], sizes[i], imode))
            Wb[1].append(init(sizes[i+1], 1, imode))

            grads[0].append(init(sizes[i+1], sizes[i]))
            grads[1].append(init(sizes[i+1], 1))

            history[0].append(init(sizes[i+1], sizes[i]))
            history[1].append(init(sizes[i+1], 1))

    return Wb, grads, history
    
def forward(W, H, b, activation, mode='linear'):
    z = None
    if mode == 'linear':
        W = np.asarray(W)
        H = np.reshape(H, (H.shape[0], -1))
        z = np.dot(W, H) + b
    return globals()[activation](z), z

def forward_propagation(X, Wb, activation):
    H = X
    Hs, As = [], [H]
    for i in range(len(Wb[0])):
        Hi = H
        if i == len(Wb[0])-1: activation = 'softmax'
        H, z = forward(Hi, Wb[0][i], Wb[1][i], activation)
        Hs.append(H)
        As.append(z)
    return H, Hs, As

def backward(z, activation='sigmoid'):
    if activation == 'sigmoid':
        return sigmoid(z) * (1 - sigmoid(z))
    elif activation == 'tanh':
        return 1 - (tanh(z))**2
    else:
        return [max(0.0, i) for i in z]
    
def backpropagation(Wb, y_hat, y, activation, decay, loss, Hs, As):
    L = len(Wb[0])
    nc, ny = y_hat
    y_ohe = []
    for i in range(ny):
        y_ohe.append(one_hot(y[i], nc))
    y_ohe = np.reshape(y_ohe, (nc, ny))

    if loss == "cross_entropy":
        da = -(y_ohe - y_hat)
    elif loss == "mse":
        da = (y_hat - y_ohe)*y_hat - y_hat*(np.dot(y_hat - y).T, y_hat)

    grads = [[[] for t in range(L)], [[] for t in range(L)]]
    for j in range(L-1, -1, -1):
        dW = np.matmul(da, np.transpose(Hs[j])) + 2*decay*Wb[j]
        db = da
        if not j:
            dh = np.dot(Wb[j].T, da)
            da = np.multiply(dh, backward(As[j-1], activation))
        grads[0][j] = dW
        grads[1][j] = db
    
    return grads
