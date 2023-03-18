import numpy as np

def gd(Wb, grads, eta, L):
    for i in range(2*L):
        Wb[i] = Wb[i] - eta*grads[i]
    return Wb

def momentum(Wb, grads, eta, gamma, history, L):
    for i in range(2*L):
        history[i] = gamma*history[i] + eta*grads[i]
        Wb[i] = Wb[i]