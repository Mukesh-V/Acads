import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(z))

def softmax(z):
    ez = np.exp(z - np.max(z))
    return ez/ez.sum()

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)