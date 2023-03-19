import numpy as np

def gd(Wb, grads, eta):
    L = len(Wb[0])
    for i in range(2):
        for j in range(L):
            Wb[i][j] -= eta * grads[i][j]
    return Wb

def momentum(Wb, grads, eta, history, gamma):
    L = len(Wb[0])
    for i in range(2):
        for j in range(L):
            history[i][j] = gamma * history[i][j] + eta * grads[i][j]
            Wb[i][j] -= history[i][j]
    return Wb, history

def rmsprop(Wb, grads, eta, history, beta, e):
    L = len(Wb[0])
    for i in range(2):
        for j in range(L):
            history[i][j] = beta * history[i][j] + (1-beta) * grads[i][j]**2
            Wb[i][j] -= (eta/(np.sqrt(history[i][j]) + e)) * grads[i][j]
    return Wb, history

def adam(Wb, grads, eta, m, v, beta1, beta2, e, t):
    L = len(Wb[0])
    m_hat = m.copy()
    v_hat = v.copy()
    for i in range(2):
        for j in range(L):
            m[i][j] = beta1 * m[i][j] + (1-beta1) * grads[i][j]
            v[i][j] = beta2 * v[i][j] + (1-beta2) * grads[i][j]**2

            m_hat[i][j] = m[i][j] / (1-np.power(beta1, t))
            v_hat[i][j] = v[i][j] / (1-np.power(beta2, t))

            Wb[i][j] -= (eta/(np.sqrt(v_hat[i][j]) + e)) * m_hat[i][j]
    return Wb, m, v

def nadam(Wb, grads, eta, m, v, beta1, beta2, e, t):
    L = len(Wb[0])
    m_hat = m.copy()
    v_hat = v.copy()
    m_bar = m.copy()
    for i in range(2):
        for j in range(L):
            m[i][j] = beta1 * m[i][j] + (1-beta1) * grads[i][j]
            v[i][j] = beta2 * v[i][j] + (1-beta2) * grads[i][j]**2

            m_hat[i][j] = m[i][j] / (1-np.power(beta1, t))
            v_hat[i][j] = v[i][j] / (1-np.power(beta2, t))

            m_bar[i][j] = beta1 * m_hat[i][j] + (1-beta1)*grads[i][j]

            Wb[i][j] -= (eta/(np.sqrt(v_hat[i][j]) + e)) * m_bar[i][j]
    return Wb, m, v