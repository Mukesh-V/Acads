from fourier import Fourier
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

tau = 2
a = 1

props = {
    'amp' : 1,
    'phi' : 0,
    'freq': 1,
    'time': 2,
    'tpd':1
}
time_start = 0.5
time_end = 3.5
obj = Fourier(props)
t = sp.Symbol('t')
obj.f = sp.Piecewise((sp.exp(1/((t-tau)**2 - a**2)), (t < tau + a) & (t > tau - a)), (0, True))
xpts = np.linspace(time_start, time_end, 100)
plt.plot(xpts, [obj.f.subs({t:x}) for x in xpts])
plt.show()

obj.getNFouriers(3)
error_energy = sp.integrate( abs(obj.f - sp.nsimplify(obj.reconstruct(t)))**2 , (t, tau-a, tau+a))
print(sp.N(error_energy))