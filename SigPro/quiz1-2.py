from fourier import Fourier
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

tau = 2
a = 1

props = {
    'amp' : 1,
    'phi' : 0,
    'freq': 1/(4*a),
    'time':1,
    'tpd': 4*a
}
time_start = -1 + tau - 2*a
time_end = 1 + tau + 2*a
obj = Fourier(props)
t = sp.Symbol('t')
obj.f = sp.Piecewise((0, (t <= tau -a) & (t >= tau+2*a)), (sp.exp(1/((t-tau)**2 - a**2)), (t < tau + a) & (t > tau - a)), (0, True))
xpts = np.linspace(time_start, time_end, 100)
plt.plot(xpts, [obj.f.subs({t:x}) for x in xpts])
plt.show()

fourier_coeffs = obj.getNFouriers(2)
for i in range(5):
    print(obj.series[i], fourier_coeffs[i])