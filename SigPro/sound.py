from scipy.io.wavfile import write
from fourier import Fourier

import sympy as sp
import numpy as np

time = 2
sr = 44000
pi = np.pi

props = {
    'amp' : 100,
    'phi' : 0,
    'freq': 1000,
    'time': time
}
aud = Fourier(props)
aud.sine()
xpts = np.linspace(0, time, sr)

t = sp.Symbol('t')
data = [sp.N(aud.f.subs({t: x})) for x in xpts]
data = np.array(data, dtype=float)
print(data)
write('mu2.wav', sr, data)