import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from scipy.integrate import quad

pi = np.pi

class Fourier:
    def __init__(self, props):
        self.amp   = props['amp']
        self.phi   = props['phi']
        self.time  = props['time']
        self.freq  = props['freq']

    def freqx(self, x):
        return 2 * pi * self.freq * x + self.phi

    # copied from StackOverflow
    # https://stackoverflow.com/questions/5965583/use-scipy-integrate-quad-to-integrate-complex-numbers
    def complexIntegrate(self, func, a, b, **kwargs):
        def real_func(x):
            return sp.real(func(x))
        def imag_func(x):
            return sp.imag(func(x))
        real_integral = quad(real_func, a, b, **kwargs)
        imag_integral = quad(imag_func, a, b, **kwargs)
        return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

    def square(self):
        self.f = lambda x : 1 * self.amp if (self.freqx(x)//pi)%2==0 else 0
        self.tpd = 1 / self.freq

    def triangle(self):
        self.f = lambda x : self.amp * (self.freqx(x) - pi*(self.freqx(x)//pi))/pi if self.freqx(x)//pi%2 == 0 else self.amp * (1 - (self.freqx(x) - pi*(self.freqx(x)//pi))/pi)
        self.tpd = 1 / self.freq
    
    def rectifiedSine(self):
        self.f = lambda x : self.amp * ( abs(np.sin(x+self.phi)) )
        self.tpd = 2 * pi / self.freq
    
    def getIndivFourier(self, n):
        integrand = lambda t : self.f(t) * sp.exp(-1j * 2 * pi * n * t / self.tpd)
        return self.complexIntegrate(integrand, 0, self.tpd)[0]/self.tpd
    
    def getNFouriers(self, n):
        series = range(-n, n+1)
        coeffs = [obj.getIndivFourier(x) for x in series]
        self.coeffs = coeffs
        self.series = series
        return coeffs
    
    def reconstruct(self, t):
        y = 0
        for i, n in enumerate(self.series):
            y += self.coeffs[i] * sp.exp(1j * 2 * pi * n * t / self.tpd) 
        return abs(y)

props = {
    "amp"  : 1,
    "phi"  : pi/3,
    "time" : 1,
    "freq" : 1,
}
obj = Fourier(props)
obj.triangle()
obj.getNFouriers(5)

xpts = np.linspace(0, props['time'], 1000, endpoint=False)
plt.plot(xpts, [obj.f(x) for x in xpts])
plt.plot(xpts, [obj.reconstruct(x) for x in xpts])
plt.show()