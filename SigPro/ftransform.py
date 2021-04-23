import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from scipy.integrate import quad

pi = np.pi
integral_limit = 5

# copied from StackOverflow
# https://stackoverflow.com/a/5966088/11607707
def complexIntegrate(func, a, b, **kwargs):
    def real_func(x):
        return sp.real(func(x))
    def imag_func(x):
        return sp.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

class FourierTransform:
    def __init__(self, props):
        self.amp  = props['amp']
        self.time = props['time']
        self.phi  = props['phi']
        self.freq = props['freq']        
        self.tpd  = 1 / self.freq

    def freqx(self, x):
        return 2 * pi * self.freq * x + self.phi

    def freqn(self, x):
        return (2 * pi * self.freq * x + self.phi) // pi

    def square(self):
        self.f = lambda x : self.amp if self.freqn(x)%2 == 0 else 0

    def triangle(self):
        self.f = lambda x : self.amp * (self.freqx(x) - pi*self.freqn(x))/pi if self.freqn(x)%2 == 0 else self.amp * (1 - (self.freqx(x) - pi*self.freqn(x))/pi)

    def rectifiedSine(self):
        self.f = lambda x : self.amp * np.sin(self.freqx(x)) if np.sin(self.freqx(x)) > 0 else 0 

    def impulse(self):
        self.f = lambda x : 0 if x%1/(2*self.freq) != 0 or x == 0 else 100 * self.amp
    
    def getIndivFourier(self, w):
        integrand = lambda t : self.f(t) * sp.exp(-1j * 2 * pi * w * t)
        return complexIntegrate(integrand, -integral_limit, integral_limit)[0]
    
    def reconstruct(self, t):
        print('t :', t)
        integrand = lambda w : self.getIndivFourier(w) * sp.exp(1j * 2 * pi * w * t)
        return complexIntegrate(integrand, -integral_limit, integral_limit)[0]

    def plot(self):
        xpts = np.linspace(0, self.time, 10, endpoint=False)
        plt.plot(xpts, [self.f(x) for x in xpts])
        plt.plot(xpts, [abs(self.reconstruct(x)) for x in xpts])
        plt.show()

if __name__ == "__main__":
    props = {
        "amp"  : 1,
        "phi"  : 0,
        "time" : 1,
        "freq" : 1
    }
    obj = FourierTransform(props)
    obj.rectifiedSine()
    obj.plot()
