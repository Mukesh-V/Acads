import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from scipy.integrate import quad

pi = np.pi

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

class DiscreteTimeFourierTransform:
    def __init__(self, props):
        self.amp  = props['amp']
        self.phi  = props['phi']
        self.freq = props['freq']        
        self.tpd  = 1 / self.freq
        self.N = props['N']

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
    
    def setDataPoints(self):
        self.xpts = np.linspace(-self.N, self.N, 1000)
        self.datapoints = [self.f(x) for x in self.xpts]
    
    def getIndivFourier(self, w):
        coeff = 0.0
        for index, n in enumerate(self.xpts):
            coeff += self.datapoints[index] * np.exp(-1j * w * n)
        return coeff
    
    def reconstruct(self, n):
        integrand = lambda w : self.getIndivFourier(w) * np.exp(1j * w * n)
        return complexIntegrate(integrand, -pi, pi)[0]

    def plot(self):
        plt.plot(self.xpts, self.datapoints)
        # print([abs(self.reconstruct(n))/(2*pi) for n in self.xpts])
        # plt.plot(self.xpts, [abs(self.reconstruct(n))/(2*pi) for n in self.xpts])
        plt.show()

if __name__ == "__main__":
    props = {
        "amp"  : 1,
        "phi"  : 0,
        "freq" : 1,
        "N"    : 8
    }
    obj = DiscreteTimeFourierTransform(props)
    obj.triangle()
    obj.setDataPoints()
    obj.plot()
