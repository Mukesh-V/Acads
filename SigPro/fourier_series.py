
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

class FourierSeries:
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
    
    def getIndivFourier(self, n):
        integrand = lambda t : self.f(t) * sp.exp(-1j * 2 * pi * n * t / self.tpd)
        return complexIntegrate(integrand, 0, self.tpd)[0]/self.tpd
    
    def getNFouriers(self, n):
        series = range(-n, n+1)
        coeffs = [self.getIndivFourier(x) for x in series]
        self.coeffs = coeffs
        self.series = series
        return coeffs
    
    def reconstruct(self, t):
        y = 0
        for i, n in enumerate(self.series):
            y += self.coeffs[i] * sp.exp(1j * 2 * pi * n * t / self.tpd)
        return y

    def plot(self):
        xpts = np.linspace(0, self.time, 1000, endpoint=False)
        plt.plot(xpts, [self.f(x) for x in xpts])
        plt.plot(xpts, [abs(self.reconstruct(x)) for x in xpts])
        plt.show()

    def errorNRG(self):
        integrand = lambda t : abs(self.f(t) - self.reconstruct(t)) ** 2
        return complexIntegrate(integrand, 0, self.tpd)[0]

    def verifyParseval(self):
        lhs = lambda t : self.f(t)**2
        lhs = abs(complexIntegrate(lhs, 0, self.tpd)[0])
        rhs = 0.0
        for i, n in enumerate(self.series):
            rhs += abs(self.coeffs[i])**2
        rhs *= self.tpd
        print('to verify Parseval theorem')
        print('LHS :', lhs)
        print('RHS :', rhs)

if __name__ == "__main__":
    props = {
        "amp"  : 1,
        "phi"  : 0,
        "time" : 4,
        "freq" : 0.5
    }
    obj = FourierSeries(props)
    obj.triangle()
    obj.getNFouriers(8)
    obj.plot()
    # energy of error
    print('Error when N = 8  :', abs(obj.errorNRG()))

    # to verify energy of error -> 0 as N -> infi
    obj.getNFouriers(50)
    print('Error when N = 50 :', abs(obj.errorNRG()))

    # to verify convergence at point of discontinuity
    obj.impulse()
    obj.getNFouriers(8)
    discon = 1/( 2*props['freq'] )
    obj.plot()
    print('')
    print('to verify convergence at discontinuity at t =', discon)
    print('from the actual function :', obj.f(discon))
    print('from the reconstruction by Fourier series :', abs(obj.reconstruct(discon)))

    # to verify Parseval's
    obj.triangle()
    obj.getNFouriers(50)
    print('')
    obj.verifyParseval()

    obj.getNFouriers(10)

    # to verify Convolution of signals
    obj2 = Fourier(props)
    obj2.square()
    obj2.getNFouriers(10)

    obj3 = Fourier(props)
    obj3.f = lambda t : abs(complexIntegrate(lambda tau: obj.f(tau)*obj2.f(t-tau), -10, 10)[0])
    obj3.getNFouriers(10)
    print('')
    print('Plots to verify convolution')
    xpts = np.linspace(0, props['time'], 10, endpoint=False) 
    plt.plot(xpts, [abs(obj3.reconstruct(x)) for x in xpts])
    plt.plot(xpts, [abs(obj.reconstruct(x)) * abs(obj2.reconstruct(x)) for x in xpts])
    plt.show()

    # to verify Multiplication of signals
    obj2 = Fourier(props)
    obj2.square()
    obj2.getNFouriers(10)

    obj3 = Fourier(props)
    obj3.f = lambda x : obj.f(x) * obj2.f(x)
    obj3.getNFouriers(10)
    print('')
    print('Plots to verify multiplication')
    xpts = np.linspace(0, props['time'], 10, endpoint=False) 
    rhs = lambda w : abs(complexIntegrate(lambda theta: obj.reconstruct(theta)*obj2.reconstruct(w-theta), -10, 10)[0])
    plt.plot(xpts, [obj.f(x) * obj2.f(x) for x in xpts])
    plt.plot(xpts, [rhs(x) for x in xpts])
    plt.plot(xpts, [abs(obj3.reconstruct(x)) for x in xpts])
    plt.show()

    # to verify Linearity
    obj2 = Fourier(props)
    obj2.square()
    obj2.getNFouriers(10)
    print('')
    print('Plots to verify linearity')
    a = 2
    b = 4
    xpts = np.linspace(0, props['time'], 100, endpoint=False) 
    plt.plot(xpts, [a * obj.f(x) + b * obj2.f(x) for x in xpts])
    plt.plot(xpts, [a * abs(obj.reconstruct(x)) + b * abs(obj2.reconstruct(x)) for x in xpts])
    plt.show()

    # to verify Conjugate
    obj2 = Fourier(props)
    obj2.rectifiedSine()
    obj2.getNFouriers(10)
    print('')
    print('Plots to verify conjugates')
    xpts = np.linspace(0, props['time'], 100, endpoint=False) 
    plt.plot(xpts, [obj2.f(x) for x in xpts])
    plt.plot(xpts, [abs(obj2.reconstruct(x)) for x in xpts])
    plt.show()