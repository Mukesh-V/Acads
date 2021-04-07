import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import scipy as scp

pi = sp.pi

class Fourier:
    def __init__(self, props):
        self.amp = props['amp']
        self.tpd = int(props['T'])

    def square(self):
        t = sp.Symbol('t')
        self.f = sp.Piecewise((self.amp, t < self.tpd/2), (0, True))
        print(self.f)
    
    def triangle(self):
        t = sp.Symbol("t")
        self.f = sp.Piecewise((self.amp * t/self.tpd , t < self.tpd/2), (self.amp - (self.amp*t)/self.tpd, True))
        print(self.f)

    def rectifiedSine(self):
        t = sp.Symbol('t')
        self.f = sympy.Piecewise((self.amp * sp.sin(t*2*pi/self.tpd), t<self.tpd/2), (0, True))
        print(self.f)
    
    def getIndivFourier(self, n):
        t = sp.Symbol("t")
        arg = 2 * n * sp.pi * t/self.tpd
        ce = sp.exp(-sp.I * arg)
        a = sp.integrate(self.f*ce, (t, 0, self.tpd)) / self.tpd
        return sp.N(a)

    def getNFouriers(self, N):
        series = range(-N, N+1)
        coeffs = [self.getIndivFourier(x) for x in series]
        self.coeffs = coeffs
        self.series = series
        return coeffs

    def reconstruct(self, t):
        y = 0
        for i, n in enumerate(self.series):
            y += self.coeffs[i] * scp.exp(1j * 2 * np.pi * n * t / self.tpd)
        return sp.N(y)
    
    def plot(self):
        t = sp.Symbol('t')
        xpts = np.linspace(0, 1, 1000, endpoint=False)
        plt.plot(xpts, [self.f.subs({t: x}) for x in xpts])
        plt.plot(xpts, [abs(self.reconstruct(x)) for x in xpts])
        plt.show()

    def errorNRG(self):
        t = sp.Symbol('t')


props = {
    "amp"  : 1,
    "T" : 1
}
obj = Fourier(props)
obj.triangle()
obj.getNFouriers(8)
obj.plot()
