import matplotlib.pyplot as plt
import sympy as sp
import numpy as np

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
        t = sp.Symbol('t')
        self.f = sp.Piecewise((self.amp * t/self.tpd , t < self.tpd/2), (self.amp - (self.amp*t)/self.tpd, True))
        print(self.f)

    def rectifiedSine(self):
        t = sp.Symbol('t')
        self.f = sp.Piecewise((self.amp * sp.sin(2 * pi * t/self.tpd), t<self.tpd/2), (0, True))
        print(self.f)
    
    def getIndivFourier(self, n):
        t = sp.Symbol('t')
        expr = sp.integrate(self.f * sp.exp(-sp.I * 2 * n * sp.pi * t/self.tpd), (t, 0, self.tpd)) / self.tpd
        return sp.N(expr)

    def getNFouriers(self, N):
        series = range(-N, N+1)
        coeffs = [self.getIndivFourier(x) for x in series]
        self.coeffs = coeffs
        self.series = series
        return coeffs

    def reconstruct(self, t):
        coeffs = sp.Array(self.coeffs)
        series = sp.Array(self.series)
        N = len(self.series)
        i = sp.Symbol('i')
        expr = sp.summation(coeffs[i] * sp.exp(sp.I * 2 * pi * series[i] * t/self.tpd), (i, 0, N-1))
        return sp.N(expr)
    
    def plot(self):
        t = sp.Symbol('t')
        xpts = np.linspace(0, 1, 100, endpoint=False)
        plt.plot(xpts, [self.f.subs({t: x}) for x in xpts])
        plt.plot(xpts, [abs(self.reconstruct(x)) for x in xpts])
        plt.show()

    def errorNRG(self):
        t = sp.Symbol('t')
        expr = sp.integrate( abs(self.f - sp.nsimplify(self.reconstruct(t)))**2 , (t, 0, self.tpd))
        return sp.N(expr)

props = {
    'amp': 1,
    'T'  : 1
}
obj = Fourier(props)
obj.rectifiedSine()
obj.getNFouriers(5)
obj.plot()

print('Error :', abs(obj.errorNRG()))