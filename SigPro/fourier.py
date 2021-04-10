import matplotlib.pyplot as plt
import sympy as sp
import numpy as np

pi = sp.pi

class Fourier:
    def __init__(self, props):
        self.amp  = props['amp']
        self.time = props['time']
        self.phi  = props['phi']
        self.freq = props['freq']        
        self.tpd  = props['tpd']

    def freqx(self, x):
        return 2 * pi * self.freq * x + self.phi

    def freqn(self, x):
        return (2 * pi * self.freq * x + self.phi) // pi

    def sine(self):
        t = sp.Symbol('t')
        self.f = sp.Piecewise((self.amp * sp.sin(self.freqx(t)), True))
        print(self.f)

    def square(self):
        t = sp.Symbol('t')
        flag = sp.Function('flag')(t)
        flag = self.freqn(t) % 2 
        self.f = sp.Piecewise((0, flag>0), (self.amp, True))
        print(self.f)
    
    def triangle(self):
        t = sp.Symbol('t')
        flag = sp.Function('flag')(t)
        flag = self.freqn(t) % 2
        self.f = sp.Piecewise((self.amp * (1 - (self.freqx(t) - pi*self.freqn(t))/pi) , flag>0), (self.amp * (self.freqx(t) - pi*self.freqn(t))/pi, True))
        print(self.f)

    def rectifiedSine(self):
        t = sp.Symbol('t')
        flag = sp.Function('flag')(t)
        flag = sp.sin(self.freqx(x))
        self.f = sp.Piecewise((self.amp * sp.sin(self.freqx(t)), flag>0), (0, True))
        print(self.f)
    
    def getIndivFourier(self, n):
        t = sp.Symbol('t')
        print(n)
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
        xpts = np.linspace(0, self.time, 100, endpoint=False)
        plt.plot(xpts, [self.f.subs({t: x}) for x in xpts])
        plt.plot(xpts, [abs(self.reconstruct(x)) for x in xpts])
        plt.show()

    def errorNRG(self):
        t = sp.Symbol('t')
        expr = sp.integrate( abs(self.f - sp.nsimplify(self.reconstruct(t)))**2 , (t, 0, self.tpd))
        return sp.N(expr)

if __name__ == "__main__":
    props = {
        'amp' : 1,
        'phi' : 0,
        'freq': 1,
        'time': 2
    }
    obj = Fourier(props)
    obj.square()
    obj.getNFouriers(3)
    # obj.plot()

    print('Error :', abs(obj.errorNRG()))