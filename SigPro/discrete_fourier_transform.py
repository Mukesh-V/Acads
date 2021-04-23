import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

class DiscreteFourierTransform:
    def __init__(self, props):
        self.amp  = props['amp']
        self.time = props['time']
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
        self.xpts = np.linspace(0, self.time, self.N, endpoint=False)
        self.datapoints = [self.f(x) for x in self.xpts]
    
    def getIndivFourier(self, n):
        coeff = 0.0
        for k in range(self.N):
            coeff += self.datapoints[k] * np.exp(-1j * 2 * pi * k * n / self.N)
        return coeff
    
    def getNFouriers(self):
        self.coeffs = [ self.getIndivFourier(k) for k in range(self.N) ]
        return self.coeffs
    
    def reconstruct(self, k):
        y = 0.0
        for n in range(self.N):
            y += self.coeffs[n] * np.exp(1j * 2 * pi * k * n / self.N)
        return y/self.N

    def plot(self):
        plt.plot(self.xpts, self.datapoints)
        plt.plot(self.xpts, [abs(self.reconstruct(n)) for n in range(self.N)])
        plt.show()

if __name__ == "__main__":
    props = {
        "amp"  : 1,
        "phi"  : 0,
        "time" : 2,
        "freq" : 1,
        "N"    : 10
    }
    obj = DiscreteFourierTransform(props)
    obj.rectifiedSine()
    obj.setDataPoints()
    obj.getNFouriers()
    obj.plot()
