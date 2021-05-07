import sys
import numpy as np
import matplotlib.pyplot as plt

sys.setrecursionlimit(10**6)

pi = np.pi

class FastFourierTransform:
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
        self.datapoints = np.array([self.f(x) for x in self.xpts])
    
    def getIndivFourier(self, segment):
        N = segment.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, segment)
    
    def getNFouriers(self, segment=[], save=True):
        if segment == []:
            segment = self.datapoints
        N = segment.shape[0]
        if N <= 2:
            return self.getIndivFourier(segment)
        even_half = self.getNFouriers(segment[::2], save=False)
        odd_half = self.getNFouriers(segment[1::2], save=False)

        weights = np.exp( -2j * pi * np.arange(N) / N )
        concat =  np.concatenate([even_half + weights[: N//2] * odd_half, even_half + weights[N//2 :] * odd_half])
        if save:
            self.coeffs = concat
        return concat
    
    def reconstruct(self):
        segment = np.array([0+0j])
        segment[0] = self.coeffs[0]
        segment = np.concatenate([segment, np.flip(self.coeffs[1:])])
        inverse = self.getNFouriers(np.array(segment), save=False)
        return inverse/self.N

    def plot(self):
        actual_xpts = np.linspace(0, self.time, 100, endpoint=False)
        plt.plot(self.xpts, [abs(x) for x in self.reconstruct()])
        plt.show()

if __name__ == "__main__":
    props = {
        "amp"  : 1,
        "phi"  : 0,
        "time" : 2,
        "freq" : 1,
        "N"    : 128
    }
    obj = FastFourierTransform(props)
    obj.rectifiedSine()
    obj.setDataPoints()
    obj.getNFouriers()
    obj.plot()
