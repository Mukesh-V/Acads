import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

class DiscreteFourierTransformQuiz:
    def __init__(self, props):
        self.N = props['N']
    
    def setDataPoints(self, datapoints):
        self.datapoints = datapoints
    
    def getIndivFourier(self, n):
        coeff = 0.0
        for k in range(self.N):
            coeff += self.datapoints[k] * np.exp(-1j * 2 * pi * k * n / self.N)
        print(n)
        return coeff
    
    def setNFouriers(self):
        self.coeffs = [ self.getIndivFourier(k) for k in range(self.N) ]
        return self.coeffs
    
    def reconstruct(self, k):
        y = 0.0
        for n in range(self.N):
            y += self.coeffs[n] * np.exp(1j * 2 * pi * k * n / self.N)
        return y/self.N

    def plotFourier(self):
        points = self.N * np.arange(self.N)/self.N
        plt.plot(points, [abs(point) for point in self.coeffs])
        plt.show()    
