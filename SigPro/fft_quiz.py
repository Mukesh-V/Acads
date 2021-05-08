import sys
import numpy as np
import matplotlib.pyplot as plt

sys.setrecursionlimit(10**6)

pi = np.pi

class FastFourierTransformQuiz:
    def __init__(self, props):
        self.N = props['N']
    
    def setDataPoints(self):
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

    def plotFourier(self):
        points = self.N * np.arange(self.N)/self.N
        plt.plot(points, [abs(point) for point in self.coeffs])
        plt.show() 
