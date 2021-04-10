from scipy.integrate import quad
import scipy as sp
import numpy as np
pi = np.pi

tau = 2
a = 1
time_period = 4*a

f = lambda t : np.exp(1/((t-tau)**2 - a**2)) if (t < tau + a) and (t > tau - a) else 0
def complexIntegrate(func, a, b, **kwargs):
    def real_func(x):
        return sp.real(func(x))
    def imag_func(x):
        return sp.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def getIndivFourier(n):
    integrand = lambda t : f(t) * sp.exp(-1j * 2 * pi * n * t / time_period)
    return complexIntegrate(integrand, 1, 3)[0]/time_period

def getNFouriers(n):
    series = range(-n, n+1)
    coeffs = [getIndivFourier(x) for x in series]
    return coeffs
    
fourier_coeffs = getNFouriers(2)
print(fourier_coeffs)

average_integrand = lambda t : f(t) 
print(complexIntegrate(average_integrand, 0, 4*a)[0])