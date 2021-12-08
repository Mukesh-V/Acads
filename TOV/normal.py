from scipy import signal
import numpy as np

m = 1200
k1 = k2 = 25000
c1 = c2 = 1000

mw = 60
kw = 30000

a, b = 1.2, 1.5
c = d = 1 

ixx = 4000
iyy = 950

A = []
A_odd = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

A_even = []
a2 = [-2*(k1+k2), -2*(c1+c2), 2*(k1*a-k2*b), 2*(c1*a-c2*b), (c-d)*(k1+k2), (c-d)*(c1+c2), k1, c1, k1, c1, k2, c2, k2, c2]
a2 = np.array([q/m for q in a2])
A_even.append(a2)

a4 = [2*(k1*a-k2*b), 2*(c1*a-c2*b), -2*(k1*a**2+k2*b**2), -2*(c1*a**2+c2*b**2), (c-d)*(k2*b-k1*a), -(c+d)*(c1*a + c2*b), -k1*a, -c1*a, -k1*a, -c1*a, k2*b, c2*b, k2*b, c2*b]
a4 = np.array([q/iyy for q in a4])
A_even.append(a4)

a6 = [(c-d)*(k1+k2), (c-d)*(c1+c2), (c-d)*(k2*b-k1*a), (c-d)*(c2*b-c1*a), -(c**2+d**2)*(k1+k2), -(c**2+d**2)*(c1+c2), -k1*c, -c1*c, k1*d, c1*d, -k2*c, -c2*c, k2*d, c2*d]
a6 = np.array([q/ixx for q in a6])
A_even.append(a6)

a8 = [k1, c1, -k1*a, -c1*a, -k1*c, -c1*c, -(k1+kw), -c1, 0, 0, 0, 0, 0, 0]
a8 = np.array([q/mw for q in a8])
A_even.append(a8)

a10 = [k1, c1, -k1*a, -c1*a, k1*d, c1*d, 0, 0, -(k1+kw), -c1, 0, 0, 0, 0]
a10 = np.array([q/mw for q in a10])
A_even.append(a10)

a12 = [k2, c2, k2*b, c2*b, -k2*c, -c2*c, 0, 0, 0, 0, -(k2+kw), -c2, 0, 0]
a12 = np.array([q/mw for q in a12])
A_even.append(a12)

a14 = [k2, c2, k2*b, c2*b, k2*d, c2*d, 0, 0, 0, 0, 0, 0, -(k2+kw), -c2]
a14 = np.array([q/mw for q in a14])
A_even.append(a14)

for x in range(1, 15):
    if x%2 == 0:
        A.append(A_even[x//2-1])
    else:
        arr = A_odd.copy()
        arr[x] = 1
        A.append(np.array(arr))
A = np.array(A)
A = np.stack(A)

B = np.zeros((14, 4))
c = 0
for x in range(7, 14, 2):
    B[x][c] = kw/mw
    c += 1

C = np.stack([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
C = np.reshape(C, (1, 14))

D = np.zeros((1, 4))

system = signal.StateSpace(A, B, C, D)

def normal_model():
    return (A, B, C, D, system)
