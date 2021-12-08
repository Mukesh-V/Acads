from scipy import signal
import numpy as np

m = 1200
k1 = k2 = 25000
c1 = c2 = 1000

mw = 60
kw = 30000

# mds = mdr = 20
# kds, kdr = 280, 500
# cds, cdr = 2  , 20

mds = mdr = 20
kds, kdr = 280, 3000
cds, cdr = 2 , 500

a, b = 1.2, 1.5
c = d = 1 

ixx = 4000
iyy = 950

A = []
A_odd = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

A_even = []
a2 = [-2*(k1+k2)-(kds+kdr), -2*(c1+c2)-(cds+cdr), 2*(k1*a-k2*b)-kdr*b, 2*(c1*a-c2*b)-cdr*b, -kds*d, -cds*d, k1, c1, k1, c1, k2, c2, k2, c2, kdr, cdr, kds, cds]
a2 = np.array([q/m for q in a2])
A_even.append(a2)

a4 = [2*(k1*a-k2*b)-kdr*b, 2*(c1*a-c2*b)-cdr*b, -2*(k1*a**2+k2*b**2)-kdr*b**2, -2*(c1*a**2+c2*b**2)-cdr*b**2, 0, 0, -k1*a, -c1*a, -k1*a, -c1*a, k2*b, c2*b, k2*b, c2*b, kdr*b, cdr*b, 0, 0]
a4 = np.array([q/iyy for q in a4])
A_even.append(a4)

a6 = [-kds*d, -cds*d, 0, 0, -(c**2+d**2)*(k1+k2)-kds*d**2, -(c**2+d**2)*(c1+c2)-cds*d**2, -k1*c, -c1*c, k1*d, c1*d, -k2*c, -c2*c, k2*d, c2*d, 0, 0, kds*d, cds*d]
a6 = np.array([q/ixx for q in a6])
A_even.append(a6)

a8 = [k1, c1, -k1*a, -c1*a, -k1*c, -c1*c, -(k1+kw), -c1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
a8 = np.array([q/mw for q in a8])
A_even.append(a8)

a10 = [k1, c1, -k1*a, -c1*a, k1*d, c1*d, 0, 0, -(k1+kw), -c1, 0, 0, 0, 0, 0, 0, 0, 0]
a10 = np.array([q/mw for q in a10])
A_even.append(a10)

a12 = [k2, c2, k2*b, c2*b, -k2*c, -c2*c, 0, 0, 0, 0, -(k2+kw), -c2, 0, 0, 0, 0, 0, 0]
a12 = np.array([q/mw for q in a12])
A_even.append(a12)

a14 = [k2, c2, k2*b, c2*b, k2*d, c2*d, 0, 0, 0, 0, 0, 0, -(k2+kw), -c2, 0, 0, 0, 0]
a14 = np.array([q/mw for q in a14])
A_even.append(a14)

a16 = [kdr, cdr, kdr*b, cdr*b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kdr, -cdr, 0, 0]
a16 = np.array([q/mdr for q in a16])
A_even.append(a16)

a18 = [kds, cds, 0, 0, kds*d, cds*d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kds, -cds]
a18 = np.array([q/mds for q in a18])
A_even.append(a18)

for x in range(1, 19):
    if x%2 == 0:
        A.append(A_even[x//2-1])
    else:
        arr = A_odd.copy()
        arr[x] = 1
        A.append(np.array(arr))
A = np.array(A)
A = np.stack(A)

B = np.zeros((18, 4))
c = 0
for x in range(7, 14, 2):
    B[x][c] = kw/mw
    c += 1

C = np.stack([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
C = np.reshape(C, (1, 18))

D = np.zeros((1, 4))

system = signal.StateSpace(A, B, C, D)

def two_tmd_model():
    return (A, B, C, D, system)
