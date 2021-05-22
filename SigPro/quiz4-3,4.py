import numpy as np

p = [ 0, 0.7 ]
d = [ 1, -1.5 ]
summation_x = 0.0
summation_y = 0.0
w = 1

for k, _ in enumerate(d):
    summation_x += p[k] * np.exp( -1j * w * k)
    summation_y += d[k] * np.exp( -1j * w * k)

H = summation_x / summation_y
print(H)

p = [ 0, 1.8 ]
d = [ 1, -0.7 ]
summation_x = 0.0
summation_y = 0.0
w = 1

for k, _ in enumerate(d):
    summation_x += p[k] * np.exp( -1j * w * k)
    summation_y += d[k] * np.exp( -1j * w * k)

H = summation_x / summation_y
print(H)