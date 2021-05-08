# binary inversion
import numpy as np

# Group-4
N = 2048
nbits = int(np.log2(N-1))
real_indices = [ number for number in range(N) ]

def binary(number):
    binary = ''
    for bit in range(nbits, -1, -1):
        binary += str(number // ( 2 ** bit ))
        number = number % ( 2 ** bit )
    return binary

def binary_reverse(binary_num):
    number = 0
    for index, bit in enumerate(range(nbits, -1, -1)):
        number += int(binary_num[index]) * ( 2 ** bit )
    return number

real_indices = [ 1, 41, 100, N-20, N-1 ]
shuffled_indices = []
for number in real_indices:
    shuffled_indices.append( binary_reverse( binary(number)[::-1] ) )
print(shuffled_indices)