import csv
import numpy as np 
from scipy.special import logsumexp
from decimal import Decimal

def simulated(x, m, L):
    return np.cosh( m*(L-x) / (m*L) )

def P1(S, sigma):
    return np.exp(-S / (2 * sigma * sigma))

L = 0.3

experimental_data = []
with open('data.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        experimental_data.append([ float(value) for value in row ])

q = list( range(20, 61, 5) )
simulated_data = []
for i in range(len(q)):
    row = []

    m = (7278.677/q[i]) ** 0.5
    for x in experimental_data:
        row.append(simulated(x[0], m, L))
    simulated_data.append(row)

final_table = []
for i in range(len(q)):

    S = 0.0
    for j, x in enumerate(experimental_data):
        S += ( simulated_data[i][j] - x[1] ) ** 2
    
    p1 = P1(S, 0.008)
    p2 = q[i] * p1

    final_table.append([ S, p1, p2 ])

sigma_p1 = 0.0
for i in range(len(q)):
    sigma_p1 += final_table[i][1]

q_m = 0.0
for i in range(len(q)):
    q_m += np.exp( logsumexp([ np.log(q[i]), np.log(final_table[i][1]), -np.log(sigma_p1)]) )

max_ppdf = -1
q_map = 0
for i in range(len(q)):
    final_table[i].append( (q[i] - q_m)**2 * final_table[i][1] )
    ppdf = np.exp( logsumexp([ np.log(final_table[i][1]), np.log(sigma_p1)]) )
    if max_ppdf < ppdf:
        q_map = q[i]
        max_ppdf = ppdf
    final_table[i].append(ppdf)

q_variance = 0.0
for i in range(len(q)):
    q_variance += final_table[i][4]
q_variance = np.exp(logsumexp([np.log(q_variance),-np.log(sigma_p1)]))

with open('bayesian_report.csv', 'w') as f:
    writer = csv.writer(f)
    for row in final_table:
        writer.writerow(row)

print('MAP :', q_map)
print('Q_M :', q_m)
print('Q_variance :', q_variance)
