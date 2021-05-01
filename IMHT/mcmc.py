import csv
import math
import numpy as np 

experimental_data = []
with open('data.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        experimental_data.append([ float(value) for value in row ])

random = []
with open('random.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        random.append([ float(value) for value in row ])

q = 1200
sigma_q = 120
sigma_t = 0.1
L = 0.07
k = 14.4
tl = 10

def simulated(x, a, b):
    return a*x + b

M = 3

final_data = []
for i in range(M+1):
    iteration_data = {}
    sum_S_old = 0.0
    sum_S_new = 0.0

    iteration_data['table'] = []
    u = random[i][0:3]

    for x in experimental_data:
        if i == 0:
            q_old = 1200
        else:
            q_old = final_data[i-1]['q']

        a = -q_old/k 
        b = tl + q_old *L/k 
        t_old = simulated(x[0], a, b)

        q_new = q_old + sigma_q * u[0] if u[1] > 0.5 else q_old - sigma_q * u[0]

        a = -q_new/k 
        b = tl + q_new *L/k 
        t_new = simulated(x[0], a, b)

        t_diff_old = t_old - x[1]
        t_diff_new = t_new - x[1]
        S_old = t_diff_old ** 2
        S_new = t_diff_new ** 2

        iteration_data['table'].append([x[0], t_old, t_new, x[1], t_diff_old, t_diff_new, S_old, S_new])

        sum_S_old += S_old
        sum_S_new += S_new
    
    with open('iteration'+str(i)+'.csv', 'w') as f:
        writer = csv.writer(f)
        for row in iteration_data['table']:
            writer.writerow(row)
    
    iteration_data['sum_S_old'] = sum_S_old
    iteration_data['sum_S_new'] = sum_S_new

    p_new = np.exp( -sum_S_new/(2 * sigma_q * sigma_q) ) / math.sqrt(2 * 3.1416 * sigma_q * sigma_q)
    p_old = np.exp( -sum_S_old/(2 * sigma_q * sigma_q) ) / math.sqrt(2 * 3.1416 * sigma_q * sigma_q)
    
    A = min(1, p_new/p_old)
    if A > u[2]:
        iteration_data['q'] = q_new
        iteration_data['flag'] = True
    else:
        iteration_data['q'] = q_old
        iteration_data['flag'] = False

    final_data.append(iteration_data)

print(len(final_data))
    
with open('mcmc_report.csv', 'w') as f:
    writer = csv.writer(f)
    for i in range(M+1):
        row = [final_data[i]['q'], final_data[i]['sum_S_old'], final_data[i]['sum_S_new'], final_data[i]['flag']]
        writer.writerow(row)