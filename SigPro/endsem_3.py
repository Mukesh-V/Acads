import csv
import math
import numpy as np

whole_data = []
with open('endsem_3.csv') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        whole_data.append([float(value) for value in row])
whole_data = np.array(whole_data)

sigma = 2

data = whole_data[:, 0]
N = len(data)
mean = np.mean(data)
interval = 1.645 * sigma / math.sqrt(N)
print( mean, mean-interval, mean+interval )

A = 1
count = 0
for day in range(30):
    data = whole_data[:, day]
    N = len(data)
    mean = np.mean(data)
    # zscore is 1.645 for 90% confidence in a standard-normal distribution
    interval = 1.645 * sigma / math.sqrt(N)

    if mean-interval <= A and mean+interval >= A:
        count += 1
    else:
        print('Day ', day+1)

print('Count :', count)

N = len(data)
while interval >= 0.09:
    data = whole_data[:, day]
    sigma = 2
    mean = np.mean(data)
    interval = 1.645 * sigma / math.sqrt(N)
    N += 1

print(interval, N)