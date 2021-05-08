import matplotlib.pyplot as plt
from math import ceil

# Group 4
A = 10
a = 1.7
p = 0.13

weeks = 20
x = [ int(n**a) for n in range(weeks) ]
y = [A, A]

for n in range(2, weeks):
    # Since number of cases is discrete ( integer values )
    new = p * x[n-1]
    recovered = (3/4) * p * x[n-2]
    rip = (1/4) * p * x[n-3]

    # people have a chance to rest in peace, only after 3 weeks from testing positive
    if n-3 < 0: 
        rip = 0

    y.append( y[n-1] - new + recovered + rip  )

print(y)
plt.grid(True, which='both')
plt.axhline(0, color='k')
plt.xlabel('Weeks')
plt.ylabel('# of Available Beds')
plt.plot(range(weeks), y)
plt.show()
