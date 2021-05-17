import matplotlib.pyplot as plt

x = [40000]
y = [40000]

n = 1
months = 100
balance = y[0]
while y[-1] > 0:
    living_expenses = 10000 * (1 + 0.01)**(n-1)
    discretionary_expenses = y[n-1]/2
    x.append( x[0] * (1 + 0.006)**n )
    y.append( x[n] - (living_expenses + discretionary_expenses) )
    balance += y[-1]
    if balance >= 1000000:
        print('%d/%d' % (n%12 + 1, n//12 + 2000))
    n += 1

print(y[-1])
print('%d/%d' % (n%12 + 1, n//12 + 2000))

plt.grid(True, which='both')
plt.axhline(0, color='k')
plt.xlabel('Months')
plt.ylabel('Monthly Saving')
plt.plot(range(n), y)
plt.show()