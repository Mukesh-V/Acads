import random
import matplotlib.pyplot as plt

def game(n, b):
    alreadyHeld, byzantines = set({1}), set({})
    for i in range(n-b+1, n+1):
        # ID Strategy (last set of IDs are byzantines)
        byzantines.add(i)

    current = 1
    for i in range(n-1):
        j = random.randint(2, n)
        if not j in alreadyHeld:
            current = j
        else:
            l = 2
            while l <= n:
                if l != current and not l in alreadyHeld:
                    break
                l += 1
            current = l
        alreadyHeld.add(current)

    return (current, current in byzantines)

if __name__ == "__main__":
    n, byz_count = 500, 0
    byz_data = []
    for b in range(0, int(n/2)-1):
        for i in range(100):
            data = game(n, b)
            if data[1]:
                byz_count += 1
        byz_data.append(byz_count/100)
        byz_count = 0

    plt.xlabel('Number of Byzantine nodes (b) per game(' + str(n) + ", b)")
    plt.ylabel('Probability that a Byzantine leader is chosen')
    plt.plot(range(0, int(n/2)-1), byz_data)
    plt.show()
    