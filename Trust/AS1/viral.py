import random
import matplotlib.pyplot as plt

def game(n, b, rounds):
    population = {}
    healthy, infected = [], []

    for i in range(b):
        while True:
            infectedNodeId = random.randint(0, n-1)
            if not infectedNodeId in population:
                break
        population[infectedNodeId] = 1
        infected.append(infectedNodeId)

    for i in range(n):
        if not i in population:
            population[i] = 0
            healthy.append(i)


    data = [(0, n-b, b)]
    for i in range(rounds):
        if not len(infected) or not len(healthy):
            break

        for j in range(len(infected)):
            if not len(infected) or not len(healthy):
                break
            if len(healthy) > 1:
                newlyInfectedId = random.randint(0, len(healthy)-1)
            else:
                newlyInfectedId = 0
            newlyInfected = healthy[newlyInfectedId]
            healthy.pop(newlyInfectedId)
            infected.append(newlyInfected)
            population[newlyInfected] = 1
        
        for k in range(len(healthy)):
            if not len(healthy) or not len(infected):
                break
            if len(infected) > 1:
                newlyHealedId = random.randint(0, len(infected)-1)
            else:
                newlyHealedId = 0
            newlyHealed = infected[newlyHealedId]
            infected.pop(newlyHealedId)
            healthy.append(newlyHealed)
            population[newlyHealed] = 0

        data.append((i+1, len(healthy), len(infected))) 
    
    return data


if __name__ == "__main__":
    round_data, final_data = [], []
    n = 500
    for i in range(1, n):
        data = game(n, i, 100)[-1]
        round_data.append(data[0])
        final_data.append(data[1]/n)

    plt.xlabel('Number of Intially Infected : b')
    plt.plot(range(1, n), round_data)
    plt.plot(range(1, n), final_data)
    plt.legend(['Number of Phases', 'Final number of healthy / n'])
    plt.show()