import matplotlib.pyplot as plt

def plot2var(name):
    with open(name) as f:
        lines = f.readlines()
        x = [line.split()[0] for line in lines]
        y = [line.split()[1] for line in lines]

        plt.plot(x, y)
        plt.show()

plot2var('plt_q1_lu.txt')
plot2var('history_q2a.txt')
