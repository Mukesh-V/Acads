import csv
import matplotlib.pyplot as plt

data = []
titles = ['Equal', 'Uniform', 'Normal']
subtitles = ['Non-perturbed', 'Perturbed']
methods = ['equal', 'equal_perturbed', 'uniform','uniform_perturbed', 'normal', 'normal_perturbed']
for method in methods:
    method_data = []
    with open(method + '.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            method_data.append([ int(row[0]), int(row[1]), float(row[2]) ])
    data.append(method_data)

figure = plt.figure()
figure.canvas.set_window_title('Layering Algorithm Analysis')
for index, method_data in enumerate(data):
    n = []
    vc = []
    time = []
    for row in method_data:
        n.append(row[0])
        vc.append(row[1])
        time.append(row[2])
    
    plt.subplot(3, 2, (index//2)*2 + 1)
    plt.grid(True)
    plt.legend(subtitles)
    plt.title(titles[index//2])
    plt.plot(n, vc)

    plt.subplot(3, 2, (index//2)*2 + 1 + 1)
    plt.grid(True)
    plt.legend(subtitles)
    plt.title(titles[index//2])
    plt.plot(n, time)

plt.suptitle('Size of VC and Running time')
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
plt.show()