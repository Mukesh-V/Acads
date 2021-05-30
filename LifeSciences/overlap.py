import random
import matplotlib.pyplot as plt

kmers = ['AACA','AATA','ACAC','ACCT','AGCC','ATAA','CACC','CATA','CCAT','CCGT','CCTT','CGGG','CGTC','CTCG','CTTA',
'GAAT','GCCG','GGAA','GGGA','GGGG','GGGG','GTCT','TAAC','TAGC','TCGG','TCTC','TTAG']

kmers = list(set(kmers))
k = len(kmers[0]) - 1

graph = {}
for kmer in kmers:
    graph[kmer] = []

for i in range(len(kmers)):
    for j in range(len(kmers)):
        if i == j:
            continue
        if kmers[i][-k:] == kmers[j][:k]:
            graph[kmers[i]].append(kmers[j])

locations = {}
for vertex in kmers:
    x = random.randint(0, 100)
    y = random.randint(0, 100)
    locations[vertex] = (x, y)
    plt.scatter(locations[vertex][0], locations[vertex][1], s=30, color='red')
    plt.annotate(vertex, locations[vertex])

for vertex in kmers:
    x = [ locations[vertex][0] ]
    y = [ locations[vertex][1] ]
    for vertex2 in graph[vertex]:
        x.append( locations[vertex2][0] )
        y.append( locations[vertex2][1] )
        plt.plot(x, y, color='grey')
        x = [ locations[vertex][0] ]
        y = [ locations[vertex][1] ]

print('Graph :', graph)

hampath = ['CCAT', 'CATA', 'ATAA']
for item in graph.keys():
    node = item
    visit = []
    while True:
        try:
            node = graph[node][0]
            if node == item:
                break
            visit.append(node)
            if len(visit) > 27:
                break
        except:
            break  
    if item == 'ATAA':
        for v in visit:
            if v == 'GGGG':
                hampath.append('GGGG')
            hampath.append(v)

print('\nHamilton Path :',hampath)

for index, vertex in enumerate(hampath):
    x = [ locations[vertex][0] ]
    y = [ locations[vertex][1] ]
    try:
        x.append( locations[hampath[index+1]][0] )
        y.append( locations[hampath[index+1]][1] )
        plt.plot(x, y, color='blue')
    except:
        continue
plt.show()

recon = ''
for vertex in hampath:
    recon += vertex[0]
recon += vertex[1:]
print('\nReconstructed String :', recon)
