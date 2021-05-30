import networkx as nx
import matplotlib.pyplot as plt
  
n = 5+6

G1 = nx.barabasi_albert_graph(n = 20+n, m = 10)
pos = nx.random_layout(G1)
plt.figure(figsize = (12, 12))
plt.title('Scale Free')
nx.draw_networkx(G1, pos)
plt.show()

G2 = nx.watts_strogatz_graph(n = 15+n, k = 4, p = 0.5)
pos = nx.circular_layout(G2)
plt.figure(figsize = (12, 12))
plt.title('Small World')
nx.draw_networkx(G2, pos)
plt.show()