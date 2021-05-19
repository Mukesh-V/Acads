import copy
import random
from numpy import random
from ugraph import UndirectedGraph

base_graph = UndirectedGraph()
'''
  a--c--e
  |  |  |
  b--d--f
'''
# graph.addEdge('a', 'b')
# graph.addEdge('a', 'c')
# graph.addEdge('b', 'd')
# graph.addEdge('c', 'd')
# graph.addEdge('c', 'e')
# graph.addEdge('d', 'f')
# graph.addEdge('e', 'f')

'''
  1---2---3
  |   | / | \
  |   |/  |  \
  4   5---6---7  
'''
base_graph.addEdge(1, 2)
base_graph.addEdge(1, 4)
base_graph.addEdge(2, 3)
base_graph.addEdge(2, 5)
base_graph.addEdge(3, 5)
base_graph.addEdge(3, 6)
base_graph.addEdge(3, 7)
base_graph.addEdge(5, 6)
base_graph.addEdge(6, 7)

graph_copies = [ copy.deepcopy(base_graph) for i in range(3) ]

for index, graph in enumerate(graph_copies):

    vertex_cover = set({})
    weights = {}

    if index == 0:
        print('Weights initialized as 1')
        for vertex in graph.vertices:
            weights[vertex] = 1
    if index == 1:
        print('Weights picked independently and uniformly')
        for vertex in graph.vertices:
            weights[vertex] = random.uniform(0.5, 1)
    if index == 2:
        print('Weights picked independently from standard normal distribution')
        for vertex in graph.vertices:
            weights[vertex] = abs(random.standard_normal())

    print(weights)
        
    while(graph.edgesExist()):

        min_c = 1.1
        for vertex in graph.vertices:
            degree = len(graph.edges[vertex])
            if degree == 0:
                graph.clearVertex(vertex)
                continue
            min_c = weights[vertex]/degree if weights[vertex]/degree < min_c else min_c
    
        residual = {}
        for vertex in graph.vertices:
            degree = len(graph.edges[vertex])
            weights[vertex] -= min_c * degree
        
        for vertex in weights.keys():
            if weights[vertex] == 0:
                vertex_cover.add(vertex)
                graph.clearVertex(vertex)
        
    print('Vertex cover :', list(sorted(vertex_cover)))
    print('')
