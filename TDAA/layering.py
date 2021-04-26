from ugraph import UndirectedGraph
import random

graph = UndirectedGraph()
'''
  a--c--e
  |  |  |
  b--d--f
'''
graph.addEdge('a', 'b')
graph.addEdge('a', 'c')
graph.addEdge('b', 'd')
graph.addEdge('c', 'd')
graph.addEdge('c', 'e')
graph.addEdge('d', 'f')
graph.addEdge('e', 'f')

'''
  1---2---3
  |   | / | \
  |   |/  |  \
  4   5---6---7  
'''
# graph.addEdge(1, 2)
# graph.addEdge(1, 4)
# graph.addEdge(2, 3)
# graph.addEdge(2, 5)
# graph.addEdge(3, 5)
# graph.addEdge(3, 6)
# graph.addEdge(3, 7)
# graph.addEdge(5, 6)

vertex_cover = set({})
weights = {}
for vertex in graph.vertices:
    weights[vertex] = 1

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
    
    print(list(sorted(vertex_cover)))
