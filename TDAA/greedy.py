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

vertices = sorted(list(graph.vertices))
vertex_cover = set({})

while(graph.edgesExist()):
    i1 = random.randint(0, len(vertices)-1)
    v1 = vertices[i1]
    i2 = random.randint(0, len(graph.edges[v1])-1)
    v2 = graph.edges[v1][i2]

    vertex_cover.add(v1)
    vertex_cover.add(v2)

    graph.clearVertex(v1)
    graph.clearVertex(v2)
    vertices = sorted(list(graph.vertices))

print(sorted(list(vertex_cover)))