from ugraph import UndirectedGraph
import random

graph = UndirectedGraph()
graph.addEdge('a', 'b')
graph.addEdge('a', 'c')
graph.addEdge('b', 'd')
graph.addEdge('c', 'e')
graph.addEdge('d', 'f')
graph.addEdge('e', 'f')

graph.printGraph()
print('')

vertices = list(graph.edges.keys())
vertex_cover = set({})

while(graph.edgesExist()):
    i1 = random.randint(0, len(vertices)-1)
    v1 = vertices[i1]
    if len(graph.edges[v1]) == 0 or v1 in vertex_cover:
        continue
    i2 = random.randint(0, len(graph.edges[v1])-1)
    v2 = graph.edges[v1][i2]

    vertex_cover.add(v1)
    vertex_cover.add(v2)
    for v1i in v1set:
        graph.removeEdge(v1, v1i)
    for v2i in v2set:
        graph.removeEdge(v2, v2i)

print(sorted(list(vertex_cover)))