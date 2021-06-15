import os
import csv
import copy
import time

from ugraph import UndirectedGraph
import random

def GreedyVC(graph):
  vertices = sorted(list(graph.vertices))
  n = len(vertices)
  vertex_cover = set({})

  '''
  Greedy :
    vertex_cover = []
    WHILE graph has edges:
      DO
        1. select an edge randomly from the graph
        2. remove all the edges incident on either of the endpoints
        3. append the selected endpoints to vertex_cover
    RETURN vertex_cover
  '''
  start = time.time()
  while(graph.edgesExist()):
      # We "select" an edge by randomly selecting a vertex and another from its neighbourhood. 
      i1 = random.randint(0, len(vertices)-1)
      v1 = vertices[i1]
      i2 = random.randint(0, len(graph.edges[v1])-1)
      v2 = graph.edges[v1][i2]

      vertex_cover.add(v1)
      vertex_cover.add(v2)

      graph.clearVertex(v1)
      graph.clearVertex(v2)
      vertices = sorted(list(graph.vertices))
  end = time.time()
  
  vc = len(vertex_cover)
  time_taken = end - start
  print('n :', n)
  print('Vertex cover :', vc)
  print('Time taken :', time_taken)
  print('')

  with open('greedy.csv', 'a') as file:
    writer = csv.writer(file)
    writer.writerow([n, vc, time_taken])

# Driver code
# Remove the file, in case it exists
try:
  os.remove('greedy.csv')
except:
  pass

for n in range(50, 2001, 50):
    graph = UndirectedGraph()
    graph.randomAddEdges(n)
    GreedyVC(graph)