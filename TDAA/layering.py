import os
import csv
import copy
import time
import random

from numpy import random
from ugraph import UndirectedGraph

def Layering(graph, weights, filename, perturb_percentage=None):
    '''
    This function inputs a graph, weights of vertices, name of file to write data into
    and an optional perturb percentage to write into the file
    '''
    vertex_cover = set({})
    n = len(graph.vertices)
    
    '''
    Layering :
    vertex_cover = []
    WHILE graph has edges:
      DO
        1. remove all zero-degree vertices
        2. calculate c = minimum of weight-degree ratio
        3. update weights of all vertices => w' = w - c.degree
        4. append all zero-weighted vertices into vertex_cover
    RETURN vertex_cover
    '''
    start = time.time()
    while(graph.edgesExist()):
        c = 100
        # Removing zero-degree vertices and calculating c
        for vertex in graph.vertices:
            degree = len(graph.edges[vertex])
            if degree == 0:
                graph.clearVertex(vertex)
                continue
            c = weights[vertex]/degree if weights[vertex]/degree < c else c
    
        # Updating weights
        for vertex in graph.vertices:
            degree = len(graph.edges[vertex])
            weights[vertex] -= min_c * degree
        
        # Appending to vertex_cover
        for vertex in weights.keys():
            if weights[vertex] == 0:
                vertex_cover.add(vertex)
                graph.clearVertex(vertex)
    end = time.time()

    vcset = list(sorted(vertex_cover))
    vc = len(vcset)
    time_taken = end - start

    with open(filename + '.csv', 'a') as file:
        writer = csv.writer(file)
        if perturb_percentage == None:
            writer.writerow([n, vc, time_taken])
        else:
            writer.writerow([n, vc, time_taken, perturb_percentage])
            
    print('n :', n)
    print('Vertex cover :', vc)
    print('Time taken :', end-start)
    print('')

# Driver code
# Remove files, if existent
dirlist = os.listdir(os.getcwd())
for item in dirlist:
    if 'csv' in item:
        os.remove(item)

'''
We shall consider three cases here, 
1. All weights are initialized as 1
2. All weights are picked independently and uniformly between 0.5 and 1
3. All weights are picked independently from positive half of standard normal distribution

We shall perturb them with 1/100 of a value picked randomly from positive half of standard normal dist.
For all the three, we shall consider n ranging from 50 to 2000, in steps of 50

That makes it 40 * 3 * 2 executions of the algorithm
'''
for n in range(50, 2001, 50):
    print(n)
    base_graph = UndirectedGraph()
    base_graph.randomAddEdges(n)
    # making copies for three cases
    graph_copies = [ copy.deepcopy(base_graph) for i in range(3) ]

    
    # 0 - Equal, 1 - Uniform, 2 - Std.Normal
    for index, graph in enumerate(graph_copies):
        # making a copy of graph for a perturbed instance
        perturb_graph = copy.deepcopy(graph)

        weights = {}
        if index == 0:
            print('Weights initialized as 1')
            filename = 'equal'
            for vertex in graph.vertices:
                weights[vertex] = 1
        if index == 1:
            print('Weights picked independently and uniformly')
            filename = 'uniform'
            for vertex in graph.vertices:
                weights[vertex] = random.uniform(0.5, 1)
        if index == 2:
            print('Weights picked independently from standard normal distribution')
            filename = 'normal'
            for vertex in graph.vertices:
                weights[vertex] = abs(random.standard_normal())
        
        Layering(graph, weights, filename)

        # making a copy of weights for the perturbed instance
        perturb_weights = copy.deepcopy(weights)
        avg_perturb_percentage = 0.0

        for vertex in perturb_graph.vertices:
            perturb = abs(random.standard_normal()) / 100
            try:
                avg_perturb_percentage += perturb*100/perturb_weights[vertex]
            except:
                avg_perturb_percentage += perturb*100
            perturb_weights[vertex] += perturb

        avg_perturb_percentage /= len(perturb_graph.vertices)

        Layering(perturb_graph, perturb_weights, filename+'_perturbed', perturb_percentage)