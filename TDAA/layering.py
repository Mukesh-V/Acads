import os
import csv
import copy
import time
import random

from numpy import random
from ugraph import UndirectedGraph

def Layering(graph, weights, filename):
    vertex_cover = set({})
    n = len(graph.vertices) - 1
    
    start = time.time()
    while(graph.edgesExist()):
        min_c = 100
        for vertex in graph.vertices:
            degree = len(graph.edges[vertex])
            if degree == 0:
                graph.clearVertex(vertex)
                continue
            min_c = weights[vertex]/degree if weights[vertex]/degree < min_c else min_c
    
        for vertex in graph.vertices:
            degree = len(graph.edges[vertex])
            weights[vertex] -= min_c * degree
        
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
        writer.writerow([n, vc, time_taken])

    print('n :', n)
    print('Vertex cover :', vc)
    print('Time taken :', end-start)
    print('')

dirlist = os.listdir(os.getcwd())
for item in dirlist:
    if 'csv' in item:
        os.remove(item)

for n in range(50, 2001, 50):
    print(n)
    base_graph = UndirectedGraph()
    base_graph.randomAddEdges(n)
    graph_copies = [ copy.deepcopy(base_graph) for i in range(3) ]

    for index, graph in enumerate(graph_copies):
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

        perturb_weights = copy.deepcopy(weights)
        for vertex in perturb_graph.vertices:
            perturb_weights[vertex] += abs(random.standard_normal()) / 100
        Layering(perturb_graph, perturb_weights, filename+'_perturbed')