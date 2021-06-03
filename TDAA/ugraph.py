import copy
import random
import matplotlib.pyplot as plt

class UndirectedGraph:
    def __init__(self):
        print('Graph initiated')
        self.edges = {}
        self.locations = {}
        self.vertices = set({})
    
    def __str__(self):
        for vertex in self.edges.keys():
            print(vertex, self.edges[vertex])
        return ''

    def allot(self, a):
        self.vertex_copy = copy.deepcopy(self.vertices)
        self.edges_copy = copy.deepcopy(self.edges)
        for vertex in list(sorted(self.vertices)):
            x = random.randint(0, a)
            y = random.randint(0, a)
            self.locations[vertex] = (x, y)

    def plot(self, vertex_cover):
        vertex_copy = copy.deepcopy(self.vertex_copy)
        for vertex in vertex_cover:
            plt.scatter(self.locations[vertex][0], self.locations[vertex][1], s=30, color='red')
            x = [ self.locations[vertex][0] ]
            y = [ self.locations[vertex][1] ]
            for vertex2 in self.edges_copy[vertex]:
                x.append( self.locations[vertex2][0] )
                y.append( self.locations[vertex2][1] )
                plt.plot(x, y, color='grey', alpha=0.05)
                x = [ self.locations[vertex][0] ]
                y = [ self.locations[vertex][1] ]
            vertex_copy.remove(vertex)
        for vertex in vertex_copy:
            plt.scatter(self.locations[vertex][0], self.locations[vertex][1], s=20, color='aquamarine')

        plt.show()

    def randomAddEdges(self, n):
        while len(self.vertices) <= n:
            a = random.randint(0, n)
            b = random.randint(0, n)

            try:
                if a == b or b in self.edges[a]:
                    self.randomAddEdges(n)
                else:
                    self.addEdge(a, b)
            except:
                self.addEdge(a, b)

    def addEdge(self, a, b):
        try:
            self.edges[a].append(b)
        except:
            self.edges[a] = [b]
        try:
            self.edges[b].append(a)
        except:
            self.edges[b] = [a]   
        self.vertices.add(a)
        self.vertices.add(b)
    
    def clearVertex(self, key):
        try:
            del self.edges[key]
            self.vertices.remove(key)

            for vertex in list(self.edges):
                if key in self.edges[vertex]:
                    self.edges[vertex].remove(key)
                if len(self.edges[vertex]) == 0:
                    del self.edges[vertex]
                    self.vertices.remove(vertex)
        except:
            pass

    def edgesExist(self):
        flag = False
        for vertex in self.edges.keys():
            if len(self.edges[vertex]) > 0:
                flag = True
                break
        return flag