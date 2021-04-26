class UndirectedGraph:
    def __init__(self):
        print('Graph initiated')
        self.edges = {}
        self.vertices = set({})
    
    def __str__(self):
        for vertex in self.edges.keys():
            print(vertex, self.edges[vertex])
        return ''

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
