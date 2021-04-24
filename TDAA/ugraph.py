class UndirectedGraph:
    def __init__(self):
        print('Graph initiated')
        self.edges = {}

    def addEdge(self, a, b):
        try:
            self.edges[a].append(b)
        except:
            self.edges[a] = [b]
        try:
            self.edges[b].append(a)
        except:
            self.edges[b] = [a]   

    def printGraph(self):
        for vertex in self.edges.keys():
            print(vertex, self.edges[vertex])
    
    def clearVertex(self, key):
        del self.edges[key]
        for vertex in self.edges.keys():
            if key in self.edges[vertex]:
                self.edges[vertex].remove(key)

    def edgesExist(self):
        flag = False
        for vertex in self.edges.keys():
            if len(self.edges[vertex]) > 0:
                flag = True
                break
        return flag
