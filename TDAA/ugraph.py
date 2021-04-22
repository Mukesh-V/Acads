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

    def removeEdge(self, a, b):
        try:
            self.edges[a].remove(b)
            self.edges[b].remove(a)
        except:
            print('Error')

    def edgesExist(self):
        flag = False
        for vertex in self.edges.keys():
            if len(self.edges[vertex]) > 0:
                flag = True
                break
        return flag
