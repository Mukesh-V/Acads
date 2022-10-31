import threading 
import time
import random

import pydot

class Event:
    def __init__(self, parent=None, selfparent=None, timestamp=None):
        self.id = random.randint(pow(10, 6), pow(10, 7))
        self.parent = parent
        self.selfparent = selfparent
        self.timestamp = timestamp
    
    def __repr__(self):
        return str(vars(self))

def protocol(id, mode, state, receipt):
    selfparent = None
    while True:
        if not state[id] or receipt[id]:
            time.sleep(random.random())
            if mode == 'vanilla':
                state[id].append(Event(receipt[id][0], selfparent, int(time.time())))
                selfparent = state[id][-1].id
                receipt[id].pop(0)
                while True:
                    recipient = random.randint(0, n-1)
                    if recipient != id:
                        break
                receipt[recipient].append(selfparent)
        
        time.sleep(1)
        if sum([len(state[x]) for x in range(n)]) > m:
            return

def hashgraph(mode):
    state, receipt = {}, {}
    threads = []
    for i in range(n):
        receipt[i] = [None] 
        state[i] = []
        x = threading.Thread(target=protocol, args=(i, mode, state, receipt))
        threads.append(x)
        x.start()

    for x in threads:
        x.join()

    return state

def dotgraph(data, type):
    graph = "digraph G { \n\trankdir=LR \n\tnewrank=true \n "

    first_events = []
    for node in data.keys():
        graph += "\tsubgraph cluster" + str(node)
        graph += "{ \n \t\t"

        first_events.append(data[node][0].id)
        events = ""
        for i in range(len(data[node])-1):
            events += str(data[node][i].id) + " -> " + str(data[node][i+1].id) + " [minlen=" + str(data[node][i+1].timestamp - data[node][i].timestamp) + "] "
        events += "\n\t}\n"
        graph += events 

    graph += "\t{rank=same; " + ";".join([str(x) for x in first_events]) + "}\n"

    for node in data.keys():
        events = "\t"
        for event in data[node]:
            if event.parent:
                events += str(event.parent) + " -> " + str(event.id) + " "

        graph += events + "\n"
    graph += "}"
        
    with open("op_{graphtype}.dot".format(graphtype=type), 'w') as file:
        file.write(graph)

    dotgraph = pydot.graph_from_dot_data(graph)[0]
    dotgraph.write_png("op_{graphtype}.png".format(graphtype=type))

n, m = 5, 15
mode = "vanilla"
graph = hashgraph(mode)
dotgraph(graph, mode)