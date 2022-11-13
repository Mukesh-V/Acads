import threading 
import time
import random
import argparse

import pydot

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--nodes", help="Number of nodes")
parser.add_argument("-m", "--events", help="Number of events")
parser.add_argument("-mo", "--mode", help="Mode of protocol - vanilla/isolate/partition")
parser.add_argument("-iid", "--iid", help="ID of node to isolate")
parser.add_argument("-st", "--start", help="Start time of special effects as Integer")
parser.add_argument("-pd", "--period", help="Period of special effects as Integer")
args = parser.parse_args()

class Event:
    def __init__(self, parent=None, selfparent=None, timestamp=None):
        self.id = random.randint(pow(10, 3), pow(10, 4))
        self.parent = parent
        self.selfparent = selfparent
        self.timestamp = timestamp % global_start
    
    def __repr__(self):
        return str(vars(self))

def protocol(id, state, receipt):
    selfparent = None
    while True:
        if not state[id] or receipt[id]:
            time.sleep(random.random())
            event = Event(receipt[id][0], selfparent, int(time.time()))
            state[id].append(event)

            op_parent = event.parent if event.parent else "NULL"
            op_selfparent = event.selfparent if event.selfparent else "NULL"
            ordered.append([id, event.id, op_parent, op_selfparent, event.timestamp])

            selfparent, recipient = state[id][-1].id, None
            receipt[id].pop(0)
            while True:
                recipient = random.randint(0, n-1)

                if mode == 'vanilla':
                    if recipient != id:
                        break

                elif mode == 'isolate':
                    if id != isolated_id or not (starting_timestamp <= event.timestamp <= ending_timestamp):
                        if recipient != id:
                            if recipient != isolated_id or not (starting_timestamp <= event.timestamp <= ending_timestamp):
                                break
                    else:
                        recipient = None 
                        break

                elif mode == 'partition':
                    if recipient != id:
                        if starting_timestamp <= event.timestamp <= ending_timestamp:
                            if  recipient%2 == id%2:  break
                        else: break

            if not recipient == None:   receipt[recipient].append(selfparent)
        
        time.sleep(1)
        if sum([len(state[x]) for x in range(n)]) > m:
            return

def hashgraph():
    state, receipt = {}, {}
    threads = []
    for i in range(n):
        receipt[i] = [None] 
        state[i] = []
        x = threading.Thread(target=protocol, args=(i, state, receipt))
        threads.append(x)
        x.start()

    for x in threads:
        x.join()

    return state

def dotgraph():
    graph = "digraph G { \n\trankdir=LR \n\tnewrank=true \n "

    first_events = []
    for node in state.keys():
        graph += "\tsubgraph cluster" + str(node)
        graph += "{ \n \t\t"

        first_events.append(state[node][0].id)
        events = ""
        for i in range(len(state[node])-1):
            events += str(state[node][i].id) + " -> " + str(state[node][i+1].id) + " [minlen=" + str(state[node][i+1].timestamp - state[node][i].timestamp) + "] "
        events += "\n\t}\n"
        graph += events 

    graph += "\t{rank=same; " + ";".join([str(x) for x in first_events]) + "}\n"

    for node in state.keys():
        events = "\t"
        for event in state[node]:
            if event.parent:
                events += str(event.parent) + " -> " + str(event.id) + " "

        graph += events + "\n"
    graph += "}"
        
    with open("op_{graphtype}.dot".format(graphtype=mode), 'w') as file:
        file.write(graph)

    dotgraph = pydot.graph_from_dot_data(graph)[0]
    dotgraph.write_png("op_{graphtype}.png".format(graphtype=mode))

n, m, mode, ordered = int(args.nodes), int(args.events), args.mode, []
global_start = int(time.time())
isolated_id = int(args.iid) if args.iid else None
if args.start and args.period:
    starting_timestamp = int(args.start)
    ending_timestamp = starting_timestamp + int(args.period)

state = hashgraph()
dotgraph()
with open("op_{graphtype}.txt".format(graphtype=mode), "w") as file:
    file.write("{num}\n".format(num=n))
    file.write("\n".join([" ".join([str(item) for item in x]) for x in ordered]))
    file.write("\nDone")