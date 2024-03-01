import networkx as nx
import threading

def task(graph, nodes, num):
    print(f"Task {num} started")
    for node in nodes:
        graph.add_node(node)
    print(f"Task {num} completed")

num_threads = 7
graph = nx.Graph()
nodes = [i for i in range(40000000)]

def split_nodes(nodes, threads: int):
    nodes_per_split = len(nodes) // threads

    splits = []
    buffer = 0
    for thread in range(threads):
        splits.append(nodes[buffer:buffer+nodes_per_split])

        if thread == (threads - 1):
            break

        buffer += nodes_per_split

    splits[-1] = nodes[buffer:]    

    return splits

threads = []
splits = split_nodes(nodes, num_threads)
for i in range(num_threads):
    print(len(splits[i]))
    thread = threading.Thread(target=task, args=(graph, splits[i], i))
    threads.append(thread)

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print("All threads completed")

print(graph)
