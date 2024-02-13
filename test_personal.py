import matplotlib.pyplot as plt
import networkx as nx
import math
from personal_hnsw import HNSW
import numpy as np

index = HNSW(mL=1)

np.random.seed(seed=1)
sample = np.random.random((1000, 3))
queries = np.random.random((3, 3))

nearest_to_queries = {}
for idx, query in enumerate(queries):
    distances = []
    for jdx, vector in enumerate(sample):
        distances.append((jdx, index.get_distance(query, vector)))

    nearest_to_queries[idx] = sorted(distances, key=lambda x: x[1])[0]

for key, value in nearest_to_queries.items():
    print(f'Nearest to query {key} is {value}')


for idx, vector in enumerate(sample):
    if idx > 100:
        break
    index.insert(vector)

print(len(index.layers))

for layer in index.layers:
    nx.draw(layer)
    plt.show()
