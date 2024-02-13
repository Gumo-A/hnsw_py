from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import math
from personal_hnsw import HNSW
import numpy as np
import time

index = HNSW(
    mL=0.75, 
    layers=4,
    efConstruction=25
)
n = 3000
dim = 12

np.random.seed(seed=1)
sample = np.random.random((n, dim))
queries = np.random.random((10, dim))

nearest_to_queries = {}
for idx, query in enumerate(queries):
    distances = []
    for jdx, vector in enumerate(sample):
        distances.append((jdx, index.get_distance(query, vector)))

    nearest_to_queries[idx] = sorted(distances, key=lambda x: x[1])[0]

for key, value in nearest_to_queries.items():
    print(f'Nearest to query {key} is {value}')


print(f'adding {n} vectors to HNSW')
for idx, vector in tqdm(enumerate(sample), total=sample.shape[0]):
    index.insert(vector)

for idx, query in enumerate(queries):
    print(f'ANN of query {idx}', index.ann_by_vector(query, 4))

# for idx, layer in enumerate(index.layers):
#     nx.draw(layer)
#     plt.show()
#     print('There are', layer.order(), 'nodes in the layer', idx)

