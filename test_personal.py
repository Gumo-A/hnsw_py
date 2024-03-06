from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import math
from personal_hnsw import HNSW
import numpy as np
import time

n = 1000
dim = 50

np.random.seed(seed=1)
sample = np.random.random((n, dim))
queries = np.random.random((1000, dim))


def brute_force(sample, queries):
    index = HNSW()
    start = time.time()
    nearest_to_queries = {}
    for idx, query in enumerate(queries):
        distances = []
        for jdx, vector in enumerate(sample):
            distances.append((jdx, index.get_distance(query, vector, angular=True)))

        nearest_to_queries[idx] = sorted(distances, key=lambda x: x[1])[0]
    end = time.time()
    brute_force_time = round(end - start, 2)
    return nearest_to_queries, brute_force_time


def ann(index, sample):
    print(f'Adding {n} vectors to HNSW')
    for idx, vector in tqdm(enumerate(sample), total=sample.shape[0]):
        index.insert(vector)
    index.clean_layers()

    start = time.time()
    nearest_to_queries_ann = {}
    for idx, query in enumerate(queries):
        anns = index.ann_by_vector(query, 10)
        nearest_to_queries_ann[idx] = anns[0]
    end = time.time()
    ann_time = round(end - start, 2)

    return (
        nearest_to_queries_ann,
        ann_time
    )


def get_measures(nearest_to_queries, nearest_to_queries_ann):
    measures = defaultdict(list)
    for key, value in nearest_to_queries_ann.items():
        result = True if value[0] == nearest_to_queries[key][0] else False
        measures['acc@1'].append(result)

    measures['acc@1'] = np.array(measures['acc@1'])

    return measures


if __name__ == '__main__':
    nearest_to_queries, brute_force_time = brute_force(sample, queries)
    print('Elapsed time to brute force find NNs', brute_force_time)

    index = HNSW(
        M=10,
        layers=10,
        efConstruction=35
    )

    nearest_to_queries_ann, ann_time = ann(index, sample)
    print('Elapsed time to find ANNs', ann_time)

    measures = get_measures(nearest_to_queries, nearest_to_queries_ann)

    print(f'ACC@1 (mL = {index.mL}):', measures['acc@1'].mean())
    print('The index ended up with', len(index.layers), 'layers')

