import sys
import pickle
import numpy as np
import networkx as nx
from personal_hnsw import HNSW
import matplotlib.pyplot as plt
from helpers.glove_helpers import (
    brute_force_places,
    parallel_places,
    get_distance,
    get_measures,
    ann
)

n = 1000

np.random.seed(seed=1)

with open('/home/gamal/places_embeddings/places.pkl', 'rb') as file:
    places = pickle.load(file)['embeddings']

print(places.shape)
sample_idx = np.random.randint(0, places.shape[0])


if __name__ == '__main__':

    limit = int(sys.argv[1])

    bf_nn = parallel_places(places[:limit])

    index = HNSW(
        M=20,
    )
    index.build_index(places[:limit])

    nearest_to_queries_ann = ann(index, places[:limit], range(limit), ef=12)
    measures = get_measures(bf_nn, nearest_to_queries_ann)

    print(f'Recall@10:', measures.mean())

    print(index.layers[0].nodes(data=True)[0]['vector'].shape)

