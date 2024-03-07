import sys
import pickle
import numpy as np
import networkx as nx
from personal_hnsw import HNSW
import matplotlib.pyplot as plt
from helpers.glove_helpers import (
    load_glove,
    load_brute_force,
    get_distance,
    get_measures,
    ann
)

np.random.seed(0)

if __name__ == '__main__':

    dim, limit, angular = int(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3]))

    bruteforce_data = load_brute_force(dim=dim, limit=limit, name_append=f'_angular_{angular}')
    embeddings, words = load_glove(dim=dim, limit=limit, include_words=True)

    for M in [2**i for i in range(3, 6)]:
        index = HNSW(
            M=M, 
            # efConstruction=efConstruction,
            angular=angular
        )
        index.build_index(embeddings)

        sample_size = 100
        for ef in [i for i in range(12, 35, 4)]:
            sample_indices = np.random.randint(0, embeddings.shape[0], sample_size)
            print(f'Finding ANNs with ef={ef}')
            anns = ann(index, embeddings[sample_indices, :], sample_indices, ef=ef)
            measures = get_measures(bruteforce_data, anns)
            print('Recall@10:', round(measures.mean(), 5))

        print('Parameters:')
        index.print_parameters()
        time_measures = {}
        for key, val in index.time_measurements.items():
            time_measures[key] = round(np.array(val).mean(), 6)

        print('Time measurements:')
        print(time_measures)

