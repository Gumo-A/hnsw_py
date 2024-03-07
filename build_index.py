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


if __name__ == '__main__':

    dim, limit, angular = int(sys.argv[1]), int(sys.argv[2]), bool(sys.argv[3])

    bruteforce_data = load_brute_force(dim=dim, limit=limit, name_append=f'_angular_{angular}')
    embeddings, words = load_glove(dim=dim, limit=limit, include_words=True)

    index = HNSW(
        M=24, 
    )
    print('Parameters:')
    index.print_parameters()

    index.build_index(embeddings)

    sample_size = 100
    sample_indices = np.random.randint(0, embeddings.shape[0], sample_size)

    ef = 20

    print(f'Finding ANNs with ef={ef}')
    anns, elapsed_time = ann(index, embeddings[sample_indices, :], sample_indices, ef=ef)
    measures = get_measures(bruteforce_data, anns)
    print(measures.mean())

    with open('./indices/test_index.hnsw', 'wb') as file:
        pickle.dump(index, file)

