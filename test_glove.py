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

    dim, limit = int(sys.argv[1]), int(sys.argv[2]) 

    bruteforce_data = load_brute_force(dim=dim, limit=limit)
    embeddings, words = load_glove(dim=dim, limit=limit, include_words=True)

    for M in [2**i for i in range(3, 6)]:
        index = HNSW(
            M=M, 
            # efConstruction=efConstruction,
            # angular=False
        )
        print('Parameters:')
        index.print_parameters()
        index.build_index(embeddings)

        # for layer in index.layers:
        #     for node in layer.nodes():
        #         if layer.degree(node) == 0:
        #             print('Friendless node', node)


        sample_size = 100
        for ef in [i for i in range(2, 35, 4)]:
            sample_indices = np.random.randint(0, embeddings.shape[0], sample_size)
            print(f'Finding ANNs with ef={ef}')
            anns, elapsed_time = ann(index, embeddings[sample_indices, :], sample_indices, ef=ef)
            measures = get_measures(bruteforce_data, anns)
            print(measures.mean())

    
        for key, val in index.time_measurements.items():
            print(key)
            print(np.array(val).mean())

    # for layer in index.layers:
    #     nx.draw(layer)
    #     plt.show()

    with open('./indices/test_index.hnsw', 'wb') as file:
        pickle.dump(index, file)
