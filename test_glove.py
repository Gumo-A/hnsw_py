import sys
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

    index = HNSW(M=34, angular=False)
    index.build_index(embeddings)

    print(index.get_average_degrees()) 
    for layer in index.layers:
        print(layer.order())

    for layer in index.layers:
        for node in layer.nodes():
            if layer.degree(node) == 0:
                print('Friendless node', node)


    sample_size = 2000
    for ef in [4, 8, 16, 32]:
        sample_indices = np.random.randint(0, embeddings.shape[0], sample_size)
        print(f'Finding ANNs with ef={ef}')
        anns, elapsed_time = ann(index, embeddings[sample_indices, :], sample_indices, ef=ef)
        measures = get_measures(bruteforce_data, anns)
        print(measures.mean())

    # for layer in index.layers:
    #     nx.draw(layer)
    #     plt.show()

