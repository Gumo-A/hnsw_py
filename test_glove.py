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

    index = HNSW(
        M=28,
        # Mmax=16,
        # Mmax0=32,
        # mL=1,
        efConstruction=6
    )
    index.build_index(embeddings)

    print(index.get_average_degrees()) 

    index.reinsert_friendless_nodes()

    for layer in index.layers:
        for node in layer.nodes():
            if layer.degree(node) == 0:
                print('Friendless node', node)

    # for ef in [1, 3, 5, 7]:
    for ef in [32]:
        print(f'Finding ANNs with ef={ef}')
        anns, elapsed_time = ann(index, embeddings, ef=ef)
        measures = get_measures(bruteforce_data, anns)
        print(measures.mean())

    # for layer in index.layers:
    #     nx.draw(layer)
    #     plt.show()

