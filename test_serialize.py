import sys
import pickle
import numpy as np
import networkx as nx
from hnsw import HNSW
from helpers.glove_helpers import (
    load_glove,
    brute_force_return,
    load_brute_force,
    get_distance,
    get_measures,
    ann
)

# np.random.seed(0)

if __name__ == '__main__':

    dim, limit, angular, M = int(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3])), int(sys.argv[4])

    index = HNSW()
    index.load(f'./indices/lim{limit}_dim{dim}_angular_{angular}_M{M}.hnsw')
    index.print_parameters()

    embeddings, words = load_glove(dim=dim, limit=limit, include_words=True)
    # brute_force_data = load_brute_force(dim, limit, f'_angular_{angular}')
    # sample_indices = np.random.randint(0, embeddings.shape[0], 1000)

    for i in range(10):
        n = np.random.randint(0, embeddings.shape[0])
        word = words[n]
        anns = index.ann_by_vector(embeddings[n, :], 10, 36)
        print(word)
        print([words[i] for i in anns])
    
    n = 1000
    brute_force_data = brute_force_return(n=n, embeddings=embeddings, sample_size=100, angular=angular)
    sample_indices = np.array(list(brute_force_data.keys()))


    ef = 36
    print(f'Finding ANNs with ef={ef}')
    anns = ann(index, embeddings[sample_indices, :], sample_indices, n=n, ef=ef)
    measures = get_measures(brute_force_data, anns)
    print('Recall@10:', round(measures.mean(), 5))

