import sys
import numpy as np
import multiprocessing 
from multiprocessing import Pool

from personal_hnsw import HNSW
from helpers.glove_helpers import (
    load_glove,
    parallel_nn,
    write_brute_force_nn,
)


if __name__ == '__main__':
    processes = int(sys.argv[3]) if len(sys.argv) > 4 else None
    dim, limit, angular = int(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3]))

    embeddings, words = load_glove(dim=dim, limit=limit, include_words=True)

    nearest_neighbors = parallel_nn(
        embeddings=embeddings, 
        limit=limit, 
        dim=dim, 
        processes=processes,
        angular=angular
    )

    sample_size = 10
    for _ in range(sample_size):
        n = np.random.randint(0, len(words))
        print(words[n])
        neighbors = []
        for neighbor, distance in nearest_neighbors[n]:
            neighbors.append(words[neighbor])
        print(neighbors)

    write_brute_force_nn(nearest_neighbors, limit, dim, name_append=f'_angular_{angular}')
