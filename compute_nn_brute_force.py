import sys
import numpy as np
import multiprocessing 
from multiprocessing import Pool

from personal_hnsw import HNSW
from helpers.glove_helpers import (
    brute_force_parallel,
    parallel_nn,
    load_glove,
    write_brute_force_nn,
    load_brute_force,
    split
)


if __name__ == '__main__':
    processes = int(sys.argv[3]) if len(sys.argv) > 4 else None
    dim, limit = int(sys.argv[1]), int(sys.argv[2])

    embeddings = load_glove(dim=dim, limit=limit)

    nearest_neighbors = parallel_nn(
        embeddings=embeddings, 
        limit=limit, 
        dim=dim, 
        processes=processes
    )

    true_nn = load_brute_force(dim, limit)

    truths = []
    for key in true_nn.keys():
        truths.append(key in nearest_neighbors.keys())
    assert all(truths)

    n = np.random.randint(0, len(true_nn.keys()))
    test_neighbors = true_nn[n]
    parallel_neighbors = nearest_neighbors[n]
    print(test_neighbors)
    print(parallel_neighbors)

    write_brute_force_nn(nearest_neighbors, limit, dim, name_append='_parallel')
