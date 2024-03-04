import numpy as np
import sys
from multiprocessing import Pool
import multiprocessing 
from personal_hnsw import HNSW
from helpers.glove_helpers import (
    brute_force_parallel,
    load_glove,
    write_brute_force_nn,
    load_brute_force,
    split
)


if __name__ == '__main__':
    dim, limit = int(sys.argv[1]), int(sys.argv[2])

    embeddings = load_glove(dim=dim, limit=limit)

    num_processes = multiprocessing.cpu_count()

    splits, nb_per_split = split(embeddings, num_processes)
    splits = [(10, embeddings, split, limit, dim, nb_per_split, i) for i, split in enumerate(splits)]

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(brute_force_parallel, splits)

    nearest_neighbors = {}
    for result in results:
        for idx, neighbors in result.items():
            nearest_neighbors[idx] = neighbors

    true_nn = load_brute_force(dim, limit)

    n = np.random.randint(0, len(true_nn.keys()))
    test_neighbors = true_nn[n]
    parallel_neighbors = nearest_neighbors[n]
    print(test_neighbors)
    print(parallel_neighbors)

    write_brute_force_nn(nearest_neighbors, limit, dim)
