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
    dim, limit = int(sys.argv[1]), int(sys.argv[2])

    embeddings = load_glove(dim=dim, limit=limit)

    nearest_neighbors = parallel_nn(
        embeddings=embeddings, 
        limit=limit, 
        dim=dim, 
        processes=processes
    )

    write_brute_force_nn(nearest_neighbors, limit, dim, name_append='_parallel')
