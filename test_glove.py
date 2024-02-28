import sys
import numpy as np
from personal_hnsw import HNSW
from helpers.glove_helpers import (
    load_glove,
    load_brute_force,
    get_distance,
    get_measures,
    ann
)

if __name__ == '__main__':

    dim, limit = int(sys.argv[1]), int(sys.argv[2])

    bruteforce_data = load_brute_force(dim=dim, limit=limit)
    embeddings, words = load_glove(dim=dim, limit=limit, include_words=True)

    index = HNSW(M=5, Mmax=10, mL=2, efConstruction=5)
    index.build_index(embeddings)
    index.clean_layers()

    anns, elapsed_time = ann(index, embeddings)

    measures = get_measures(bruteforce_data, anns)
    print(measures.mean())

    print(index.get_average_degrees())
