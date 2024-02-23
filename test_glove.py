import sys
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
    embeddings = load_glove(dim=dim, limit=limit)

    index = HNSW()
    anns, elapsed_time = ann(index, embeddings)
    print(list(anns.items())[:10])
    measures = get_measures(bruteforce_data, anns)
    print(measures.mean())


