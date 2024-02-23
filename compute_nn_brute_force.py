import sys
from personal_hnsw import HNSW
from helpers.glove_helpers import (
    brute_force_nn, 
    load_glove,
)


if __name__ == '__main__':
    dim, limit = int(sys.argv[1]), int(sys.argv[2])

    embeddings = load_glove(dim=dim, limit=limit)
    brute_force_nn(
        n=10, 
        index=HNSW(), 
        embeddings=embeddings,
        limit=limit,
        dim=dim
    )
