import sys
import pickle
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


if __name__ == '__main__':

    dim, limit, angular, M = int(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3])), int(sys.argv[4])

    embeddings, words = load_glove(dim=dim, limit=limit, include_words=True)
    embeddings = embeddings.astype(np.float16)


    index = HNSW(M=M, angular=angular)
    index.add_vectors(
        embeddings, 
        range(embeddings.shape[0]), 
        checkpoint=True, 
        checkpoint_path=f'./indices/lim{limit}_dim{dim}_angular_{angular}_M{index.M}_checkpoint.hnsw',
        save_freq=10_000
    )

    index.save(f'./indices/lim{limit}_dim{dim}_angular_{angular}_M{M}.hnsw')
