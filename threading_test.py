import sys
import math
import threading
from itertools import batched
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from hnsw import HNSW
from helpers.glove_helpers import (
    load_glove,
    load_brute_force,
    get_measures,
    ann
)

def split_data(vectors, vector_ids, processes):

    per_split = math.ceil(vectors.shape[0] / processes)

    return batched(zip(vectors, vector_ids), per_split)

if __name__ == '__main__':

    dim, limit, angular = int(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3]))

    bruteforce_data = load_brute_force(dim=dim, limit=limit, name_append=f'_angular_{angular}')
    embeddings, words = load_glove(dim=dim, limit=limit, include_words=True)

    embeddings = embeddings.astype(np.float16)

    index = HNSW(
        M=18,
        angular=angular
    )
    nb_threads = 2
    data = split_data(embeddings, range(embeddings.shape[0]), nb_threads)
    with ThreadPoolExecutor(max_workers=nb_threads) as executor:
        executor.map(index.insert_many, data)
    executor.shutdown(wait=True)

    index.print_parameters()

    ef = 12
    sample_size = 100
    sample_indices = np.random.randint(0, embeddings.shape[0], sample_size)
    print(f'Finding ANNs with ef={ef}')
    anns = ann(index, embeddings[sample_indices, :], sample_indices, ef=ef)
    measures = get_measures(bruteforce_data, anns)
    print('Recall@10:', round(measures.mean(), 5))

    n = np.random.randint(0, embeddings.shape[0])
    word = words[n]
    print(word)
    anns = index.ann_by_vector(embeddings[n], 10, 12)
    print([words[i] for i in anns])

    print('Parameters:')
    index.print_parameters()

