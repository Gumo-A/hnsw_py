from tqdm import tqdm
from personal_hnsw import HNSW
import numpy as np
import pickle


def load_brute_force(dim, limit):
    path = f'/home/gamal/glove_dataset/brute_force/lim_{limit}_dim_{dim}'
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def load_glove(dim=50, limit=None):
    if limit:
        total = limit
    else:
        total = 400_000

    embeddings = []
    with open(f'/home/gamal/glove_dataset/glove.6B.{dim}d.txt', 'r') as file:
        c = 0
        for line in tqdm(file, total=total):
            line = line.strip().split(' ')[1:]
            line = list(map(float, line))
            embeddings.append(line)
            c += 1
            if c >= limit:
                break

    return np.array(embeddings)    

def brute_force_nn(
        n: int, 
        index: HNSW, 
        embeddings: np.array, 
        limit=None, 
        dim=50
    ):

    """
        Computes and stores in home dir the n nearest neighbors of each 
        element in the embeddings matrix.
    """


    if limit:
        total = limit
    else:
        total = 400_000

    nearest_neighbors = {}
    for idx in tqdm(range(embeddings.shape[0]), total=total, desc='Brute force finding NNs'):

        dists_vector = index.get_distance(embeddings[idx], embeddings, b_matrix=True)
        dists_vector = [(jdx, dist) for jdx, dist in enumerate(dists_vector)]

        dists_vector = sorted(
            dists_vector,
            key=lambda x: x[1]
        )[1:n+1]

        nearest_neighbors[idx] = dists_vector

    path = f'/home/gamal/glove_dataset/brute_force/lim_{limit}_dim_{dim}'
    with open(path, 'wb') as file:
        pickle.dump(nearest_neighbors, file)

    return None
