import numpy as np
from personal_hnsw import HNSW
from tqdm import tqdm
import pickle
import sys


def load_glove(dim=50, limit=None):
    if limit:
        total = limit
    else:
        total = 400_000

    embeddings = []
    with open(f'/home/gamal/glove_dataset/glove.6B.{dim}d.txt', 'r') as file:
        c = 0
        for line in tqdm(file, total=total, desc='Loading glove data'):
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

        # nearest_neighbors[idx] = [(-1, 9999.0),]
        # for jdx in range(0, embeddings.shape[0]):

        #     if jdx == idx:
        #         continue

        #     distance = index.get_distance(
        #         embeddings[idx],
        #         embeddings[jdx]
        #     )

        #     max_dist = (0, 0)
        #     for kdx, val in enumerate(nearest_neighbors[idx]):
        #         if val[1] > max_dist[1]:
        #             max_dist = (kdx, val[1])


        #     if len(nearest_neighbors[idx]) < n:
        #         nearest_neighbors[idx].append((jdx, distance))

        #     elif distance < max_dist[1]: 
        #         nearest_neighbors[idx].pop(max_dist[0])
        #         nearest_neighbors[idx].append((jdx, distance))

        # nearest_neighbors[idx] = sorted(
        #     nearest_neighbors[idx], 
        #     key=lambda x: x[1]
        # )

    path = f'/home/gamal/glove_dataset/brute_force/lim_{limit}_dim_{dim}'
    with open(path, 'wb') as file:
        pickle.dump(nearest_neighbors, file)

    return None


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
