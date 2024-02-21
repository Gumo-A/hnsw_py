from tqdm import tqdm
from personal_hnsw import HNSW
import numpy as np
import pickle
import sys

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



if __name__ == '__main__':

    dim, limit = int(sys.argv[1]), int(sys.argv[2])

    bruteforce_data = load_brute_force(dim=dim, limit=limit)
    embeddings = load_glove(dim=dim, limit=limit)

    test_val = 0
    test_emb = embeddings[test_val]
    index = HNSW()

    dists_vector = index.get_distance(test_emb, embeddings, b_matrix=True)
    dists_vector = [(idx, dist) for idx, dist in enumerate(dists_vector)]

    dists_vector = sorted(
        dists_vector,
        key=lambda x: x[1]
    )
    print(index.get_distance(embeddings[test_val], embeddings[dists_vector[1][0]]))
    print(bruteforce_data[test_val])
    print(dists_vector[1:11])
