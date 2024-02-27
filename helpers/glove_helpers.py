from collections import defaultdict
from tqdm import tqdm
import time
from personal_hnsw import HNSW
import numpy as np
import pickle


def ann(index, sample):

    start = time.time()
    nearest_to_queries_ann = {}
    for idx, query in tqdm(enumerate(sample), desc='Finding ANNs', total=sample.shape[0]):
        anns = index.ann_by_vector(query, 10)
        nearest_to_queries_ann[idx] = anns[:10]
    end = time.time()
    ann_time = round(end - start, 2)

    return (
        nearest_to_queries_ann,
        ann_time
    )

def load_brute_force(dim, limit):
    path = f'/home/gamal/glove_dataset/brute_force/lim_{limit}_dim_{dim}'
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def load_glove(dim=50, limit=None, include_words=False):
    if limit:
        total = limit
    else:
        total = 400_000

    embeddings = []
    with open(f'/home/gamal/glove_dataset/glove.6B.{dim}d.txt', 'r') as file:
        c = 0
        words = []
        for line in tqdm(file, total=total, desc='Loading embeddings'):
            line = line.strip().split(' ')
            word, emb = line[0], line[1:]
            emb = list(map(float, emb))
            embeddings.append(emb)
            words.append(word)
            c += 1
            if c >= limit:
                break

    return (np.array(embeddings), words) if include_words else np.array(embeddings) 

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

def get_distance(a, b, b_matrix=False):
    if not b_matrix:
        return np.linalg.norm(a-b)
    else:
        return np.linalg.norm(a-b, axis=1)
    
def get_measures(nearest_to_queries, nearest_to_queries_ann):

    measures = defaultdict(list)
    for key, value in nearest_to_queries_ann.items():
        measures['acc@1'].append(value[0] == nearest_to_queries[key][0])

    return np.array(measures['acc@1'])
