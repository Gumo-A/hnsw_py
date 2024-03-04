from collections import defaultdict
from tqdm import tqdm
import time
from personal_hnsw import HNSW
import numpy as np
import pickle


def ann(index, sample, ef=10):

    start = time.time()
    nearest_to_queries_ann = {}
    for idx, query in tqdm(enumerate(sample), desc='Finding ANNs', total=sample.shape[0]):
        anns = index.ann_by_vector(vector=query, n=10, ef=ef)
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
            if c >= total:
                break

    return (np.array(embeddings), words) if include_words else np.array(embeddings) 


def brute_force_nn(
        n: int, 
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
        for idx in tqdm(range(embeddings.shape[0]), total=embeddings.shape[0]):

            dists_vector = get_distance(embeddings[idx], embeddings, b_matrix=True)
            dists_vector = [(jdx, dist) for jdx, dist in enumerate(dists_vector)]

            dists_vector = sorted(
                dists_vector,
                key=lambda x: x[1]
            )[1:n+1]

            nearest_neighbors[idx] = dists_vector

    return None

def brute_force_parallel(
    n: int,
    all_emb: np.array,
    emb: np.array,
    limit=None,
    dim=50,
    per_split: int = None,
    split_nb: int = None
):
    
    if limit:
        total = limit
    else:
        total = 400_000

    to_add = split_nb*per_split
    nearest_neighbors = {}
    if split_nb == 0:
        for idx in tqdm(
            range(emb.shape[0]), 
            total=emb.shape[0],
        ):

            dists_vector = get_distance(emb[idx], all_emb, b_matrix=True)
            dists_vector = [(jdx, dist) for jdx, dist in enumerate(dists_vector)]

            dists_vector = sorted(
                dists_vector,
                key=lambda x: x[1]
            )[1:n+1]

            idx += to_add
            nearest_neighbors[idx] = dists_vector
    else:
        for idx in range(emb.shape[0]): 

                dists_vector = get_distance(emb[idx], all_emb, b_matrix=True)
                dists_vector = [(jdx, dist) for jdx, dist in enumerate(dists_vector)]

                dists_vector = sorted(
                    dists_vector,
                    key=lambda x: x[1]
                )[1:n+1]

                idx += to_add
                nearest_neighbors[idx] = dists_vector

    return nearest_neighbors
    

def write_brute_force_nn(nearest_neighbors: dict[list], limit, dim):
    path = f'/home/gamal/glove_dataset/brute_force/parallel_lim_{limit}_dim_{dim}'
    with open(path, 'wb') as file:
        pickle.dump(nearest_neighbors, file)


def get_distance(a, b, b_matrix=False):
    if not b_matrix:
        return np.linalg.norm(a-b)
    else:
        return np.linalg.norm(a-b, axis=1)

    
def get_measures(nearest_to_queries, nearest_to_queries_ann):

    measures = defaultdict(list)
    for key, value in nearest_to_queries_ann.items():
        true_nns = list(map(lambda x: x[0], nearest_to_queries[key]))
        for ann, dist in value:
            measures['recall@10'].append(ann in true_nns)

    return np.array(measures['recall@10'])


def split(queries, num_splits):
    per_split = queries.shape[0] // num_splits

    splits = []
    buffer =0
    for i in range(num_splits):
        splits.append(queries[buffer:buffer+per_split, :])
        buffer += per_split

        if i == (num_splits - 1): break

    splits[-1] = queries[buffer:, :]

    return splits, per_split


