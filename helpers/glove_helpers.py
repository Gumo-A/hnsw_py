from collections import defaultdict
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import time
from hnsw import HNSW
import numpy as np
import pickle


def ann(index, sample, sample_indices, n=10, ef=10):

    nearest_to_queries_ann = {}
    for idx, query in tqdm(zip(sample_indices, sample), desc='Finding ANNs', total=sample.shape[0]):
        anns = index.ann_by_vector(vector=query, n=n, ef=ef)
        nearest_to_queries_ann[idx] = anns
    return nearest_to_queries_ann


def load_brute_force(dim, limit, name_append=''):

    path = f'/home/gamal/glove_dataset/brute_force/lim_{limit}_dim_{dim}{name_append}'
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


def brute_force_return(
    n: int, 
    embeddings: np.ndarray,
    sample_size: int, 
    angular: bool
):

    """
        Computes and stores in home dir the n nearest neighbors of each 
        element in the embeddings matrix.
    """

    sample_indices = np.random.choice(range(embeddings.shape[0]), sample_size, False)

    nearest_neighbors = {}
    for idx in tqdm(sample_indices, total=sample_size):

        dists_vector = get_distance(embeddings[idx], embeddings, angular=angular)
        dists_vector = [(jdx, dist) for jdx, dist in enumerate(dists_vector)]

        dists_vector = sorted(
            dists_vector,
            key=lambda x: x[1]
        )[1:n+1]

        nearest_neighbors[idx] = dists_vector

    return nearest_neighbors

def parallel_nn(n, embeddings, limit, dim, processes=None, angular=False):
    if angular:
        embeddings = normalize_vectors(embeddings)

    num_processes = processes if processes else multiprocessing.cpu_count()

    splits, nb_per_split = split(embeddings, num_processes)
    splits = [(n, embeddings, split, limit, dim, nb_per_split, i, angular) for i, split in enumerate(splits)]

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(brute_force_parallel, splits)

    nearest_neighbors = {}
    for result in results:
        for idx, neighbors in result.items():
            nearest_neighbors[idx] = neighbors

    return nearest_neighbors


def parallel_places(embeddings, processes=None, angular=False):
    num_processes = processes if processes else multiprocessing.cpu_count()

    splits, nb_per_split = split(embeddings, num_processes)
    splits = [(10, embeddings, split, nb_per_split, i) for i, split in enumerate(splits)]

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(brute_force_places, splits)

    nearest_neighbors = {}
    for result in results:
        for idx, neighbors in result.items():
            nearest_neighbors[idx] = neighbors

    return nearest_neighbors


def brute_force_parallel(
    n: int,
    all_emb: np.array,
    emb: np.array,
    limit=None,
    dim=50,
    per_split: int = None,
    split_nb: int = None,
    angular: bool = False
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

            dists_vector = get_distance(emb[idx], all_emb, b_matrix=True, angular=angular)
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

def brute_force_parallel_even(
    n: int,
    all_emb: np.array,
    emb: np.array,
    limit=None,
    dim=50,
    per_split: int = None,
    split_nb: int = None,
    angular: bool = False
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

            dists_vector = get_distance(emb[idx], all_emb, b_matrix=True, angular=angular)
            dists_vector = [(jdx, dist) for jdx, dist in enumerate(dists_vector)]
            dists_vector_even = []
            for jdx, dist in dists_vector:
                if jdx % 2 != 0:
                    continue
                dists_vector_even.append((jdx, dist))

            dists_vector_even = sorted(
                dists_vector_even,
                key=lambda x: x[1]
            )[1:n+1]

            idx += to_add
            nearest_neighbors[idx] = dists_vector_even
    else:
        for idx in range(emb.shape[0]): 

                dists_vector = get_distance(emb[idx], all_emb, b_matrix=True)
                dists_vector = [(jdx, dist) for jdx, dist in enumerate(dists_vector)]
                dists_vector_even = []
                for jdx, dist in dists_vector:
                    if jdx % 2 != 0:
                        continue
                    dists_vector_even.append((jdx, dist))

                dists_vector_even = sorted(
                    dists_vector_even,
                    key=lambda x: x[1]
                )[1:n+1]

                idx += to_add
                nearest_neighbors[idx] = dists_vector_even

    return nearest_neighbors
    
def brute_force_places(
    n: int,
    all_emb: np.array,
    emb: np.array,
    per_split: int = None,
    split_nb: int = None,
    angular: bool = False
):
    
    to_add = split_nb*per_split
    nearest_neighbors = {}
    if split_nb == 0:
        for idx in tqdm(
            range(emb.shape[0]), 
            total=emb.shape[0],
        ):

            dists_vector = get_distance(emb[idx], all_emb, b_matrix=True, angular=angular)
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

def write_brute_force_nn(nearest_neighbors: dict[list], limit, dim, name_append=''):
    path = f'/home/gamal/glove_dataset/brute_force/lim_{limit}_dim_{dim}{name_append}'
    with open(path, 'wb') as file:
        pickle.dump(nearest_neighbors, file)


def write_bf_places(nearest_neighbors: dict[list], name_append=''):
    path = f'/home/gamal/places_embeddings/bf_places{name_append}'
    with open(path, 'wb') as file:
        pickle.dump(nearest_neighbors, file)


def normalize_vectors(vectors, single_vector=False):
    if single_vector:
        return vectors/np.linalg.norm(vectors)
    else:
        norm = np.linalg.norm(vectors, axis=1)
        norm = np.expand_dims(norm, axis=1)
        return vectors/norm


def get_distance(a, b, b_matrix=False, angular=False):
    if angular:
        return 1 - np.dot(a, b.T)
    else:
        if not b_matrix:
            return np.linalg.norm(a-b)
        else:
            return np.linalg.norm(a-b, axis=1)

    
def get_measures(nearest_to_queries, nearest_to_queries_ann):

    measures = defaultdict(list)
    for node, neighbors_distances in nearest_to_queries_ann.items():
        true_nns = list(map(lambda x: x[0], nearest_to_queries[node]))
        for ann in neighbors_distances:
            measures['recall@10'].append(ann in true_nns[:len(neighbors_distances)])

    return np.array(measures['recall@10'])


def split(queries, num_splits):
    per_split = queries.shape[0] // num_splits

    splits = []
    buffer = 0
    for i in range(num_splits):
        splits.append(queries[buffer:buffer+per_split, :])
        buffer += per_split

    return splits, per_split


