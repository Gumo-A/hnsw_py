import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager
import multiprocessing
from hnsw import HNSW
from itertools import batched, zip_longest
import math
import sys
from helpers.glove_helpers import (
    load_glove,
    load_brute_force,
    get_distance,
    get_measures,
    ann
)


def split_data(vectors, vector_ids, processes):

    per_split = math.ceil(vectors.shape[0] / processes)

    return batched(zip(vectors, vector_ids), per_split), processes

def combine_data(index_data):
    
    max_layers = 1
    for process_idx, data_package in index_data.items():
        if len(data_package['layers']) > max_layers:
            max_layers = len(data_package['layers'])
            entry_point = data_package['params']['ep']

    new_data = {
        'node_ids': set(),
        'params': {},
        'layers': [[] for _ in range(max_layers)],
        'nodes': [[] for _ in range(max_layers)]
    }

    for process_idx, data_package in index_data.items():
        new_data['node_ids'] = new_data['node_ids'].union(data_package['node_ids'])

        for layer_number in range(len(data_package['layers'])):
            new_data['layers'][layer_number] = new_data['layers'][layer_number] + list(data_package['layers'][layer_number])
            new_data['nodes'][layer_number] = new_data['nodes'][layer_number] + list(data_package['nodes'][layer_number])

        for key, value in data_package['params'].items():
            new_data['params'][key] = value
        new_data['params']['ep'] = entry_point

    return new_data

def insert_parallel(index_data: dict, vector_data, process_number) -> HNSW:

    index = HNSW(angular=True)

    for idx, item in tqdm(enumerate(vector_data), disable=process_number!=0, total=len(vector_data)):

        index.insert(item[0], item[1])

        if ( (idx % 200 == 0) and (idx != 0) ) or ( idx == (len(vector_data) - 1) ):
            index_data[process_number] = index.save(return_data=True)

            new_data = combine_data(index_data)

            index.load(index_data=new_data)

    final_data = combine_data(index_data)

    return final_data

if __name__ == '__main__':

    with Manager() as manager:
        index_data = manager.dict()

        nb_processes = multiprocessing.cpu_count()

        dim, limit, angular = int(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3]))

        embeddings, words = load_glove(dim=dim, limit=limit, include_words=True)

        data_batched, nb_processes = split_data(embeddings, range(embeddings.shape[0]), nb_processes)

        process_numbers = [i for i in range(nb_processes)]

        data_batched = zip_longest([index_data], data_batched, process_numbers, fillvalue=index_data)

        with Pool(processes=nb_processes) as pool:
            results = pool.starmap(insert_parallel, data_batched)

    bruteforce_data = load_brute_force(dim=dim, limit=limit, name_append=f'_angular_{angular}')

    index = HNSW(angular=angular)
    index.load(index_data=results[-1])
    index.print_parameters()

    
    sample_size = 100
    for ef in [i for i in range(12, 37, 12)]:
        sample_indices = np.random.randint(0, embeddings.shape[0], sample_size)
        print(f'Finding ANNs with ef={ef}')
        anns = ann(index, embeddings[sample_indices, :], sample_indices, ef=ef)
        measures = get_measures(bruteforce_data, anns)
        print('Recall@10:', round(measures.mean(), 5))
