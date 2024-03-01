from multiprocessing import Pool
import multiprocessing 
import networkx as nx
import numpy as np
import time


def brute_force(sample, queries):
    for query in queries:
        for vector in sample:
            np.dot(query, vector)

def split(queries, num_splits):
    per_split = queries.shape[0] // num_splits

    splits = []
    buffer =0
    for i in range(num_splits):
        splits.append(queries[buffer:buffer+per_split, :])
        buffer += per_split

        if i == (num_splits - 1): break

    splits[-1] = queries[buffer:, :]

    return splits

if __name__ == '__main__':

    # We can further improve this by using the Manager object from 
    # multiprocessing to store the results of each process and then
    # merge them.
    # This will be useful for brute force computation of NNs and
    # for parallelization of the insertion process of the index.
    # Although I still don't know exactly how I am going to do that.
    np.random.seed(seed=1)

    n = 400000
    dim = 50
    sample = np.random.random((n, dim))
    queries = np.random.random((1000, dim))

    start = time.time()
    brute_force(sample, queries)
    end = time.time()

    print("Single thread process completed, elapsed time:", end - start)

    num_processes = multiprocessing.cpu_count()
    splits = split(queries, num_processes)
    splits = [(sample, split) for split in splits]
    start = time.time()
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(brute_force, splits)
    end = time.time()

    print("All threads completed, elapsed time:", end - start)



