from multiprocessing import Pool, Manager


def split(queries, num_splits):
    per_split = len(queries)// num_splits

    splits = []
    buffer = 0
    for i in range(num_splits):
        splits.append(queries[buffer:buffer+per_split])
        buffer += per_split

    return splits, per_split

def f(d, l):
    for v in l:
        d[v] = v

if __name__ == '__main__':
    with Manager() as manager:
        d = manager.dict()
        processes = 1
        vals = [i for i in range(100)]
        split_vals, per_split = split(vals, num_splits=processes)
        split_vals = [(d, s) for s in split_vals]
        with Pool(processes) as pool:
            pool.starmap(f, split_vals)
            
        print(d)
