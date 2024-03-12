import sys
import pickle
import numpy as np
import networkx as nx
from personal_hnsw import HNSW
import matplotlib.pyplot as plt
from helpers.glove_helpers import (
    load_glove,
    load_brute_force,
    get_distance,
    get_measures,
    ann
)

np.random.seed(0)

if __name__ == '__main__':

    dim, limit, angular = int(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3]))

    bruteforce_data = load_brute_force(dim=dim, limit=limit, name_append=f'_angular_{angular}')
    embeddings, words = load_glove(dim=dim, limit=limit, include_words=True)

    embeddings = embeddings.astype(np.float16)

    # s2s_times = []
    for i in [i for i in range(12, 37)]:
        index = HNSW(
            M=i, 
            angular=angular
        )
        index.add_vectors(embeddings, range(embeddings.shape[0]))

        # s2s_times.append([index.efConstruction, np.array(index.time_measurements['step 2 search']).mean()])

        sample_size = 100
        for ef in [i for i in range(12, 37, 12)]:
            sample_indices = np.random.randint(0, embeddings.shape[0], sample_size)
            print(f'Finding ANNs with ef={ef}')
            anns = ann(index, embeddings[sample_indices, :], sample_indices, ef=ef)
            measures = get_measures(bruteforce_data, anns)
            print('Recall@10:', round(measures.mean(), 5))

            # print('ANN by id:')
            # node_id = np.random.randint(0, embeddings.shape[0])
            # print(f'"{words[node_id]}"', 'ANNs:')
            # anns = [words[i] for i in index.ann_by_id(node_id, 3, 24)]
            # print(anns)


        print('Parameters:')
        index.print_parameters()
        time_measures = {}
        for key, val in index.time_measurements.items():
            time_measures[key] = round(np.array(val).mean(), 6)

        print('Time measurements:')
        print('Step two\'s search is', round(time_measures['step 2 search']/time_measures['insert'], 4), 'of insertion time')

# s2s_times = np.array(s2s_times)
# plt.plot(s2s_times[:, 0], s2s_times[:, 1])
# plt.title('Step 2 search time as a function of efConstruction')
# plt.show()
