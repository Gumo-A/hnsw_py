import sys
from personal_hnsw import HNSW
from helpers.glove_helpers import load_brute_force, load_glove


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
