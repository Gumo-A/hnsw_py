from hnsw import HNSW
import numpy as np


def test_distance():
    a = np.array(
        [1., 0.]
    )

    b = np.array(
        [0., 1.]
    )

    c = np.array(
        [-1., 0.]
    )

    index = HNSW(angular=True)

    dist = round(index.get_distance(a, b), 6)
    assert dist == 1.0, f'{dist}, (a, b)'

    dist = round(index.get_distance(a, a), 6)
    assert dist == 0.0, f'{dist}, (a, a)'

    dist = round(index.get_distance(a, c), 6)
    assert dist == 2.0, f'{dist}, (a, c)'

    print('Cosine distance passed')

def test_normalize():
    a = np.random.random((10, 126))

    index = HNSW()

    a = index.normalize_vectors(a)
    for vector in a:
        assert round(np.dot(vector, vector), 6) == 1, 'Dot product is not one' 

    v = np.random.random(126)
    v = index.normalize_vectors(v, single_vector=True)
    assert round(np.dot(v, v), 6) == 1, 'Dot product is not one'

    print('Normalization passed')


if __name__ == '__main__':
    test_normalize()
    test_distance()
