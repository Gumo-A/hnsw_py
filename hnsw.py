from itertools import batched
import threading

from collections import defaultdict
from tqdm import tqdm
import networkx as nx
import numpy as np
import pickle
import math
import time
import os


class HNSW:
    def __init__(
        self,
        M=16,
        Mmax=None,
        Mmax0=None,
        mL=None,
        efConstruction=None,
        angular=False,
        initial_layers=1
    ):

        self._lock = threading.Lock()

        self.time_measurements = defaultdict(list)
        
        self.distances_cache = {}

        self.ep = None
        self.node_ids = set()
        
        self.M = M
        self.Mmax0 = M*2 if Mmax0 is None else Mmax0
        self.Mmax = round(self.M*1.5) if Mmax is None else Mmax
        self.mL = 1/np.log(M) if mL is None else mL
        self.efConstruction = self.Mmax0 if efConstruction is None else efConstruction
        
        self.layers = [nx.Graph() for _ in range(initial_layers)]
        self.angular = angular

        return None

    # TODO use this func to parallelize
    def save(self, path=None, return_data=False):
        index_data = {}
        
        index_data['node_ids'] = self.node_ids

        params = {}
        for param_name in ['ep', 'M', 'Mmax', 'Mmax0', 'mL', 'efConstruction', 'angular']:
            params[param_name] = self.__dict__[param_name]
        index_data['params'] = params

        layers = []
        for layer in self.layers:
            layers.append(layer.edges())
        index_data['layers'] = layers

        nodes = []
        for layer in self.layers:
            nodes.append(layer.nodes(data=True))
        index_data['nodes'] = nodes

        if path:
            with open(path, 'wb') as file:
                pickle.dump(index_data, file)

        if return_data:
            return index_data

        

    # TODO use this func to parallelize
    def load(self, path: str = None, index_data: dict = None):

        # assert (path or index_data), 'Must provide one of "path" or "index_data"'
        # assert (path and (not index_data)) or (index_data and (not path)), 'Must provide "path" or "index_data", not both'
        
        if path:
            with open(path, 'rb') as file:
                index_data = pickle.load(file)

        self.layers = []
        for layer_number in range(len(index_data['layers'])):

            layer_edges = index_data['layers'][layer_number]
            layer_nodes = index_data['nodes'][layer_number]

            graph = nx.from_edgelist(layer_edges)
            graph.add_nodes_from(layer_nodes)

            self.layers.append(graph)

        self.node_ids = index_data['node_ids']

        for param_name, param in index_data['params'].items():
            self.__dict__[param_name] = param
        

    def print_parameters(self):
        for param, val in self.__dict__.items():
            if param in ['M', 'Mmax', 'Mmax0', 'mL', 'efConstruction', 'angular', 'ep']:
                print(param, val)
        print('Nb. layers', len(self.layers))
        print('Nb. nodes:', len(self.node_ids))
    
    def add_vectors(
        self, 
        vectors: np.array, 
        vector_ids: list[int],
        checkpoint=False, 
        checkpoint_path='./indices/checkpoints/checkpoint.hnsw',
        save_freq=1000,
        min_recall: float = None
    ):

        # TODO:
        # setup methods to estimate recall at the current state of
        # the index construction. Set it up so that if a certain
        # threshold is not met we set higher values for M and/or efConstruction.

        limit = vectors.shape[0]
        dim = vectors.shape[1]

        if self.angular:
            vectors = self.normalize_vectors(vectors)

        if checkpoint:
            if os.path.exists(checkpoint_path):
                self.load(checkpoint_path)

            data_splited, num_splits = self.split_data(vectors, vector_ids, save_freq)

            print(f'Adding vectors in batches of {save_freq}, with checkpoints every batch')
            for batch in tqdm(data_splited, desc='Total progress', ncols=0):
                print()
                print(f'fConstruction={self.efConstruction} M={self.M}')
                insertions = 0
                for vector, vector_id in tqdm(batch):
                    insertions += self.insert(vector, vector_id)

                if insertions > 0:
                    print()
                    print(f'Saving current index in {checkpoint_path}')
                    self.save(checkpoint_path)
                    self.save(checkpoint_path + '.copy')
        else:
            self.insert_many(vectors, vector_ids)

        return None

    def insert_many(self, data):
        for vector, idx in tqdm(data, total=len(data)):
            self.insert(vector, idx)
        
    def split_data(self, vectors, vector_ids, per_split):
        
        splits = [batch for batch in batched(zip(vectors, vector_ids), per_split)]
        num_splits = len(splits)

        return splits, num_splits
    
    def determine_layer(self):
        return math.floor(-np.log(np.random.random())*self.mL)
        
    def define_entrypoint(self, new_ep):
        if new_ep:
            for layer_number in range(len(self.layers)-1, -1, -1):
                if self.layers[layer_number].order() != 0:
                    self.ep = set([np.random.choice(self.layers[layer_number].nodes())])
                    return None

    def ann_by_id(self, node_id: int, n, ef):
        layer = self.layers[0]
        vector = layer._node[node_id]['vector']

        neighbors = self.search_layer(
            layer=layer,
            query=vector,
            entry_point=set([node_id]),
            ef=ef
        )

        neighbors = self.get_nearest(
            layer=layer,
            candidates=neighbors,
            query=vector,
            # this '+1' should be removed when
            # comparing to new vectors.
            # I put it here to not include the query node
            # in the results and keep the measures clean
            top=n
        )

        # same for the slice, I k=only need to return the whole list
        return neighbors[:]

    def ann_by_vector(self, vector, n, ef):

        if self.angular:
            vector = self.normalize_vectors(vector, single_vector=True)

        ep = self.ep
        L = len(self.layers) - 1

        for layer_number in range(L, -1, -1):

            layer = self.layers[layer_number]

            ep = self.search_layer(
                layer=layer,
                query=vector,
                entry_point=ep,
                ef=1,
            )

        neighbors = self.search_layer(
            layer=self.layers[0],
            query=vector,
            entry_point=ep,
            ef=ef
        )

        neighbors = self.get_nearest(
            layer=self.layers[0],
            candidates=neighbors,
            query=vector,
            # this '+1' should be removed when
            # comparing to new vectors.
            # I put it here to not include the query node
            # in the results and keep the measures clean
            top=n+1
        )

        # same for the slice, I k=only need to return the whole list
        return neighbors[1:]

    # TODO: add payload to filter
    def insert(self, vector, node_id, payload: dict = None):

        if node_id in self.node_ids:
            return 0

        if self.angular:
            vector = self.normalize_vectors(vector, single_vector=True)
        
        max_layer_nb = len(self.layers) - 1
        current_layer_nb = math.floor(-np.log(np.random.random())*self.mL)

        new_ep, max_layer_nb = self.define_new_layers(max_layer_nb, current_layer_nb)

        ep = self.ep
        with self._lock:
            ep = self.step_1(node_id, vector, payload, ep, max_layer_nb, current_layer_nb)

        with self._lock:
            self.step_2(node_id, vector, payload, ep, current_layer_nb)
            self.node_ids.add(node_id)
            self.define_entrypoint(new_ep)

        return 1

    def define_new_layers(self, max_layer_nb, current_layer_nb):
        new_ep = False
        if (current_layer_nb > max_layer_nb) or (self.ep is None):
            while current_layer_nb > max_layer_nb:
                with self._lock:
                    self.layers.append(nx.Graph())
                max_layer_nb += 1
            new_ep = True
        return new_ep, max_layer_nb

    def step_1(self, node_id, vector, payload, ep, max_layer_nb, current_layer_nb):
        for layer_number in range(max_layer_nb, current_layer_nb, -1):

            layer = self.layers[layer_number]

            if layer.order() == 0:
                continue
    
            W = self.search_layer(
                layer=layer,
                query=vector,
                entry_point=ep,
                ef=1,
                query_id=node_id,
            )
            ep = set([self.get_nearest(
                layer, 
                W, 
                vector, 
                query_id=node_id
            )])
        return ep

    def step_2(self, node_id, vector, payload, ep, current_layer_nb):
        for layer_number in range(current_layer_nb, -1, -1):

            layer = self.layers[layer_number]

            if layer.order() == 0:
                layer.add_node(
                    node_id, vector=vector,
                    # **payload
                )
                continue

            layer.add_node(
                node_id, vector=vector,
                # **payload
            )
            ep = self.search_layer(
                layer=layer,
                query=vector,
                entry_point=ep,
                ef=self.efConstruction,
                query_id=node_id,
            )
            neighbors_to_connect = self.select_neighbors_heuristic(
                layer=layer,
                inserted_node=node_id,
                candidates=ep,
                # extend_cands=True,
                keep_pruned=True
            )
            self.add_edges(
                layer=layer,
                node_id=node_id, 
                candidates=neighbors_to_connect
            )
            self.prune_connexions(layer_number, layer, neighbors_to_connect)

    def prune_connexions(self, layer_number, layer, neighbors_to_connect):

        for neighbor in neighbors_to_connect:
            if (
                ((layer_number > 0) and (layer.degree[neighbor] > self.Mmax)) or
                ((layer_number == 0) and (layer.degree[neighbor] > self.Mmax0))
            ):

                limit = self.Mmax if layer_number > 0 else self.Mmax0

                old_neighbors = list(layer.neighbors(neighbor))
                new_neighbors = self.select_neighbors_heuristic(
                    layer,
                    neighbor,
                    old_neighbors
                )
                layer.remove_edges_from([(neighbor, old) for old in old_neighbors])
                self.add_edges(
                    layer,
                    neighbor,
                    list(new_neighbors)[:limit]
                )

    def select_neighbors_simple(
        self, 
        layer_number: int, 
        inserted_node: int, 
        candidates: set[int],
        n=None
    ):

        distances = []
        for candidate in candidates:
            candidate_vector = self.layers[layer_number] \
                                ._node[candidate]['vector'] 

            if isinstance(inserted_node, int):
                inserted_vector = self.layers[layer_number] \
                                    ._node[inserted_node]['vector']
                distances.append(
                    self.get_distance(
                        a=candidate_vector, 
                        b=inserted_vector,
                        a_id=candidate,
                        b_id=inserted_node
                    )
                )

            else:
                inserted_vector = inserted_node
                distances.append(
                    self.get_distance(
                        a=candidate_vector, 
                        b=inserted_vector,
                        # a_id=candidate,
                        # b_id=inserted_node
                    )
                )
                

        if n is None:
            top_to_return = self.M
        else:
            top_to_return = n

        return sorted(
            list(
                zip(candidates, distances)
            ), 
            key=lambda x: x[1]
        )[:top_to_return]



    def select_neighbors_heuristic(
        self, 
        layer, 
        inserted_node: int, 
        candidates: set[int],
        extend_cands: bool = False,
        keep_pruned: bool = True
    ):

        inserted_vector = layer._node[inserted_node]['vector']
        R = set()
        W = candidates.copy()
        
        if extend_cands:
            for candidate in candidates:
                for cand_neighbor in layer.neighbors(candidate):
                    if cand_neighbor != inserted_node:
                        W.add(cand_neighbor)

        W_d = set()
        while (len(W) > 0) and (len(R) < self.M):
            e, dist_e = self.get_nearest(
                layer, 
                W, 
                inserted_vector, 
                query_id=inserted_node, 
                return_distance=True
            )
            W.remove(e)

            if (len(R) == 0):
                R.add(e)
                continue

            e_vector = layer._node[e]['vector']
            nearest_from_r, dist_from_r = self.get_nearest(
                layer, 
                R, 
                e_vector, 
                query_id=e,
                return_distance=True
            )
            if dist_e < dist_from_r:
                R.add(e)
            else:
                W_d.add(e)

        if keep_pruned:
            while (len(W_d) > 0) and (len(R) < self.M):
                e, dist_e = self.get_nearest(
                    layer, 
                    W_d, 
                    inserted_vector, 
                    query_id=inserted_node,
                    return_distance=True
                )
                W_d.remove(e)
                R.add(e)

        return R
        

    def add_edges(
        self, 
        layer,
        node_id: int, 
        candidates: set[int]
    ):
        edges = [(node_id, cand) for cand in candidates]
        layer.add_edges_from(edges)
        
        # for candidate, distance in sorted_candidates:
        #     layer.add_edge(
        #         node_id,
        #         candidate,
        #         distance=distance
        #     )

    
    def get_nearest(
        self, 
        layer, 
        candidates: set[int], 
        query: np.array,
        query_id: int = None,
        return_distance=False,
        top=1
    ):

        cands_dist = []
        for candidate in candidates:
            distance = self.get_distance(
                a=layer._node[candidate]['vector'], 
                b=query,
                a_id=candidate,
                b_id=query_id
            )
            cands_dist.append((candidate, distance))

        if top == 1:
            cands_dist = min(cands_dist, key=lambda x: x[1])
            return cands_dist if return_distance else cands_dist[0]
        else:
            cands_dist = sorted(cands_dist, key=lambda x: x[1])[:top]
            return cands_dist if return_distance else list(map(lambda x: x[0], cands_dist)) 


    def get_furthest(
        self, 
        layer, 
        candidates: set, 
        query: np.array,
        query_id: int,
        return_distance=False
    ):

        distances = []
        for candidate in candidates:
            vector = layer._node[candidate]['vector'] 
            distances.append(
                self.get_distance(
                    a=vector, 
                    b=query,
                    a_id=candidate,
                    b_id=query_id
                )
            )

        furthest = list(zip(candidates, distances))
        furthest = max(furthest, key=lambda x: x[1])

        return furthest if return_distance else furthest[0]

    def search_layer(
        self, 
        layer, 
        query: np.ndarray, 
        entry_point: set[int], 
        ef: int,
        query_id: int = None,
    ) -> set:

        v = entry_point.copy()
        C = entry_point.copy()
        W = entry_point.copy()
        
        while len(C) > 0:
            c, cand_query_dist = self.get_nearest(
                layer, 
                C, 
                query, 
                query_id=query_id,
                return_distance=True
            )
            C.remove(c)
            f, f2q_dist  = self.get_furthest(
                layer, 
                W, 
                query,
                query_id=query_id,
                return_distance=True
            )

            if cand_query_dist > f2q_dist : break # all element in W are evaluated 

            for neighbor in layer.neighbors(c):
                if neighbor not in v:
                    v.add(neighbor)
                    f, f2q_dist  = self.get_furthest(
                        layer, 
                        W, 
                        query,
                        query_id=query_id,
                        return_distance=True
                    )

                    n2q_dist = self.get_distance(
                        a=layer._node[neighbor]['vector'],
                        b=query,
                        a_id=neighbor,
                        b_id=query_id
                    )

                    if (n2q_dist < f2q_dist ) or (len(W) < ef):

                        C.add(neighbor)
                        W.add(neighbor)

                        if len(W) > ef:
                            W.remove(f)

        return W

    def compute_distance(self, a, b, a_id=None, b_id=None, axis=None):
        
        cache = True if (a_id and b_id) else False

        if self.angular:
            distance = self.compute_distance_angular(a, b)
            if cache: 
                self.distances_cache[(a_id, b_id)] = np.float16(distance)
            return distance
        else:
            distance = self.compute_distance_l2(a, b, axis)
            if cache: 
                self.distances_cache[(a_id, b_id)] = np.float16(distance)
            return distance

    def get_distance(self, a, b, a_id=None, b_id=None, axis=None):

        if (a_id is None) or (b_id is None):
            return self.compute_distance(
                a, b, 
                axis=axis
            )

        a_node = min([a_id, b_id])
        b_node = max([a_id, b_id])

        distance = self.distances_cache.get((a_node, b_node), False)
        if distance:
            return distance
        else:
            return self.compute_distance(
                a, b,
                a_node, b_node,
                axis=axis
            )

    def compute_distance_angular(self, a, b):
        return 1 - np.dot(a, b.T)

    def compute_distance_l2(self, a, b, axis=None):
        return np.linalg.norm(a-b, axis=axis)

    def normalize_vectors(self, vectors, single_vector=False):
        if single_vector:
            return vectors/np.linalg.norm(vectors)
        else:
            norm = np.linalg.norm(vectors, axis=1)
            norm = np.expand_dims(norm, axis=1)
            return vectors/norm

    def get_measures(self, nearest_to_queries, nearest_to_queries_ann):

        measures = defaultdict(list)
        for node, neighbors in nearest_to_queries_ann.items():
            true_nns = list(map(lambda x: x[0], nearest_to_queries[node]))
            for ann in neighbors:
                measures['recall@10'].append(ann in true_nns[:len(neighbors)])

        return np.array(measures['recall@10'])

    def load_brute_force(self, dim, limit, name_append=''):

        path = f'/home/gamal/glove_dataset/brute_force/lim_{limit}_dim_{dim}{name_append}'
        print(path)
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data

    def load_glove(self, dim, limit, include_words=False):
        embeddings = []
        with open(f'/home/gamal/glove_dataset/glove.6B.{dim}d.txt', 'r') as file:
            c = 0
            words = []
            for line in tqdm(file, total=limit, desc='Loading embeddings'):
                line = line.strip().split(' ')
                word, emb = line[0], line[1:]
                emb = list(map(float, emb))
                embeddings.append(emb)
                words.append(word)
                c += 1
                if c >= limit:
                    break

        return (np.array(embeddings), words) if include_words else np.array(embeddings)

    
    def brute_force_return(
        self,
        n: int, 
        embeddings: np.array, 
    ):

        nearest_neighbors = {}
        for idx in tqdm(range(embeddings.shape[0]), total=embeddings.shape[0]):

            dists_vector = self.get_distance(embeddings[idx], embeddings, axis=1)
            dists_vector = [(jdx, dist) for jdx, dist in enumerate(dists_vector)]

            dists_vector = sorted(
                dists_vector,
                key=lambda x: x[1]
            )[1:n+1]

            nearest_neighbors[idx] = dists_vector

        return nearest_neighbors
