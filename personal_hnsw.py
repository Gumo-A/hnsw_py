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

        self.time_measurements = defaultdict(list)
        
        self.distances_cache = {}

        self.ep = set([None])
        self.node_ids = set()
        
        self.M = M
        self.Mmax0 = M*2 if Mmax0 is None else Mmax0
        self.Mmax = round(self.M*1.5) if Mmax is None else Mmax
        self.mL = 1/np.log(M) if mL is None else mL
        self.efConstruction = self.Mmax0 if efConstruction is None else efConstruction
        
        self.layers = [nx.Graph() for _ in range(initial_layers)]
        self.angular = angular

        return None

    def save(self, path, save_distance_cache=False):
        index_data = {}
        
        index_data['node_ids'] = self.node_ids
        index_data['distances_cache'] = self.distances_cache if save_distance_cache else {}

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

        with open(path, 'wb') as file:
            pickle.dump(index_data, file)

        

    def load(self, path):
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
        self.distances_cache = index_data['distances_cache']

        for param_name, param in index_data['params'].items():
            self.__dict__[param_name] = param
        

    def print_parameters(self):
        for param, val in self.__dict__.items():
            if param in ['M', 'Mmax', 'Mmax0', 'mL', 'efConstruction', 'angular']:
                print(param, val)
        print('Nb. layers', len(self.layers))
    
    def add_vectors(
        self, 
        vectors: np.array, 
        vector_ids: list[int],
        checkpoint=False, 
        checkpoint_path='./index.hnsw',
        save_freq=1000
    ):

        # TODO:
        # setup methods to estimate recall at the current state of
        # the index construction. Set it up so that if a certain
        # threshold is not met we set higher values for M and/or efConstruction.

        if self.angular:
            vectors = self.normalize_vectors(vectors)

        if checkpoint:
            if os.path.exists(checkpoint_path):
                self.load(checkpoint_path)

            data_splited, num_splits = self.split_data(vectors, vector_ids, save_freq)

            for vectors, vector_ids in data_splited:
                self.insert_many(vectors, vector_ids)
                print(f'Saving current index in {checkpoint_path}')
                self.save(checkpoint_path)
        else:
            self.insert_many(vectors, vector_ids)
            

        return None

    def insert_many(self, vectors, vector_ids):
        print(f'Adding {vectors.shape[0]} vectors to HNSW efConstruction={self.efConstruction} M={self.M}')
        for vector, idx in tqdm(zip(vectors, vector_ids), total=len(vector_ids)):
            self.insert(vector, idx)
        
    def split_data(self, vectors, vector_ids, per_split):
        
        num_splits = vectors.shape[0] // per_split

        splits = []
        buffer = 0
        for i in range(num_splits):
            splits.append((
                vectors[buffer:buffer+per_split, :], 
                vector_ids[buffer:buffer+per_split]
            ))
            buffer += per_split

        return splits, num_splits
    
    def determine_layer(self):
        return math.floor(-np.log(np.random.random())*self.mL)
        
    def define_entrypoint(self):
        for layer_number in range(len(self.layers)-1, -1, -1):
            if self.layers[layer_number].order() != 0:
                self.ep = set([np.random.choice(self.layers[layer_number].nodes())])
                return None

    def ann_by_id(self, node_id: int):
        # TODO
        pass

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
    def insert(self, vector, node_id, payload: dict):

        if node_id in self.node_ids:
            return None
        else:
            self.node_ids.add(node_id)

        start_i = time.process_time()
        
        # start_p = time.process_time()
        
        L = len(self.layers) - 1
        l = math.floor(-np.log(np.random.random())*self.mL)

        new_ep = False
        if (l > L) or (self.ep is None):
            while l > L:
                self.layers.append(nx.Graph())
                L += 1
            new_ep = True

        ep = self.ep
        # end_p = time.process_time()
        # self.time_measurements['prelimilar'].append(end_p-start_p)


        # start_s1 = time.process_time()
        # step 1
        for layer_number in range(L, l, -1):

            layer = self.layers[layer_number]

            if layer.order() == 0:
                continue
            
            # start_s1s = time.process_time()
            W = self.search_layer(
                layer=layer,
                query=vector,
                entry_point=ep,
                ef=1,
                query_id=node_id,
            )
            # end_s1s = time.process_time()
            # self.time_measurements['step 1 search'].append(end_s1s-start_s1s)
            ep = set([self.get_nearest(
                layer, 
                W, 
                vector, 
                query_id=node_id
            )])
        # end_s1 = time.process_time()
        # self.time_measurements['step 1'].append(end_s1-start_s1)


        # start_s2 = time.process_time()
        # step 2
        for layer_number in range(l, -1, -1):

            layer = self.layers[layer_number]

            if layer.order() == 0:
                layer.add_node(
                    node_id, 
                    vector=vector,
                    **payload
                )
                continue

            layer.add_node(
                node_id, 
                vector=vector,
                **payload
            )

            start_s2s = time.process_time()
            ep = self.search_layer(
                layer=layer,
                query=vector,
                entry_point=ep,
                ef=self.efConstruction,
                query_id=node_id,
                step=2
            )
            end_s2s = time.process_time()
            self.time_measurements['step 2 search'].append(end_s2s-start_s2s)

            # start_s2h = time.process_time()
            neighbors_to_connect = self.select_neighbors_heuristic(
                layer=layer,
                inserted_node=node_id,
                candidates=ep,
                # extend_cands=True,
                keep_pruned=True
            )
            # end_s2h = time.process_time()
            # self.time_measurements['step 2 heuristic'].append(end_s2h-start_s2h)

            self.add_edges(
                layer=layer,
                node_id=node_id, 
                sorted_candidates=neighbors_to_connect
            )

            # start_s2p = time.process_time()
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

            # end_s2p = time.process_time()
            # self.time_measurements['step 2 prune'].append(end_s2p-start_s2p)

        # end_s2 = time.process_time()
        # self.time_measurements['step 2'].append(end_s2-start_s2)


        self.define_entrypoint()

        end_i  = time.process_time()
        self.time_measurements['insert'].append(end_i-start_i)

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
        sorted_candidates: set[int]
    ):
        edges = [(node_id, cand) for cand in sorted_candidates]
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
        step=1
    ) -> set:

        v = entry_point.copy()
        C = entry_point.copy()
        W = entry_point.copy()
        
        while len(C) > 0:
            # start_w = time.process_time()
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

            # end_w = time.process_time()
            # if step == 2: self.time_measurements['search s2w'].append(end_w-start_w)

            if cand_query_dist > f2q_dist :
                break # all element in W are evaluated 

            # start_f = time.process_time()
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

            # end_f = time.process_time()
            # if step == 2: self.time_measurements['search s2f'].append(end_f-start_f)

        return W

    def compute_distance(self, a, b, a_id=None, b_id=None, axis=None):
        
        cache = True if (a_id and b_id) else False

        if self.angular:
            distance = self.compute_distance_angular(a, b)
            if cache: self.distances_cache[(a_id, b_id)] = np.float16(distance)
            return distance
        else:
            distance = self.compute_distance_l2(a, b, axis)
            if cache: self.distances_cache[(a_id, b_id)] = np.float16(distance)
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
