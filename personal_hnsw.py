from collections import defaultdict
from tqdm import tqdm
import networkx as nx
import numpy as np
import pickle
import math
import time


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
        
        self.distance_count = 0
        self.cache_count = 0
        self.distances_cache = {}

        self.ep = None
        
        self.M = M
        self.Mmax0 = M*2 if Mmax0 is None else Mmax0
        self.Mmax = round(self.M*1.5) if Mmax is None else Mmax
        self.mL = 1/np.log(M) if mL is None else mL
        self.efConstruction = self.Mmax0 if efConstruction is None else efConstruction
        
        self.current_vector_id = 0
        self.layers = [nx.Graph() for _ in range(initial_layers)]
        self.angular = angular

        return None

    def save(self, path):
        index_data = {}
        
        index_data['current_id'] = self.current_vector_id
        index_data['distances_cache'] = self.distances_cache

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

        self.current_vector_id = index_data['current_id']
        self.distances_cache = index_data['distances_cache']

        for param_name, param in index_data['params'].items():
            self.__dict__[param_name] = param
        

    def print_parameters(self):
        for param, val in self.__dict__.items():
            if param in ['M', 'Mmax', 'Mmax0', 'mL', 'efConstruction', 'angular']:
                print(param, val)
        print('Nb. layers', len(self.layers))
    
    def build_index(self, sample: np.array):

        if self.angular:
        # if True:
            sample = self.normalize_vectors(sample)

        print(f'Adding {sample.shape[0]} vectors to HNSW')
        for vector in tqdm(sample, total=sample.shape[0]):
            self.insert(vector)

        self.clean_layers()

        return None

    def clean_layers(self):
        """
            Removes all empty layers from the top of the
            layers stack.
        """

        max_layer_to_keep = len(self.layers) - 1
        for idx in range(len(self.layers)):
            if self.layers[idx].order() == 0:
                max_layer_to_keep = min(max_layer_to_keep, idx)

        self.layers = self.layers[:max_layer_to_keep]

        return None

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
        # if True:
            vector = self.normalize_vectors(vector, single_vector=True)

        ep = self.ep
        L = len(self.layers) - 1

        for layer_number in range(L, -1, -1):
            ep = self.search_layer(
                layer_number=layer_number,
                query=vector,
                entry_point=ep,
                ef=1,
                query_id=None
            )

        # neighbors = self.select_neighbors_simple(
        #     layer_number=0,
        #     inserted_node=vector,
        #     candidates=ep,
        #     n=n+1
        # )

        neighbors = self.search_layer(
            layer_number=0,
            query=vector,
            entry_point=ep,
            ef=ef
        )

        neighbors = self.get_nearest(
            layer_number=0,
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

    def insert(self, vector, node_reinsert=None):

        start_i = time.process_time()
        
        # start_p = time.process_time()
        node_id = self.current_vector_id if node_reinsert is None else node_reinsert
        
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

            if self.layers[layer_number].order() == 0:
                continue
            
            # start_s1s = time.process_time()
            W = self.search_layer(
                layer_number=layer_number,
                query=vector,
                entry_point=ep,
                ef=1,
                query_id=self.current_vector_id,
            )
            # end_s1s = time.process_time()
            # self.time_measurements['step 1 search'].append(end_s1s-start_s1s)
            ep = self.get_nearest(
                layer_number, 
                W, 
                vector, 
                query_id=self.current_vector_id
            )
        # end_s1 = time.process_time()
        # self.time_measurements['step 1'].append(end_s1-start_s1)


        start_s2 = time.process_time()
        # step 2
        for layer_number in range(l, -1, -1):

            layer = self.layers[layer_number]

            if layer.order() == 0:
                layer.add_node(
                    node_id, 
                    vector=vector
                )
                continue

            layer.add_node(
                node_id, 
                vector=vector
            )

            start_s2s = time.process_time()
            ep = self.search_layer(
                layer_number=layer_number,
                query=vector,
                entry_point=ep,
                ef=self.efConstruction,
                query_id=self.current_vector_id,
                step=2
            )
            end_s2s = time.process_time()
            self.time_measurements['step 2 search'].append(end_s2s-start_s2s)

            start_s2h = time.process_time()
            neighbors_to_connect = self.select_neighbors_heuristic(
                layer_number=layer_number,
                inserted_node=node_id,
                candidates=ep,
                # extend_cands=True,
                keep_pruned=True
            )
            end_s2h = time.process_time()
            self.time_measurements['step 2 heuristic'].append(end_s2h-start_s2h)

            self.add_edges(
                layer=layer,
                node_id=node_id, 
                sorted_candidates=neighbors_to_connect
            )

            start_s2p = time.process_time()
            for neighbor in neighbors_to_connect:
                if (
                    ((layer_number > 0) and (layer.degree[neighbor] > self.Mmax)) or
                    ((layer_number == 0) and (layer.degree[neighbor] > self.Mmax0))
                ):

                    limit = self.Mmax if layer_number > 0 else self.Mmax0

                    old_neighbors = list(layer.neighbors(neighbor))
                    new_neighbors = self.select_neighbors_heuristic(
                        layer_number,
                        neighbor,
                        old_neighbors
                    )
                    layer.remove_edges_from([(neighbor, old) for old in old_neighbors])
                    self.add_edges(
                        layer,
                        neighbor,
                        list(new_neighbors)[:limit]
                    )

            end_s2p = time.process_time()
            self.time_measurements['step 2 prune'].append(end_s2p-start_s2p)

        end_s2 = time.process_time()
        self.time_measurements['step 2'].append(end_s2-start_s2)


        self.define_entrypoint()
        if node_reinsert is None:
            self.current_vector_id += 1

        end_i  = time.process_time()
        self.time_measurements['insert'].append(end_i-start_i)

    def get_friendless_nodes(self):
        friendless = []
        for layer in self.layers:
            for node in layer.nodes():
                if layer.degree[node] == 0:
                    friendless.append(node)
        return friendless

    def reinsert_friendless_nodes(self):

        friendless = self.get_friendless_nodes()
        for node in friendless:
            vector = self.layers[0]._node[node]['vector']
            self.insert(vector, node)

    def get_average_degrees(self):
        degrees = {}
        for idx, layer in enumerate(self.layers):
            layer_degrees = layer.degree()
            layer_degrees = list(map(lambda x: x[1], layer_degrees))
            degrees[idx] = np.array(layer_degrees).mean()
        return degrees

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
        layer_number: int, 
        inserted_node: int, 
        candidates: set[int],
        extend_cands: bool = False,
        keep_pruned: bool = True
    ):

        layer = self.layers[layer_number]
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
                layer_number, 
                W, 
                inserted_vector, 
                query_id=inserted_node, 
                return_distance=True
            )
            W.remove(e)

            if (len(R) == 0):
                # R.add((e, dist_e))
                R.add(e)
                continue

            e_vector = layer._node[e]['vector']
            nearest_from_r, dist_from_r = self.get_nearest(
                layer_number, 
                list(R), 
                # [elem[0] for elem in R], 
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
                    layer_number, 
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
        layer_number: int, 
        candidates: set[int], 
        query: np.array,
        query_id: int = None,
        return_distance=False,
        top=1
    ) -> list:

        """
            Gets the nearest element from the candidate list 
            to the query
        """

        layer = self.layers[layer_number]

        # assert isinstance(query, np.ndarray), query

        # vectors = [layer._node[candidate]['vector'] for candidate in candidates]
        # distances = [self.get_distance(query, vector) for vector in vectors]

        cands_dist = []
        for candidate in candidates:
            distance = self.get_distance(
                a=layer._node[candidate]['vector'], 
                b=query,
                a_id=candidate,
                b_id=query_id
            )
            cands_dist.append((candidate, distance))

        # cands_dist = list(zip(candidates, distances))

        if top == 1:
            cands_dist = min(cands_dist, key=lambda x: x[1])
            return cands_dist if return_distance else cands_dist[0]
        else:
            cands_dist = sorted(cands_dist, key=lambda x: x[1])[:top]
            return cands_dist if return_distance else list(map(lambda x: x[0], cands_dist)) 


    def get_furthest(
        self, 
        layer_number: int, 
        candidates: set, 
        query: np.array,
        query_id: int
    ):
        """
            Gets the furthest element from the candidate list to the query
        """

        layer = self.layers[layer_number]

        # vectors = [layer._node[candidate]['vector'] for candidate in candidates]
        # distances = [self.get_distance(query, vector) for vector in vectors]

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

        return furthest[0]


    def search_layer(
        self, 
        layer_number: int, 
        query: np.array, 
        entry_point: set[int], 
        ef: int,
        query_id: int = None,
        step=1
    ) -> set:

        v = set([entry_point]) if not isinstance(entry_point, set) \
                                else entry_point.copy()
        C = set([entry_point]) if not isinstance(entry_point, set) \
                                else entry_point.copy()
        W = set([entry_point]) if not isinstance(entry_point, set) \
                                else entry_point.copy()
        
        layer = self.layers[layer_number]

        while len(C) > 0:
            start_w = time.process_time()
            c = self.get_nearest(
                layer_number, 
                C, 
                query, 
                query_id=query_id
            )
            C.remove(c)
            f = self.get_furthest(
                layer_number, 
                W, 
                query,
                query_id=query_id
            )

            cand_query_dist = self.get_distance(
                a=layer._node[c]['vector'],
                b=query,
                a_id=c,
                b_id=query_id
            )
            furthest_query_dist = self.get_distance(
                a=layer._node[f]['vector'],
                b=query,
                a_id=f,
                b_id=query_id
            ) 
            end_w = time.process_time()
            if step == 2: self.time_measurements['search s2w'].append(end_w-start_w)

            if cand_query_dist > furthest_query_dist:
                break # all element in W are evaluated 

            start_f = time.process_time()
            for neighbor in layer.neighbors(c):
                if neighbor not in v:
                    v.add(neighbor)
                    f = self.get_furthest(
                        layer_number, 
                        W, 
                        query,
                        query_id=query_id
                    )

                    neighbor_query_dist = self.get_distance(
                        a=layer._node[neighbor]['vector'],
                        b=query,
                        a_id=neighbor,
                        b_id=query_id
                    )
                    furthest_query_dist = self.get_distance(
                        a=layer._node[f]['vector'],
                        b=query,
                        a_id=f,
                        b_id=query_id
                    )

                    if (neighbor_query_dist < furthest_query_dist) \
                         or (len(W) < ef):

                        C.add(neighbor)
                        W.add(neighbor)
                        if len(W) > ef:
                            W.remove(f)
            end_f = time.process_time()
            if step == 2: self.time_measurements['search s2f'].append(end_f-start_f)

        return W

    def compute_distance(self, a, b, a_id=None, b_id=None, b_matrix=False):
        
        self.distance_count += 1
        cache = True if (a_id and b_id) else False

        if self.angular:
            distance = self.compute_distance_angular(a, b)
            if cache: self.distances_cache[(a_id, b_id)] = distance
            return distance
        else:
            distance = self.compute_distance_l2(a, b, b_matrix)
            if cache: self.distances_cache[(a_id, b_id)] = distance
            return distance

    def get_distance(self, a, b, a_id=None, b_id=None, b_matrix=False):

        if (a_id is None) or (b_id is None):
            return self.compute_distance(
                a, b, 
                b_matrix=b_matrix
            )

        a_node = min([a_id, b_id])
        b_node = max([a_id, b_id])

        distance = self.distances_cache.get((a_node, b_node), False)
        if distance:
            self.cache_count += 1
            return distance
        else:
            return self.compute_distance(
                a, b,
                a_node, b_node,
                b_matrix=b_matrix
            )

    def compute_distance_angular(self, a, b):
        return 1 - np.dot(a, b.T)

    def compute_distance_l2(self, a, b, b_matrix=False):
        if not b_matrix:
            return np.linalg.norm(a-b)
        else:
            return np.linalg.norm(a-b, axis=1)

    def normalize_vectors(self, vectors, single_vector=False):
        if single_vector:
            return vectors/np.linalg.norm(vectors)
        else:
            norm = np.linalg.norm(vectors, axis=1)
            norm = np.expand_dims(norm, axis=1)
            return vectors/norm
