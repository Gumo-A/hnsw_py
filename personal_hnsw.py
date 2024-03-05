from tqdm import tqdm
import networkx as nx
import numpy as np
import math


class HNSW:
    def __init__(
        self,
        M=16,
        Mmax=None,
        Mmax0=None,
        mL=None,
        efConstruction=None,
        angular=False,
        initial_layers=8,
    ):
        self.distance_count = 0
        
        self.M = M
        self.Mmax0 = M*2 if Mmax0 is None else Mmax0
        self.Mmax = int(round(self.Mmax0*0.75)) if Mmax is None else Mmax
        self.mL = 1/np.log(M) if mL is None else mL
        self.efConstruction = self.Mmax0 if efConstruction is None else efConstruction

        self.current_vector_id = 0
        self.layers = [nx.Graph() for _ in range(initial_layers)]
        self.angular = angular

        return None
    
    def build_index(self, sample: np.array):

        if self.angular:
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
        
    def get_entrypoint(self):

        for layer_number in range(len(self.layers)-1, -1, -1):
            if len(self.layers[layer_number].nodes()) != 0:
                return set(
                    [
                        np.random.choice(
                            self.layers[layer_number].nodes()
                        )
                    ]
                )

        return None
        
    def ann_by_id(self, node_id: int):
        # TODO
        pass

    def ann_by_vector(self, vector, n, ef):

        if self.angular:
            vector = self.normalize_vectors(vector, single_vector=True)

        ep = self.get_entrypoint()
        L = len(self.layers) - 1

        for layer_number in range(L, -1, -1):
            ep = self.search_layer(
                layer_number=layer_number,
                query=vector,
                entry_point=ep,
                ef=ef
            )

        neighbors = self.select_neighbors_simple(
            layer_number=0,
            inserted_node=vector,
            candidates=ep,
            n=n+1
        )

        # neighbors = self.search_layer(
        #     layer_number=layer_number,
        #     query=vector,
        #     entry_point=ep,
        #     ef=ef
        # )

        return list(neighbors)[1:]

    def insert(self, vector, node_reinsert=None):

        node_id = self.current_vector_id if node_reinsert is None else node_reinsert
        
        ep = self.get_entrypoint()
        L = len(self.layers) - 1
        l = math.floor(-np.log(np.random.random())*self.mL)

        while l > L:
            self.layers.append(nx.Graph())
            L += 1

        # step 1
        for layer_number in range(L, l, -1):

            if self.layers[layer_number].order() == 0:
                continue
            
            W = self.search_layer(
                layer_number=layer_number,
                query=vector,
                entry_point=ep,
                ef=1
            )
            ep = self.get_nearest(layer_number, W, vector)


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

            ep = self.search_layer(
                layer_number=layer_number,
                query=vector,
                entry_point=ep,
                ef=self.efConstruction
            )

            neighbors_to_connect = self.select_neighbors_heuristic(
                layer_number=layer_number,
                inserted_node=node_id,
                candidates=ep,
                # extend_cands=True,
                keep_pruned=True
            )

            self.add_edges(
                layer=layer,
                node_id=node_id, 
                sorted_candidates=neighbors_to_connect
            )

            for neighbor, dist in neighbors_to_connect:
                if (
                    ((layer_number > 0) and (layer.degree[neighbor] > self.Mmax)) or
                    ((layer_number == 0) and (layer.degree[neighbor] > self.Mmax0))
                ):

                    limit = self.Mmax if layer_number > 0 else self.Mmax0

                    # old_neighbors = set(layer.neighbors(neighbor))

                    # new_neighbors = self.select_neighbors_heuristic(
                    #     layer_number,
                    #     neighbor,
                    #     old_neighbors,
                    #     extend_cands=True,
                    #     keep_pruned=True
                    # )

                    old_edges = list(layer.edges(neighbor, data=True))
                    old_edges = sorted(old_edges, key=lambda x: x[2]['distance'])
                    to_remove = old_edges[limit:]
                    to_remove = list(map(lambda x: (x[0], x[1]), to_remove))

                    # old_edges = [(neighbor, old) for old in old_neighbors]
                    layer.remove_edges_from(to_remove)

                    # for old in old_neighbors:
                    #     layer.remove_edge(neighbor, old)

                    # self.add_edges(
                    #     layer,
                    #     neighbor,
                    #     new_neighbors
                    # )


        if node_reinsert is None:
            self.current_vector_id += 1

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
            else:
                inserted_vector = inserted_node
                
            distances.append(
                self.get_distance_l2(candidate_vector, inserted_vector)
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
            e, dist_e = self.get_nearest(layer_number, W, inserted_vector, return_distance=True)
            W.remove(e)

            if (len(R) == 0):
                R.add((e, dist_e))
                continue

            e_vector = layer._node[e]['vector']
            nearest_from_r, dist_from_r = self.get_nearest(layer_number, [elem[0] for elem in R], e_vector, return_distance=True)
            if dist_e < dist_from_r:
                R.add((e, dist_e))
            else:
                W_d.add(e)

        if keep_pruned:
            while (len(W_d) > 0) and (len(R) < self.M):
                e, dist_e = self.get_nearest(layer_number, W_d, inserted_vector, return_distance=True)
                W_d.remove(e)
                R.add((e, dist_e))

        return R
        

    def add_edges(
        self, 
        layer,
        node_id: int, 
        sorted_candidates: set[int]
    ):
        edges = [(node_id, cand, {'distance': dist}) for cand, dist in sorted_candidates]
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
        return_distance=False
    ):
        """
            Gets the nearest element from the candidate list 
            to the query
        """

        layer = self.layers[layer_number]

        # assert isinstance(query, np.ndarray), query

        vectors = [layer._node[candidate]['vector'] for candidate in candidates]
        distances = [self.get_distance_l2(query, vector) for vector in vectors]

        # distances = []
        # for candidate in candidates:
        #     vector = layer._node[candidate]['vector'] 
        #     distances.append(self.get_distance(vector, query, ))

        cands_dist = list(zip(candidates, distances))
        cands_dist = min(cands_dist, key=lambda x: x[1])

        if return_distance:
            return cands_dist[0], cands_dist[1]
        else:
            return cands_dist[0]

    def get_furthest(
        self, 
        layer_number: int, 
        candidates: set, 
        query: np.array
    ):
        """
            Gets the furthest element from the candidate list to the query
        """

        layer = self.layers[layer_number]

        vectors = [layer._node[candidate]['vector'] for candidate in candidates]
        distances = [self.get_distance_l2(query, vector) for vector in vectors]

        # distances = []
        # for candidate in candidates:
        #     vector = layer._node[candidate]['vector'] 
        #     distances.append(self.get_distance(vector, query))

        furthest = list(zip(candidates, distances))
        furthest = max(furthest, key=lambda x: x[1])

        return furthest[0]


    def search_layer(
        self, 
        layer_number: int, 
        query: np.array, 
        entry_point: set[int], 
        ef: int
    ):
        v = set([entry_point]) if not isinstance(entry_point, set) \
                                else entry_point.copy()
        C = set([entry_point]) if not isinstance(entry_point, set) \
                                else entry_point.copy()
        W = set([entry_point]) if not isinstance(entry_point, set) \
                                else entry_point.copy()
        
        layer = self.layers[layer_number]

        while len(C) > 0:
            c = self.get_nearest(layer_number, C, query)
            C.remove(c)
            f = self.get_furthest(layer_number, W, query)

            cand_query_dist = self.get_distance_l2(
                layer._node[c]['vector'],
                query,
                
            )
            furthest_query_dist = self.get_distance_l2(
                layer._node[f]['vector'],
                query,
            ) 

            if cand_query_dist > furthest_query_dist:
                break # all element in W are evaluated 

            for neighbor in layer.neighbors(c):
                if neighbor not in v:
                    v.add(neighbor)
                    f = self.get_furthest(layer_number, W, query)

                    neighbor_query_dist = self.get_distance_l2(
                        layer._node[neighbor]['vector'],
                        query,
                    )
                    furthest_query_dist = self.get_distance_l2(
                        layer._node[f]['vector'],
                        query,
                    )

                    if (neighbor_query_dist < furthest_query_dist) \
                         or (len(W) < ef):

                        C.add(neighbor)
                        W.add(neighbor)
                        if len(W) > ef:
                            W.remove(f)

        return W

    def get_distance_angular(self, a, b):
        return 1 - np.dot(a, b)

    def get_distance_l2(self, a, b, b_matrix=False):
        self.distance_count += 1
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
