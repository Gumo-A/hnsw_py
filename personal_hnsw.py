import networkx as nx
import numpy as np
import math

class HNSW:
    def __init__(
        self,
        M=2,
        mL=None,
        layers=4,
        efConstruction=5
    ):
        self.M = M
        self.mL = 1/np.log(M) if mL is None else mL
        self.current_vector_id = 0
        self.efConstruction = efConstruction
        self.layers = [nx.Graph() for _ in range(layers)]

    def clean_layers(self):
        """
            Removes all empty layers from the top of the
            layers stack
        """

        max_layer_to_keep = len(self.layers) - 1
        for idx in range(len(self.layers)):
            if self.layers[idx].order() == 0:
                max_layer_to_keep = min(max_layer_to_keep, idx)

        self.layers = self.layers[:max_layer_to_keep]

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
        
    def ann_by_vector(self, vector, n):
        ep = self.get_entrypoint()
        L = len(self.layers) - 1

        for layer_number in range(L, -1, -1):
            ep = self.search_layer(
                layer_number=layer_number,
                query=vector,
                entry_point=ep,
                ef=1
            )

        neighbors = self.select_neighbors(
            layer_number=0,
            node_id=vector,
            candidates=ep,
            n=n
        )
        return neighbors

    def insert(self, vector):

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
            if self.layers[layer_number].order() == 0:
                self.layers[layer_number].add_node(
                    self.current_vector_id, 
                    vector=vector
                )
                continue

            self.layers[layer_number].add_node(
                self.current_vector_id, 
                vector=vector
            )

            ep = self.search_layer(
                layer_number=layer_number,
                query=vector,
                entry_point=ep,
                ef=self.efConstruction
            )

            neighbors_to_connect = self.select_neighbors(
                layer_number=layer_number,
                node_id=self.current_vector_id,
                candidates=ep
            )
            
            self.add_edges_simple(
                layer_number=layer_number,
                node_id=self.current_vector_id, 
                sorted_candidates=neighbors_to_connect
            )

            # TODO: shrink connections if they exceed Mmax

        self.current_vector_id += 1


    def select_neighbors(
        self, 
        layer_number: int, 
        node_id: int, 
        candidates: set[int],
        n=None
    ):


        distances = []
        for candidate in candidates:
            candidate_vector = self.layers[layer_number] \
                                .nodes()[candidate]['vector'] 

            if isinstance(node_id, int):
                inserted_vector = self.layers[layer_number] \
                                    .nodes()[node_id]['vector']
            else:
                inserted_vector = node_id
                
            distances.append(
                self.get_distance(candidate_vector, inserted_vector)
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

    def add_edges_simple(
        self, 
        layer_number: int, 
        node_id: int, 
        sorted_candidates: set[int]
    ):
        
        for candidate, distance in sorted_candidates:
            self.layers[layer_number].add_edge(
                node_id,
                candidate,
                distance=distance
            )
            

    def add_edges_heuristic(
            self, 
            layer_number: int, 
            node_id: int, 
            sorted_candidates: set[int]
        ):
        # TODO
        pass


    def get_nearest(
        self, 
        layer_number: int, 
        candidates: set[int], 
        query: np.array
    ):
        """
            Gets the nearest element from the candidate list 
            to the query
        """

        distances = []
        layer = self.layers[layer_number]

        for candidate in candidates:
            vector = layer.nodes()[candidate]['vector'] 
            distances.append(self.get_distance(vector, query))

        cands_dists = list(zip(candidates, distances))
        cands_dists = sorted(cands_dists, key=lambda x: x[1])

        nearest_candidate = cands_dists[0][0]

        return nearest_candidate

    def get_furthest(
        self, 
        layer_number: int, 
        candidates: set, 
        query: np.array
    ):
        """
            Gets the furthest element from the candidate list to the query
        """

        distances = []
        for candidate in candidates:
            vector = self.layers[layer_number]\
                        .nodes()[candidate]['vector'] 
            distances.append(self.get_distance(vector, query))

        return sorted(
            list(
                zip(candidates, distances)
            ), 
            key=lambda x: x[1]
        )[-1][0]


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

            cand_query_dist = self.get_distance(
                layer.nodes()[c]['vector'],
                query
            )
            furthest_query_dist = self.get_distance(
                layer.nodes()[f]['vector'],
                query
            ) 

            if cand_query_dist > furthest_query_dist:
                break # all element in W are evaluated 

            for neighbor in layer.neighbors(c):
                if neighbor not in v:
                    v.add(neighbor)
                    f = self.get_furthest(layer_number, W, query)

                    neighbor_query_dist = self.get_distance(
                        layer.nodes()[neighbor]['vector'],
                        query
                    )
                    furthest_query_dist = self.get_distance(
                        layer.nodes()[f]['vector'],
                        query
                    )

                    if (neighbor_query_dist < furthest_query_dist) \
                         or (len(W) < ef):

                        C.add(neighbor)
                        W.add(neighbor)
                        if len(W) > ef:
                            W.remove(f)

        return W

    def get_distance(self, a, b, b_matrix=False):
        if not b_matrix:
            # return np.linalg.norm(a-b)
            return (((a - b)**2).sum())**0.5
        else:
            return np.linalg.norm(a-b, axis=1)

