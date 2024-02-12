import networkx as nx
import numpy as np
import math

class HNSW:
    def __init__(
        self,
        M=2,
        mL=0.25,
        layers=4,
        efConstruction=5
    ):
        self.M = M
        self.mL = mL
        self.current_vector_id = 0
        self.efConstruction = efConstruction
        self.layers = [nx.Graph() for _ in range(layers)]

        print('HNSW index initialized with', len(self.layers), 'layers.')

    def determine_layer(self):
        return math.floor(-np.log(np.random.random())*self.mL)
        
    def get_entrypoint(self):

        for layer_number, layer in enumerate(self.layers):
            if len(self.layers[layer_number].nodes()) != 0:
                return set([np.random.choice(self.layers[layer_number].nodes()) ])
        return None
        
    def insert(self, vector):

        W = set()
        ep = self.get_entrypoint()
        L = len(self.layers) - 1
        l = math.floor(-np.log(np.random.random())*self.mL)

        # step 1
        for layer_number in range(max(L, l), l, -1):

            if len(self.layers[layer_number].nodes()) == 0:
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
            if len(self.layers[layer_number]) == 0:
                self.layers[layer_number].add_node(self.current_vector_id, vector=vector)
                ep = set([self.current_vector_id])
                continue

            self.layers[layer_number].add_node(self.current_vector_id, vector=vector)

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
                neighbors=ep
            )

        self.current_vector_id += 1


    def select_neighbors(self, layer_number: int, node_id: int, candidates: set[int]):

        distances = []
        for candidate in candidates:
            vector = self.layers[layer_number].nodes()[candidate]['vector'] 
            distances.append(
                self.get_distance(
                    vector,
                    self.layers[layer_number].nodes()[node_id]['vector']
                )
            )
        return sorted(list(zip(candidates, distances)), key=lambda x: x[1])[:self.M]

    def add_edges_simple(self, layer_number: int, node_id: int, sorted_candidates: list[int]):
        
        for candidate, distance in sorted_candidates:
            self.layers[layer_number].add_edge(
                node_id,
                candidate,
                distance=distance
            )
            
    def get_nearest(self, layer_number: int, candidates: set, query: np.array):
        """
            Gets the nearest element from the candidate list to the query
        """

        distances = []
        for candidate in candidates:
            vector = self.layers[layer_number].nodes()[candidate]['vector'] 
            distances.append(self.get_distance(vector, query))

        return sorted(list(zip(candidates, distances)), key=lambda x: x[1])[0][0]


    def get_furthest(self, layer_number: int, candidates: set, query: np.array):
        """
            Gets the furthest element from the candidate list to the query
        """

        distances = []
        for candidate in candidates:
            vector = self.layers[layer_number].nodes()[candidate]['vector'] 
            distances.append(self.get_distance(vector, query))
        print(distances)
        print(candidates)
        return sorted(list(zip(candidates, distances)), key=lambda x: x[1])[-1][0]


    def search_layer(self, layer_number: int, query: np.array, entry_point: int, ef: int):
        
        v = set([entry_point]) if not isinstance(entry_point, set) else entry_point
        C = set([entry_point]) if not isinstance(entry_point, set) else entry_point
        W = set([entry_point]) if not isinstance(entry_point, set) else entry_point
        
        while len(C) > 0:
            c = self.get_nearest(layer_number, C, query)
            C.remove(c)
            f = self.get_furthest(layer_number, W, query)

            cand_query_dist = self.get_distance(
                self.layers[layer_number].nodes()[c]['vector'],
                query
            )
            furthest_query_dist = self.get_distance(
                self.layers[layer_number].nodes()[f]['vector'],
                query
            ) 

            if cand_query_dist > furthest_query_dist:
                break # all element in W are evaluated 

            for neighbor in self.layers[layer_number].neighbors(c):
                if neighbor not in v:
                    v.add(neighbor)
                    f = self.get_furthest(layer_number, W, query)

                    neighbor_query_dist = self.get_distance(
                        self.layers[layer_number].nodes()[neighbor]['vector'],
                        query
                    )
                    furthest_query_dist = self.get_distance(
                        self.layers[layer_number].nodes()[f]['vector'],
                        query
                    )

                    if (neighbor_query_dist < furthest_query_dist) or (len(W) < ef):
                        C.add(neighbor)
                        W.add(neighbor)
                        if len(W) > ef:
                            W.remove(f)

        return W

    def get_distance(self, u, v):
        return (((u - v)**2).sum())**0.5

