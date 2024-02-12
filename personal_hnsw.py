import networkx as nx
import numpy as np
import math

class HNSW:
    def __init__(
        self,
        mL=0.25,
    ):
        self.current_vector_id = 0
        self.mL = mL
        self.layer_numbers = set()
        self.layers = []
        self.efConstruction = 5

        for i in range(10000):
            level = self.determine_layer()
            for j in range(level+1):
                self.layer_numbers.add(j)

        for level in self.layer_numbers:
            self.layers.append(nx.Graph(level=level))

        print('HNSW index initialized with', max(self.layer_numbers), 'layers.')

    def determine_layer(self):
        return math.floor(-np.log(np.random.random())*self.mL)
        
    def insert(self, vector):

        currently_found_nearest = set()
        max_layer = max(self.layer_numbers)
        entry_point = np.random.choice(self.layers[max_layer].nodes()) if self.layers[max_layer].order() > 0 else None
        
        print('Inserting new vector')
        new_vector_layer = self.determine_layer()
        print('In layer', new_vector_layer)

        # step 1
        for layer_number in range(max(max_layer, new_vector_layer), new_vector_layer, -1):
            entry_point = self.get_nn_in_layer(
                layer_number=layer_number,
                query=vector,
                entry_point=entry_point,
                ef=1
            )

        # step 2
        for layer_number in range(new_vector_layer, -1, -1):
            entry_point = self.get_nn_in_layer(
                layer_number=layer_number,
                query=vector,
                entry_point=entry_point,
                ef=self.efConstruction
            )
            self.layers[layer_number].add_node(self.current_vector_id, vector=vector)

            print(layer_number)
            self.add_edges_simple(
                layer_number=layer_number,
                node_id=self.current_vector_id, 
                neighbors=entry_point
            )

            self.current_vector_id += 1

    def add_edges_simple(self, layer_number: int, node_id: int, neighbors: list[int]):

        distances = []
        for neighbor in neighbors:
            vector = self.layers[layer_number].nodes()[neighbor]['vector'] 
            distances.append(self.get_distance(vector, query))
        sorted_neighbors = sorted(list(zip(neighbors, distances)), key=lambda x: x[1])

        for addition in range(self.M):
            self.layers[layer_number].add_edge(
                node_id,
                sorted_neighbors[addition][0]
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
        return sorted(list(zip(candidates, distances)), key=lambda x: x[1])[-1][0]


    def get_nn_in_layer(self, layer_number: int, query: np.array, entry_point: int, ef: int):

        if entry_point is None:
            return None
        
        visited_elements = set([entry_point])
        candidates = set([entry_point])
        found_nn = set([entry_point])
        
        while len(candidates) > 0:
            nearest_candidate = self.get_nearest(layer_number, candidates, query)
            candidates.remove(nearest_candidate)
            furthest_neighbor = self.get_furthest(layer_number, found_nn, query)

            cand_query_dist = self.get_distance(
                self.layers[layer_number].nodes()[nearest_candidate]['vector'],
                query
            )
            furthest_query_dist = self.get_distance(
                self.layers[layer_number].nodes()[furthest_neighbor]['vector'],
                query
            ) 

            if cand_query_dist > furthest_query_dist:
                break # all element in found_nn are evaluated 

            for neighbor in self.layers[layer_number].neighbors(nearest_candidate):
                if neighbor not in visited_elements:
                    visited_elements.add(neighbor)
                    furthest_neighbor = self.get_furthest(layer_number, found_nn, query)

                    neighbor_query_dist = self.get_distance(
                        self.layers[layer_number].nodes()[neighbor]['vector'],
                        query
                    )
                    furthest_query_dist = self.get_distance(
                        self.layers[layer_number].nodes()[furthest_neighbor]['vector'],
                        query
                    )

                    if (neighbor_query_dist < furthest_query_dist) or (len(found_nn) < ef):
                        candidates.add(neighbor)
                        found_nn.add(neighbor)
                        if len(found_nn) > ef:
                            found_nn.pop(furthest_neighbor)

        return found_nn
                        
        # neighbors = list(self.layers[layer_number].neighbors(entry_point))
        # distances = []
        # for neighbor in neighbors:
        #     distances.append(self.get_distance(neighbor, query))

        # nearest = sorted(list(zip(neighbors, distances)), key=lambda x: x[1])

        # dist_to_nearest = self.get_distance(query, self.layers[layer_number][nearest[0]])
        # dist_to_entry = self.get_distance(query, self.layers[layer_number][entry_point])

        # if dist_to_nearest > dist_to_entry:
        #     return nearest[:ef]
        # else:
        #     self.get_nn_in_layer(
        #         layer_number=layer_number,
        #         query=query,
        #         entry_point=nearest[:ef]
        #     )
        
    def get_distance(self, u, v):
        return (((u - v)**2).sum())**0.5

