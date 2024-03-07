import pickle


with open('./indices/test_index.hnsw', 'rb') as file:
    index = pickle.load(file)

index.print_parameters()
