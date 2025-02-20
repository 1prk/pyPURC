import osmnx as ox

def get_network(place):
    return ox.graph_from_place(place)

def flatten(x):
    return [item for sublist in x for item in sublist]