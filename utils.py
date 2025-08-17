import numpy as np
import networkx as nx
import random

def train_test_split_graph(graph, test_ratio=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Check if graph is directed
    if graph.is_directed():
        return "This function is for undirected graphs."

    all_edges = list(graph.edges())
    np.random.shuffle(all_edges)
    num_test_edges = int(len(all_edges) * test_ratio)

    # Split positive edges
    test_edges_positive = []
    train_graph = nx.Graph(graph)
    for edge in all_edges:
        train_graph.remove_edge(*edge)
        if nx.is_connected(train_graph):
            test_edges_positive.append(edge)
            if len(test_edges_positive) == num_test_edges:
                break
        else:
            train_graph.add_edge(*edge)  # Keep edge if removing it disconnects the graph

    # Generate negative edges
    all_possible_edges = set(
        tuple(sorted((u, v))) for u in graph.nodes() for v in graph.nodes() if u != v
    )
    existing_edges = set(graph.edges())
    train_edges = set(train_graph.edges())
    non_existent_edges = list(all_possible_edges - existing_edges - train_edges)
    np.random.shuffle(non_existent_edges)
    test_edges_negative = non_existent_edges[:num_test_edges]

    return train_graph, test_edges_positive, test_edges_negative
