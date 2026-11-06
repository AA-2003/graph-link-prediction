from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np


@dataclass
class GraphDataset:
    name: str
    graph: nx.Graph
    features: np.ndarray
    description: str


def load_dataset(name="karate", cache="data", ppi_index=0):
    if name == "karate":
        graph = nx.Graph()
        source = nx.karate_club_graph()
        graph.add_nodes_from(source.nodes)
        graph.add_edges_from(source.edges)
        return GraphDataset(
            "karate",
            graph,
            np.eye(len(graph), dtype=np.float32),
            "Zachary's karate club; unweighted edges and one-hot node identity features.",
        )
    if name not in {"Cora", "PubMed", "PPI"}:
        raise ValueError(f"Unsupported dataset: {name}")
    try:
        from torch_geometric.datasets import PPI, Planetoid
        from torch_geometric.transforms import NormalizeFeatures
    except ImportError:
        raise RuntimeError("Install the gnn extra to load Cora, PubMed or PPI.") from None
    root = Path(cache) / name
    if name == "PPI":
        collection = PPI(root=str(root), split="train", transform=NormalizeFeatures())
        if not 0 <= ppi_index < len(collection):
            raise ValueError(f"PPI graph index must be between 0 and {len(collection) - 1}.")
        data = collection[ppi_index]
        description = f"PPI training graph {ppi_index}; transductive link prediction within this single graph."
    else:
        collection = Planetoid(root=str(root), name=name, transform=NormalizeFeatures())
        data = collection[0]
        description = f"{name}; undirected citation graph with row-normalized original node features."
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_edges_from(data.edge_index.t().cpu().tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    features = data.x.detach().cpu().numpy().astype(np.float32, copy=True)
    return GraphDataset(name, graph, features, description)
