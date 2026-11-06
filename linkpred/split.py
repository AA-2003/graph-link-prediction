import hashlib
import json
import random
from bisect import bisect_right
from dataclasses import dataclass

import networkx as nx
import numpy as np


def canonical_edges(graph):
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Use a simple undirected graph.")
    if set(graph.nodes) != set(range(len(graph))):
        raise ValueError("Node IDs must be consecutive integers starting at zero.")
    return sorted({(min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges if u != v})


def _rank(u, v, n):
    return u * (2 * n - u - 1) // 2 + v - u - 1


def _unrank(value, n):
    low, high = 0, n - 1
    while low + 1 < high:
        middle = (low + high) // 2
        if middle * (2 * n - middle - 1) // 2 <= value:
            low = middle
        else:
            high = middle
    return low, low + 1 + value - low * (2 * n - low - 1) // 2


def sample_negative_edges(num_nodes, positive_edges, count, seed=42):
    blocked = sorted({_rank(min(u, v), max(u, v), num_nodes) for u, v in positive_edges if u != v})
    available = num_nodes * (num_nodes - 1) // 2 - len(blocked)
    if count < 0 or count > available:
        raise ValueError(f"Need {count} unique non-edges, but only {available} are available.")
    negatives = []
    for index in random.Random(seed).sample(range(available), count):
        # Map a rank in the complement to a rank among all unordered node pairs.
        low, high = index, index + len(blocked)
        while low < high:
            middle = (low + high) // 2
            if middle + 1 - bisect_right(blocked, middle) <= index:
                low = middle + 1
            else:
                high = middle
        negatives.append(_unrank(low, num_nodes))
    return tuple(negatives)


@dataclass(frozen=True)
class EdgeSplit:
    num_nodes: int
    train_pos: tuple
    val_pos: tuple
    test_pos: tuple
    train_neg: tuple
    val_neg: tuple
    test_neg: tuple
    seed: int

    @property
    def train_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_nodes))
        graph.add_edges_from(self.train_pos)
        return graph

    def examples(self, part):
        if part not in {"train", "val", "test"}:
            raise ValueError("Unknown evaluation split.")
        positives, negatives = getattr(self, part + "_pos"), getattr(self, part + "_neg")
        pairs = np.asarray(positives + negatives, dtype=np.int64).reshape(-1, 2)
        labels = np.concatenate((np.ones(len(positives)), np.zeros(len(negatives))))
        return pairs, labels

    def to_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    @property
    def identifier(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def split_edges(graph, seed=42, val_ratio=0.1, test_ratio=0.2):
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("Validation/test ratios must be positive and sum to less than one.")
    positives = canonical_edges(graph)
    random.Random(seed).shuffle(positives)
    n_val, n_test = int(len(positives) * val_ratio), int(len(positives) * test_ratio)
    if min(n_val, n_test) < 1 or n_val + n_test >= len(positives):
        raise ValueError("The graph is too small for non-empty training, validation and test sets.")
    negatives = sample_negative_edges(len(graph), positives, len(positives), seed + 1)
    n_train = len(positives) - n_val - n_test
    return EdgeSplit(
        len(graph),
        tuple(positives[:n_train]),
        tuple(positives[n_train : n_train + n_val]),
        tuple(positives[n_train + n_val :]),
        negatives[:n_train],
        negatives[n_train : n_train + n_val],
        negatives[n_train + n_val :],
        seed,
    )
