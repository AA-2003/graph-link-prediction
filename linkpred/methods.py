import hashlib
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import eigsh
from sklearn.utils.extmath import randomized_svd


class ResourceLimit(ValueError):
    pass


def classic_scores(graph, pairs, method):
    functions = {"jaccard": nx.jaccard_coefficient, "adamic-adar": nx.adamic_adar_index}
    return np.asarray([score for _, _, score in functions[method](graph, pairs.tolist())])


@dataclass
class Embedding:
    source: np.ndarray
    target: np.ndarray

    def scores(self, pairs):
        u, v = np.asarray(pairs, dtype=np.int64).T
        forward = np.sum(self.source[u] * self.target[v], axis=1)
        reverse = np.sum(self.source[v] * self.target[u], axis=1)
        return (forward + reverse) / 2


def hope_embedding(graph, dimensions=32, beta=0.01, seed=42, max_nodes=4000):
    n = len(graph)
    if n > max_nodes:
        raise ResourceLimit(f"Dense Katz/HOPE is limited to {max_nodes} nodes; this graph has {n}.")
    if dimensions < 2 or dimensions % 2 or dimensions // 2 > n:
        raise ValueError("HOPE dimensions must be even and between 2 and twice the node count.")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("HOPE beta must be finite and positive.")
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=range(n), dtype=float, weight=None, format="csr")
    radius = (
        float(abs(eigsh(adjacency, k=1, which="LM", return_eigenvectors=False, v0=np.ones(n))[0]))
        if n > 2
        else float(max(dict(graph.degree()).values(), default=0))
    )
    if beta * radius >= 1:
        raise ValueError(f"Katz convergence requires beta * spectral_radius < 1; radius={radius:.5g}.")
    dense = adjacency.toarray()
    katz = solve(np.eye(n) - beta * dense, beta * dense, assume_a="pos")
    left, singular_values, right_t = randomized_svd(katz, n_components=dimensions // 2, random_state=seed)
    scale = np.sqrt(singular_values)
    return Embedding(left * scale, right_t.T * scale)


def _stable_hash(value):
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little")


def node2vec_embedding(graph, dimensions=32, walk_length=10, num_walks=10, window=5, seed=42):
    try:
        from node2vec import Node2Vec
    except ImportError:
        raise RuntimeError("Install the embeddings extra to run Node2Vec.") from None
    if min(dimensions, walk_length, num_walks, window) < 1:
        raise ValueError("Node2Vec dimensions and walk settings must be positive.")
    walker = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=1,
        quiet=True,
        seed=seed,
        weight_key="weight",
    )
    model = walker.fit(window=window, min_count=1, batch_words=4, workers=1, seed=seed, hashfxn=_stable_hash)
    vectors = np.stack([model.wv[str(node)] for node in range(len(graph))])
    return Embedding(vectors, vectors)
