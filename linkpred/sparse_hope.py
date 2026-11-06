import networkx as nx
import numpy as np
from scipy.linalg import eigh, orth
from scipy.sparse.linalg import eigsh

from linkpred.methods import Embedding, ResourceLimit


def sparse_hope_embedding(graph, dimensions=64, beta=0.01, seed=42, max_dense_nodes=4000):
    n = len(graph)
    rank = dimensions // 2
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Spectral HOPE requires a simple undirected graph.")
    if dimensions < 2 or dimensions % 2 or rank > n:
        raise ValueError("HOPE dimensions must be even and between 2 and twice the node count.")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("HOPE beta must be finite and positive.")
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=range(n), dtype=float, weight=None, format="csr")
    if n <= 2 * rank + 1:
        if n > max_dense_nodes:
            raise ResourceLimit("The requested rank needs a dense fallback; reduce the HOPE dimensions.")
        eigenvalues, vectors = eigh(adjacency.toarray())
        radius = float(np.max(np.abs(eigenvalues)))
    else:
        rng = np.random.default_rng(seed)
        high_values, high_vectors = eigsh(
            adjacency, k=rank, which="LA", v0=rng.standard_normal(n), tol=1e-9, maxiter=10000
        )
        low_values, low_vectors = eigsh(
            adjacency, k=rank, which="SA", v0=rng.standard_normal(n), tol=1e-9, maxiter=10000
        )
        radius = max(float(np.max(np.abs(high_values))), float(np.max(np.abs(low_values))))
        # Orthonormalization also handles overlaps in repeated eigenspaces.
        basis = orth(np.column_stack((high_vectors, low_vectors)), rcond=1e-10)
        projected = basis.T @ (adjacency @ basis)
        eigenvalues, small_vectors = eigh((projected + projected.T) / 2)
        vectors = basis @ small_vectors
    if beta * radius >= 1 - 1e-10:
        raise ValueError(f"Katz convergence requires beta * spectral_radius < 1; radius={radius:.5g}.")
    proximity_values = beta * eigenvalues / (1 - beta * eigenvalues)
    selected = np.argsort(-np.abs(proximity_values), kind="stable")[:rank]
    values, vectors = proximity_values[selected], vectors[:, selected]
    scale = np.sqrt(np.abs(values))
    return Embedding(vectors * scale, vectors * (np.sign(values) * scale))
