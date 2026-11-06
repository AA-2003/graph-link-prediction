import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
from scipy.linalg import svd

from linkpred.methods import hope_embedding
from linkpred.sparse_hope import sparse_hope_embedding


class SparseHopeTests(unittest.TestCase):
    def test_full_rank_matches_dense_katz(self):
        graph = nx.path_graph(7)
        dense = hope_embedding(graph, dimensions=14, beta=0.05)
        sparse = sparse_hope_embedding(graph, dimensions=14, beta=0.05)
        np.testing.assert_allclose(sparse.source @ sparse.target.T, dense.source @ dense.target.T, atol=1e-10)

    def test_truncated_result_matches_exact_best_rank_katz_approximation(self):
        graph = nx.barabasi_albert_graph(80, 3, seed=4)
        adjacency = nx.to_numpy_array(graph)
        katz = np.linalg.solve(np.eye(80) - 0.01 * adjacency, 0.01 * adjacency)
        left, values, right = svd(katz, full_matrices=False)
        expected = (left[:, :4] * values[:4]) @ right[:4]
        embedding = sparse_hope_embedding(graph, dimensions=8, beta=0.01)
        np.testing.assert_allclose(embedding.source @ embedding.target.T, expected, atol=1e-8)

    def test_sparse_path_does_not_allocate_dense_adjacency(self):
        graph = nx.barabasi_albert_graph(120, 2, seed=5)
        adjacency = nx.to_scipy_sparse_array(graph, dtype=float, weight=None, format="csr")
        with patch("linkpred.sparse_hope.nx.to_scipy_sparse_array", return_value=adjacency):
            with patch.object(type(adjacency), "toarray", side_effect=AssertionError("Dense allocation")):
                embedding = sparse_hope_embedding(graph, dimensions=8)
        self.assertEqual(embedding.source.shape, (120, 4))

    def test_repeated_eigenspaces_and_invalid_beta(self):
        graph = nx.complete_graph(30)
        embedding = sparse_hope_embedding(graph, dimensions=8, beta=0.01)
        reconstructed = embedding.source @ embedding.target.T
        self.assertTrue(np.isfinite(reconstructed).all())
        np.testing.assert_allclose(reconstructed, reconstructed.T, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "convergence"):
            sparse_hope_embedding(graph, dimensions=8, beta=1)


if __name__ == "__main__":
    unittest.main()
