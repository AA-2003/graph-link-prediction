import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np

from linkpred.data import load_dataset
from linkpred.metrics import evaluate
from linkpred.split import split_edges

HAS_GNN = find_spec("torch") is not None and find_spec("torch_geometric") is not None
HAS_NODE2VEC = find_spec("node2vec") is not None


@unittest.skipUnless(HAS_NODE2VEC, "Install the embeddings extra.")
class Node2VecTests(unittest.TestCase):
    def test_training_is_repeatable_with_single_worker_and_stable_hash(self):
        from linkpred.methods import node2vec_embedding

        graph = split_edges(load_dataset().graph).train_graph
        first = node2vec_embedding(graph, dimensions=8, walk_length=5, num_walks=2)
        second = node2vec_embedding(graph, dimensions=8, walk_length=5, num_walks=2)
        np.testing.assert_allclose(first.source, second.source, atol=1e-7)
        self.assertEqual(first.source.shape, (len(graph), 8))
        pairs = np.asarray([[0, 3], [2, 5]])
        np.testing.assert_allclose(first.scores(pairs), first.scores(pairs[:, ::-1]))


@unittest.skipUnless(HAS_GNN, "Install the gnn extra.")
class GNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        torch.set_num_threads(2)
        cls.dataset = load_dataset()
        cls.split = split_edges(cls.dataset.graph)

    def test_encoder_edges_exclude_both_orientations_of_heldout_edges(self):
        from linkpred.gnn import training_edge_index

        index = training_edge_index(self.split).numpy().T
        observed = {tuple(pair) for pair in index}
        expected = set(self.split.train_pos) | {(v, u) for u, v in self.split.train_pos}
        self.assertEqual(observed, expected)
        forbidden = set(self.split.val_pos + self.split.test_pos)
        self.assertFalse(observed & forbidden)
        self.assertFalse(observed & {(v, u) for u, v in forbidden})

    def test_each_architecture_trains_and_scores_undirected_single_edges(self):
        import torch
        from linkpred.gnn import fit_gnn

        pairs, labels = self.split.examples("test")
        for name in ("gcn", "sage", "gat"):
            with self.subTest(model=name):
                fitted = fit_gnn(self.dataset.features, self.split, name, epochs=2, dimensions=8)
                scores = fitted.scores(pairs)
                self.assertEqual(scores.shape, labels.shape)
                np.testing.assert_allclose(scores, fitted.scores(pairs[:, ::-1]))
                self.assertTrue(np.isfinite(scores).all())
                self.assertTrue(all(0 <= score <= 1 for score in scores))
                with torch.inference_mode():
                    embeddings = fitted.model.encode(fitted.features, fitted.edge_index)
                    one = torch.tensor([[0], [1]])
                    forward = fitted.model.decode(embeddings, one)
                    reverse = fitted.model.decode(embeddings, one.flip(0))
                self.assertEqual(tuple(forward.shape), (1,))
                torch.testing.assert_close(forward, reverse)
                self.assertEqual(len(fitted.history), 2)

    def test_best_validation_checkpoint_is_restored_instead_of_last_epoch(self):
        import torch
        from linkpred.gnn import fit_gnn

        reference = fit_gnn(self.dataset.features, self.split, "gcn", epochs=1, dimensions=8)
        validation = [{"auc_roc": value, "average_precision": value} for value in (0.9, 0.1, 0.2)]
        with patch("linkpred.gnn.evaluate", side_effect=validation):
            fitted = fit_gnn(self.dataset.features, self.split, "gcn", epochs=3, dimensions=8)
        self.assertEqual(fitted.best_epoch, 1)
        for name, tensor in fitted.model.state_dict().items():
            torch.testing.assert_close(tensor, reference.model.state_dict()[name], rtol=0, atol=0)

    def test_checkpoint_roundtrip_uses_state_dict_and_preserves_predictions(self):
        import torch
        from linkpred.gnn import fit_gnn, load_checkpoint

        fitted = fit_gnn(self.dataset.features, self.split, "gcn", epochs=1, dimensions=8)
        pairs, labels = self.split.examples("test")
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "gcn.pth"
            fitted.save(checkpoint, self.split.identifier)
            model, metadata = load_checkpoint(checkpoint)
        self.assertEqual(metadata["split_id"], self.split.identifier)
        with torch.inference_mode():
            embeddings = model.encode(fitted.features, fitted.edge_index)
            scores = model.decode(embeddings, torch.as_tensor(pairs).t()).sigmoid().numpy()
        np.testing.assert_array_equal(scores, fitted.scores(pairs))
        self.assertTrue(np.isfinite(evaluate(labels, scores)["auc_roc"]))


if __name__ == "__main__":
    unittest.main()
