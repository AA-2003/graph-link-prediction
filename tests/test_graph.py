import csv
import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np

from linkpred.data import load_dataset
from linkpred.experiment import RunConfig, run_experiment, select_embedding
from linkpred.methods import ResourceLimit, classic_scores, hope_embedding
from linkpred.metrics import evaluate
from linkpred.split import canonical_edges, sample_negative_edges, split_edges


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.graph = load_dataset().graph

    def test_reproducible_shared_split_preserves_nodes_and_heldout_edges(self):
        self.graph.add_node(34)
        split = split_edges(self.graph)
        repeated = split_edges(self.graph)
        self.assertEqual(split, repeated)
        self.assertEqual(split.identifier, repeated.identifier)
        self.assertEqual(set(split.train_graph), set(self.graph))
        positives = set(canonical_edges(self.graph))
        parts = [set(getattr(split, name + "_pos")) for name in ("train", "val", "test")]
        self.assertEqual(set.union(*parts), positives)
        self.assertFalse(parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
        self.assertEqual(set(split.train_graph.edges), parts[0])
        negative_parts = [set(getattr(split, name + "_neg")) for name in ("train", "val", "test")]
        negatives = set.union(*negative_parts)
        self.assertEqual(len(negatives), len(positives))
        self.assertFalse(positives & negatives)
        self.assertTrue(all(u < v for u, v in negatives))
        self.assertEqual(sum(map(len, negative_parts)), len(negatives))

    def test_dense_complement_sampler_returns_exact_unique_non_edges(self):
        allowed = {(0, 1), (3, 4)}
        positives = set(combinations(range(6), 2)) - allowed
        sampled = sample_negative_edges(6, positives, 2)
        self.assertEqual(set(sampled), allowed)
        with self.assertRaisesRegex(ValueError, "only 2"):
            sample_negative_edges(6, positives, 3)

    def test_sampler_handles_large_node_space_without_enumerating_pairs(self):
        sampled = sample_negative_edges(1_000_000, [(0, 1), (2, 3)], 20)
        self.assertEqual(len(set(sampled)), 20)
        self.assertTrue(all(0 <= u < v < 1_000_000 for u, v in sampled))
        self.assertFalse(set(sampled) & {(0, 1), (2, 3)})

    def test_self_loops_are_removed_and_bad_splits_fail_clearly(self):
        self.graph.add_edge(0, 0)
        split = split_edges(self.graph)
        self.assertNotIn((0, 0), split.train_pos + split.val_pos + split.test_pos)
        with self.assertRaises(ValueError):
            split_edges(self.graph, val_ratio=0.7, test_ratio=0.5)
        with self.assertRaises(ValueError):
            split_edges(nx.path_graph(3))
        with self.assertRaises(ValueError):
            split_edges(nx.complete_graph(10))
        with self.assertRaises(ValueError):
            canonical_edges(nx.DiGraph([(0, 1)]))

    def test_example_labels_match_the_persisted_pairs(self):
        split = split_edges(self.graph)
        for name in ("train", "val", "test"):
            pairs, labels = split.examples(name)
            count = len(getattr(split, name + "_pos"))
            self.assertEqual(pairs.shape, (count * 2, 2))
            np.testing.assert_array_equal(labels, [1] * count + [0] * count)


class MethodTests(unittest.TestCase):
    def test_classic_scores_match_known_neighborhoods(self):
        graph = nx.Graph([(0, 1), (1, 2), (2, 3)])
        pairs = np.asarray([[0, 2], [0, 3]])
        np.testing.assert_allclose(classic_scores(graph, pairs, "jaccard"), [0.5, 0])
        np.testing.assert_allclose(classic_scores(graph, pairs, "adamic-adar"), [1 / np.log(2), 0])

    def test_full_rank_hope_reconstructs_katz_and_has_symmetric_scores(self):
        graph = nx.path_graph(5)
        embedding = hope_embedding(graph, dimensions=10, beta=0.1)
        adjacency = nx.to_numpy_array(graph)
        expected = np.linalg.solve(np.eye(5) - 0.1 * adjacency, 0.1 * adjacency)
        np.testing.assert_allclose(embedding.source @ embedding.target.T, expected, atol=1e-10)
        pairs = np.asarray([[0, 2], [1, 4], [3, 4]])
        np.testing.assert_allclose(embedding.scores(pairs), embedding.scores(pairs[:, ::-1]))
        np.testing.assert_allclose(embedding.scores(pairs), expected[pairs[:, 0], pairs[:, 1]], atol=1e-10)

    def test_hope_rejects_nonconvergent_beta_and_guards_dense_allocation(self):
        graph = nx.path_graph(5)
        with self.assertRaisesRegex(ValueError, "convergence"):
            hope_embedding(graph, dimensions=4, beta=10)
        with patch("networkx.to_scipy_sparse_array") as build:
            with self.assertRaises(ResourceLimit):
                hope_embedding(graph, dimensions=4, max_nodes=2)
            build.assert_not_called()
        with self.assertRaises(ValueError):
            hope_embedding(graph, dimensions=3)

    def test_hyperparameters_are_selected_using_only_validation_examples(self):
        split = split_edges(load_dataset().graph)
        val_pairs, val_labels = split.examples("val")
        calls = []

        class Candidate:
            def __init__(self, good):
                self.good = good

            def scores(self, pairs):
                np.testing.assert_array_equal(pairs, val_pairs)
                calls.append(pairs.copy())
                return val_labels if self.good else 1 - val_labels

        good, bad = Candidate(True), Candidate(False)
        with patch("linkpred.experiment.hope_embedding", side_effect=[good, bad]):
            selected, params, metrics, trials = select_embedding(split.train_graph, split, "hope", RunConfig(tune=True))
        self.assertIs(selected, good)
        self.assertEqual(params["beta"], 0.01)
        self.assertEqual(metrics["auc_roc"], 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(trials), 2)

    def test_metrics_reject_misaligned_or_nonfinite_scores(self):
        self.assertEqual(evaluate([1, 0], [0.9, 0.1])["auc_roc"], 1)
        for scores in ([0.1], [float("nan"), 0.2]):
            with self.assertRaises(ValueError):
                evaluate([1, 0], scores)


class ReportTests(unittest.TestCase):
    def test_reports_share_split_and_keep_skipped_metrics_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            report = run_experiment(RunConfig(max_hope_nodes=2), ("jaccard", "adamic-adar", "hope"), output)
            self.assertEqual({row["split_id"] for row in report["results"]}, {report["metadata"]["split_id"]})
            skipped = report["results"][-1]
            self.assertEqual(skipped["status"], "skipped")
            self.assertNotIn("test", skipped)
            self.assertTrue((output / "scores.png").is_file())
            self.assertTrue((output / "scores.svg").is_file())
            stored = json.loads((output / "results.json").read_text())
            self.assertEqual(stored["metadata"]["split_id"], report["metadata"]["split_id"])
            with (output / "results.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[-1]["test_auc"], "")
            with (output / "test_scores.csv").open(newline="", encoding="utf-8") as stream:
                predictions = list(csv.DictReader(stream))
            for method in ("jaccard", "adamic-adar"):
                selected = [row for row in predictions if row["method"] == method]
                metrics = evaluate([int(row["label"]) for row in selected], [float(row["score"]) for row in selected])
                row = next(row for row in report["results"] if row["method"] == method)
                self.assertEqual(metrics, row["test"])
            with self.assertRaises(FileExistsError):
                run_experiment(output=output)

    def test_invalid_configuration_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "invalid"
            with self.assertRaises(ValueError):
                run_experiment(RunConfig(epochs=0), output=output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
