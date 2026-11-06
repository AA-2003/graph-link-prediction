import csv
import hashlib
import json
from pathlib import Path
import networkx as nx
import numpy as np
import torch
from linkpred.data import load_dataset
from linkpred.gnn import FittedGNN, load_checkpoint, training_edge_index
from linkpred.split import EdgeSplit

root = Path("reports/full-benchmark")
output = Path("benchmarks/2026-10-23")
torch.set_num_threads(3)
results = {"hope_eigenpair_checks": [], "gnn_checkpoint_replays": []}
for name in ("Cora", "PubMed", "PPI"):
    dataset = load_dataset(name, cache="data", ppi_index=0)
    for seed in (42, 123, 2026):
        case_root = root / "runs" / name / f"seed-{seed}"

        def report_path(method):
            paths = sorted((case_root / method).glob("attempt-*/results.json"), reverse=True)
            return next(p for p in paths if json.loads(p.read_text(encoding="utf-8"))["results"][0]["status"] == "ok")

        hope_path = report_path("hope")
        report = json.loads(hope_path.read_text(encoding="utf-8"))
        data = json.loads((hope_path.parent / "split.json").read_text(encoding="utf-8"))
        for key in ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg"):
            data[key] = tuple(tuple(pair) for pair in data[key])
        split = EdgeSplit(**data)
        adjacency = nx.to_scipy_sparse_array(
            split.train_graph, nodelist=range(split.num_nodes), dtype=float, weight=None, format="csr"
        )
        with np.load(hope_path.parent / "hope_embedding.npz") as arrays:
            source, target = arrays["source"], arrays["target"]
        norms = np.linalg.norm(source, axis=0)
        assert (norms > 0).all()
        eigenvectors = source / norms
        proximity = np.sign(np.sum(source * target, axis=0)) * norms**2
        beta = report["results"][0]["parameters"]["beta"]
        eigenvalues = proximity / (beta * (1 + proximity))
        residual = adjacency @ eigenvectors - eigenvectors * eigenvalues
        relative = float(np.linalg.norm(residual) / max(np.linalg.norm(adjacency @ eigenvectors), 1e-12))
        assert relative < 1e-6, relative
        results["hope_eigenpair_checks"].append(
            {"dataset": name, "seed": seed, "relative_adjacency_eigenpair_residual": relative, "rank": source.shape[1]}
        )
        features = torch.as_tensor(dataset.features, dtype=torch.float32)
        edge_index = training_edge_index(split)
        pairs, _ = split.examples("test")
        for method in ("gcn", "sage", "gat"):
            path = report_path(method)
            r = json.loads(path.read_text(encoding="utf-8"))
            assert (
                hashlib.sha256(np.ascontiguousarray(dataset.features).tobytes()).hexdigest()
                == r["metadata"]["features_sha256"]
            )
            model, checkpoint = load_checkpoint(path.parent / f"{method}_checkpoint.pth", device="cpu")
            assert checkpoint["split_id"] == split.identifier == r["metadata"]["split_id"]
            assert checkpoint["best_epoch"] == r["results"][0]["best_epoch"]
            fitted = FittedGNN(model, features, edge_index, [], checkpoint["best_epoch"], r["results"][0]["validation"])
            replay = fitted.scores(pairs)
            with (path.parent / "test_scores.csv").open(newline="", encoding="utf-8") as stream:
                original = np.array([float(row["score"]) for row in csv.DictReader(stream)])
            max_error = float(np.max(np.abs(replay - original)))
            assert max_error < 5e-4, (name, seed, method, max_error)
            results["gnn_checkpoint_replays"].append(
                {
                    "dataset": name,
                    "seed": seed,
                    "method": method,
                    "replayed_on": "cpu",
                    "maximum_absolute_prediction_error_vs_saved_cuda_scores": max_error,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "split_id": split.identifier,
                }
            )
        print(name, seed, "HOPE residual", relative, "checkpoint replays passed", flush=True)
results["status"] = "passed"
(output / "numerical-checks.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print("Passed 9 sparse HOPE eigenpair checks and 27 saved-checkpoint prediction replays.")
