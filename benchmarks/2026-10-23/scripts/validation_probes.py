import csv
import hashlib
import json
import sys
from pathlib import Path
import networkx as nx
import numpy as np
from linkpred.data import load_dataset
from linkpred.metrics import evaluate
from linkpred.split import EdgeSplit

root = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/full-benchmark")
output = Path(sys.argv[2] if len(sys.argv) > 2 else "benchmarks/2026-10-23")
output.mkdir(parents=True, exist_ok=True)
rows, diagnostics = [], []
for name in ("Cora", "PubMed", "PPI"):
    dataset = load_dataset(name, cache="data", ppi_index=0)
    features = dataset.features
    norms = np.linalg.norm(features, axis=1)
    for seed in (42, 123, 2026):
        report_path = root / "runs" / name / f"seed-{seed}" / "jaccard" / "attempt-001"
        report = json.loads((report_path / "results.json").read_text(encoding="utf-8"))
        assert (
            hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()
            == report["metadata"]["features_sha256"]
        )
        data = json.loads((report_path / "split.json").read_text(encoding="utf-8"))
        for key in ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg"):
            data[key] = tuple(tuple(pair) for pair in data[key])
        split = EdgeSplit(**data)
        assert split.identifier == report["metadata"]["split_id"]
        graph = split.train_graph
        pairs, labels = split.examples("val")
        neighbors = {u: set(graph[u]) for u in graph}
        common = np.array([len(neighbors[u] & neighbors[v]) for u, v in pairs], dtype=float)
        resource = np.array([sum(1 / graph.degree(w) for w in neighbors[u] & neighbors[v]) for u, v in pairs])
        degree_product = np.array([graph.degree(u) * graph.degree(v) for u, v in pairs], dtype=float)
        left, right = pairs[:, 0], pairs[:, 1]
        denominator = norms[left] * norms[right]
        cosine = np.divide(
            np.einsum("ij,ij->i", features[left], features[right]),
            denominator,
            out=np.zeros(len(pairs)),
            where=denominator > 0,
        )
        for method, scores in (
            ("common-neighbors", common),
            ("resource-allocation", resource),
            ("preferential-attachment", degree_product),
            ("feature-cosine", cosine),
        ):
            metrics = evaluate(labels, scores)
            rows.append(
                {
                    "dataset": name,
                    "seed": seed,
                    "method": method,
                    "validation_auc": metrics["auc_roc"],
                    "validation_ap": metrics["average_precision"],
                    "split_id": split.identifier,
                    "evaluation_partition": "validation-only",
                }
            )
            print(name, seed, method, f"AUC={metrics['auc_roc']:.4f}", flush=True)
        diagnostics.append(
            {
                "dataset": name,
                "seed": seed,
                "nodes": len(graph),
                "training_edges": graph.number_of_edges(),
                "isolated_training_nodes": len(list(nx.isolates(graph))),
                "training_components": nx.number_connected_components(graph),
                "validation_positive_with_common_neighbor_fraction": float(np.mean(common[labels == 1] > 0)),
                "validation_negative_with_common_neighbor_fraction": float(np.mean(common[labels == 0] > 0)),
            }
        )
for filename, values in (("validation-probes.csv", rows), ("graph-diagnostics.csv", diagnostics)):
    with (output / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
print("Validation-only probes complete; test predictions were not produced.")
