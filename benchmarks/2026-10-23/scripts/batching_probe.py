import csv
import json
import time
from pathlib import Path
import numpy as np
from node2vec import Node2Vec
from threadpoolctl import threadpool_limits
from linkpred.data import load_dataset
from linkpred.methods import _stable_hash
from linkpred.metrics import evaluate
from linkpred.split import EdgeSplit

root = Path("reports/full-benchmark")
output = Path("benchmarks/2026-10-23")
output.mkdir(parents=True, exist_ok=True)
rows = []
for name in ("Cora", "PubMed", "PPI"):
    dataset = load_dataset(name, cache="data", ppi_index=0)
    for seed in (42, 123, 2026):
        case = root / "runs" / name / f"seed-{seed}" / "node2vec" / "attempt-001"
        original = json.loads((case / "results.json").read_text(encoding="utf-8"))
        data = json.loads((case / "split.json").read_text(encoding="utf-8"))
        for key in ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg"):
            data[key] = tuple(tuple(pair) for pair in data[key])
        split = EdgeSplit(**data)
        assert split.identifier == original["metadata"]["split_id"]
        graph = split.train_graph
        pairs, labels = split.examples("val")
        candidates = []
        with threadpool_limits(limits=3):
            started = time.perf_counter()
            for window in (5, 10):
                walker = Node2Vec(
                    graph,
                    dimensions=64,
                    walk_length=20,
                    num_walks=20,
                    workers=1,
                    quiet=True,
                    seed=seed,
                    weight_key="weight",
                )
                model = walker.fit(
                    window=window, min_count=1, batch_words=10000, workers=1, seed=seed, hashfxn=_stable_hash
                )
                vectors = np.stack([model.wv[str(node)] for node in range(len(graph))])
                scores = np.einsum("ij,ij->i", vectors[pairs[:, 0]], vectors[pairs[:, 1]])
                metrics = evaluate(labels, scores)
                candidates.append({"window": window, **metrics})
            elapsed = time.perf_counter() - started
        best = max(candidates, key=lambda c: (c["auc_roc"], c["average_precision"]))
        old = original["results"][0]
        rows.append(
            {
                "dataset": name,
                "seed": seed,
                "baseline_batch_words": 4,
                "probe_batch_words": 10000,
                "baseline_validation_auc": old["validation"]["auc_roc"],
                "probe_validation_auc": best["auc_roc"],
                "baseline_validation_ap": old["validation"]["average_precision"],
                "probe_validation_ap": best["average_precision"],
                "baseline_method_seconds": old["elapsed_seconds"],
                "probe_fit_validation_seconds": elapsed,
                "probe_selected_window": best["window"],
                "probe_candidates": json.dumps(candidates),
                "split_id": split.identifier,
                "evaluation_partition": "validation-only",
            }
        )
        print(
            f"{name}/{seed}: batch 10000 {elapsed:.1f}s, validation AUC {best['auc_roc']:.4f} vs {old['validation']['auc_roc']:.4f}",
            flush=True,
        )
        with (output / "node2vec-batching-probe.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
print("Completed nine validation-only batching probes; no test scores computed.")
