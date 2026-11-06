import csv
import json
import time
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from linkpred.metrics import evaluate
from linkpred.split import EdgeSplit

root = Path("reports/full-benchmark")
output = Path("benchmarks/2026-10-23")
output.mkdir(parents=True, exist_ok=True)
rows = []
for name in ("Cora", "PubMed", "PPI"):
    for seed in (42, 123, 2026):
        case = root / "runs" / name / f"seed-{seed}" / "node2vec" / "attempt-001"
        report = json.loads((case / "results.json").read_text(encoding="utf-8"))
        data = json.loads((case / "split.json").read_text(encoding="utf-8"))
        for key in ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg"):
            data[key] = tuple(tuple(pair) for pair in data[key])
        split = EdgeSplit(**data)
        assert split.identifier == report["metadata"]["split_id"]
        with np.load(case / "node2vec_embedding.npz") as arrays:
            vectors = arrays["source"].copy()
            assert np.array_equal(vectors, arrays["target"])
        train_pairs, train_labels = split.examples("train")
        val_pairs, val_labels = split.examples("val")

        def pair_features(pairs):
            a, b = vectors[pairs[:, 0]], vectors[pairs[:, 1]]
            return np.concatenate((a * b, np.abs(a - b)), axis=1)

        norm = np.linalg.norm(vectors, axis=1)
        denominator = norm[val_pairs[:, 0]] * norm[val_pairs[:, 1]]
        dot = np.einsum("ij,ij->i", vectors[val_pairs[:, 0]], vectors[val_pairs[:, 1]])
        cosine = np.divide(dot, denominator, out=np.zeros(len(dot)), where=denominator > 0)
        for method, scores, seconds in (("cosine", cosine, 0.0),):
            m = evaluate(val_labels, scores)
            rows.append(
                {
                    "dataset": name,
                    "seed": seed,
                    "decoder": method,
                    "validation_auc": m["auc_roc"],
                    "validation_ap": m["average_precision"],
                    "additional_fit_validation_seconds": seconds,
                    "baseline_dot_validation_auc": report["results"][0]["validation"]["auc_roc"],
                    "split_id": split.identifier,
                    "evaluation_partition": "validation-only",
                }
            )
        with threadpool_limits(limits=3):
            start = time.perf_counter()
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
            model.fit(pair_features(train_pairs), train_labels)
            predictions = model.predict_proba(pair_features(val_pairs))[:, 1]
            seconds = time.perf_counter() - start
        m = evaluate(val_labels, predictions)
        rows.append(
            {
                "dataset": name,
                "seed": seed,
                "decoder": "logistic-product-absdiff",
                "validation_auc": m["auc_roc"],
                "validation_ap": m["average_precision"],
                "additional_fit_validation_seconds": seconds,
                "baseline_dot_validation_auc": report["results"][0]["validation"]["auc_roc"],
                "split_id": split.identifier,
                "evaluation_partition": "validation-only",
            }
        )
        print(
            f"{name}/{seed}: dot {report['results'][0]['validation']['auc_roc']:.4f}, cosine {rows[-2]['validation_auc']:.4f}, learned decoder {m['auc_roc']:.4f}",
            flush=True,
        )
with (output / "node2vec-decoder-probes.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("Validation-only decoder exploration complete. No test scores computed.")
