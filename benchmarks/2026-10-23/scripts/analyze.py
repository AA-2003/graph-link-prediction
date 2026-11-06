import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ("jaccard", "adamic-adar", "hope", "node2vec", "gcn", "sage", "gat")
LABELS = ("Jaccard", "Adamic-Adar", "HOPE", "Node2Vec", "GCN", "GraphSAGE", "GAT")
DATASETS = ("Cora", "PubMed", "PPI")
SEEDS = (42, 123, 2026)
GNN = set(METHODS[-3:])


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, fields, groups=("dataset", "method")):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in groups)].append(row)
    result = []
    for key, values in buckets.items():
        row = dict(zip(groups, key))
        row["n"] = len(values)
        for field in fields:
            numbers = [v[field] for v in values if v.get(field) is not None]
            row[field + "_mean"] = statistics.mean(numbers) if numbers else None
            row[field + "_sd"] = statistics.stdev(numbers) if len(numbers) > 1 else 0.0 if numbers else None
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="reports/full-benchmark")
    parser.add_argument("--output", default="benchmarks/2026-10-23")
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = read(root / "protocol.json")
    state_cases = {}
    for stage in ("cpu", "gnn"):
        state = read(root / f"{stage}_status.json")
        assert state["status"] == "complete", state["status"]
        for case in state["cases"]:
            state_cases[(case["dataset"], case["seed"], case["method"])] = case

    from linkpred.metrics import evaluate
    from linkpred.experiment import source_fingerprint

    assert source_fingerprint() == protocol["source_sha256"]
    raw, diagnostics, audits, histories = [], [], [], {}
    environments, sizes = {}, {}
    for dataset in DATASETS:
        for seed in SEEDS:
            split_reference = None
            pairs_reference = None
            metadata_reference = None
            for method in METHODS:
                case = state_cases[(dataset, seed, method)]
                assert case["status"] == "ok" and case["returncode"] == 0
                report_path = root / case["report"]
                report = read(report_path)
                meta, result = report["metadata"], report["results"][0]
                assert result["method"] == method and result["status"] == "ok"
                assert meta["source_sha256"] == protocol["source_sha256"]
                assert meta["configuration"]["seed"] == seed
                split = read(report_path.parent / "split.json")
                split_hash = hashlib.sha256(
                    json.dumps(split, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                assert split_hash == meta["split_id"] == result["split_id"]
                with (report_path.parent / "test_scores.csv").open(newline="", encoding="utf-8") as stream:
                    predictions = list(csv.DictReader(stream))
                pairs = [(int(r["u"]), int(r["v"]), int(r["label"])) for r in predictions]
                scores = np.asarray([float(r["score"]) for r in predictions])
                assert np.isfinite(scores).all()
                measured = evaluate(np.asarray([r[2] for r in pairs]), scores)
                for metric in measured:
                    assert abs(measured[metric] - result["test"][metric]) < 1e-12
                if split_reference is None:
                    split_reference, pairs_reference, metadata_reference = split, pairs, meta
                    groups = {
                        part: {tuple(pair) for pair in split[part]}
                        for part in ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg")
                    }
                    for part, values in groups.items():
                        assert len(values) == len(split[part])
                        assert all(0 <= u < v < split["num_nodes"] for u, v in values)
                    for a, key in enumerate(groups):
                        for other in list(groups)[a + 1 :]:
                            assert not groups[key] & groups[other]
                    assert len(split["test_pos"]) == len(split["test_neg"])
                else:
                    assert split == split_reference
                    assert pairs == pairs_reference
                    assert meta["features_sha256"] == metadata_reference["features_sha256"]
                device = protocol["gnn_device"] if method in GNN else "cpu"
                environments[device] = meta["versions"]
                sizes[dataset] = {k: meta[k] for k in ("nodes", "positive_edges", "features")}
                sizes[dataset].update(
                    {
                        part: len(split[part])
                        for part in ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg")
                    }
                )
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "device": device,
                    "test_auc": result["test"]["auc_roc"],
                    "test_ap": result["test"]["average_precision"],
                    "validation_auc": result["validation"]["auc_roc"],
                    "validation_ap": result["validation"]["average_precision"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "wall_seconds": case["wall_seconds"],
                    "peak_rss_mib": case["peak_process_rss_bytes"] / 2**20,
                    "cuda_allocated_mib": result.get("cuda_peak_allocated_bytes", 0) / 2**20,
                    "cuda_reserved_mib": result.get("cuda_peak_reserved_bytes", 0) / 2**20,
                    "best_epoch": result.get("best_epoch"),
                    "parameters": json.dumps(result["parameters"], sort_keys=True),
                    "split_id": meta["split_id"],
                    "features_sha256": meta["features_sha256"],
                    "report": str(report_path.as_posix()),
                }
                raw.append(row)
                if method in GNN:
                    history = read(report_path.parent / f"{method}_history.json")
                    assert len(history) == protocol["epochs"]
                    best = max(
                        history, key=lambda h: (h["validation"]["auc_roc"], h["validation"]["average_precision"])
                    )
                    assert best["epoch"] == result["best_epoch"]
                    assert best["validation"] == result["validation"]
                    assert (report_path.parent / f"{method}_checkpoint.pth").is_file()
                    first_close = next(
                        h["epoch"]
                        for h in history
                        if h["validation"]["auc_roc"] >= best["validation"]["auc_roc"] - 0.005
                    )
                    diagnostics.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "method": method,
                            "best_epoch": result["best_epoch"],
                            "epoch_within_0_005_best_auc": first_close,
                            "final_validation_auc": history[-1]["validation"]["auc_roc"],
                            "best_validation_auc": best["validation"]["auc_roc"],
                            "final_auc_drop": best["validation"]["auc_roc"] - history[-1]["validation"]["auc_roc"],
                            "final_loss": history[-1]["loss"],
                            "best_epoch_loss": best["loss"],
                            "test_minus_validation_auc": result["test"]["auc_roc"] - result["validation"]["auc_roc"],
                        }
                    )
                    histories[(dataset, seed, method)] = history
            audits.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "methods": len(METHODS),
                    "split_id": metadata_reference["split_id"],
                    "features_sha256": metadata_reference["features_sha256"],
                    "identical_test_pairs_and_labels": True,
                    "unique_disjoint_all_six_edge_groups": True,
                    "metrics_recomputed_from_saved_predictions": True,
                }
            )
    assert len(raw) == 63
    for name in set(environments["cpu"]) & set(environments[protocol["gnn_device"]]):
        left, right = environments["cpu"][name], environments[protocol["gnn_device"]][name]
        assert (left.split("+")[0] == right.split("+")[0]) if name == "torch" else (left == right)
    summary = aggregate(
        raw,
        (
            "test_auc",
            "test_ap",
            "validation_auc",
            "validation_ap",
            "elapsed_seconds",
            "wall_seconds",
            "peak_rss_mib",
            "cuda_allocated_mib",
            "cuda_reserved_mib",
            "best_epoch",
        ),
    )
    write_csv(output / "raw-results.csv", raw)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "training-diagnostics.csv", diagnostics)
    write_json(
        output / "audit.json",
        {
            "status": "passed",
            "successful_cases": len(raw),
            "source_sha256": protocol["source_sha256"],
            "checks": audits,
        },
    )
    write_json(output / "protocol.json", protocol)
    write_json(output / "environment-versions.json", environments)
    write_json(output / "dataset-sizes.json", sizes)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
    lookup = {(r["dataset"], r["method"]): r for r in summary}
    colors = ["#7c8793", "#4b657a", "#4c9a8e", "#59a14f", "#4e79a7", "#b07aa1", "#e18b44"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for col, dataset in enumerate(DATASETS):
        for idx, metric in enumerate(("test_auc", "test_ap")):
            ax = axes[idx, col]
            means = [lookup[(dataset, m)][metric + "_mean"] for m in METHODS]
            errors = [lookup[(dataset, m)][metric + "_sd"] for m in METHODS]
            ax.errorbar(means, np.arange(7), xerr=errors, fmt="none", ecolor="#59636e", capsize=3)
            ax.scatter(means, np.arange(7), c=colors, s=55, zorder=3)
            ax.axvline(0.5, color="#b9bfc5", linestyle=":", linewidth=1)
            ax.set_yticks(np.arange(7), LABELS if col == 0 else [""] * 7)
            ax.invert_yaxis()
            ax.set_xlim(0.45, 1)
            ax.set_xlabel("Test AUC-ROC" if idx == 0 else "Test average precision")
            if idx == 0:
                ax.set_title(dataset if dataset != "PPI" else "PPI training graph 0")
            ax.grid(axis="x", alpha=0.18)
    fig.suptitle(
        "Link prediction: mean and sample SD over 3 shared random splits\nBalanced test negatives; GNNs use node features; 200 training epochs",
        fontsize=13,
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"metrics.{suffix}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        values = [lookup[(dataset, m)]["elapsed_seconds_mean"] for m in METHODS]
        ax.barh(np.arange(7), values, color=colors)
        ax.set_xscale("log")
        ax.set_yticks(np.arange(7), LABELS)
        ax.invert_yaxis()
        ax.set_title(dataset if dataset != "PPI" else "PPI training graph 0")
        ax.set_xlabel("Mean method time (seconds, log scale)")
        for j, value in enumerate(values):
            ax.text(value * 1.12, j, f"{value:.2f}" if value < 1 else f"{value:.1f}", va="center", fontsize=8)
        ax.set_xlim(min(values) / 2, max(values) * 5)
    fig.suptitle(
        "Method time includes fitting, validation selection and test scoring\nJaccard / Adamic-Adar / HOPE / Node2Vec: CPU; GCN / GraphSAGE / GAT: CUDA",
        fontsize=12,
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"runtime.{suffix}")
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, constrained_layout=True)
    for i, dataset in enumerate(DATASETS):
        for j, method in enumerate(METHODS[-3:]):
            ax = axes[i, j]
            for seed in SEEDS:
                history = histories[(dataset, seed, method)]
                ax.plot(
                    [h["epoch"] for h in history],
                    [h["validation"]["auc_roc"] for h in history],
                    label=str(seed),
                    linewidth=1,
                )
            ax.set_title(f"{dataset} / {dict(zip(METHODS, LABELS))[method]}")
            minimum = min(h["validation"]["auc_roc"] for seed in SEEDS for h in histories[(dataset, seed, method)])
            ax.set_ylim(max(0.0, min(0.45, minimum - 0.02)), 1.01)
            ax.grid(alpha=0.2)
            if j == 0:
                ax.set_ylabel("Validation AUC-ROC")
            if i == 2:
                ax.set_xlabel("Epoch")
    axes[0, 0].legend(title="Seed", fontsize=8)
    fig.suptitle("Validation learning curves (checkpoint selected by validation only)", fontsize=13)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"learning-curves.{suffix}")
    plt.close(fig)
    for row in summary:
        print(
            f"{row['dataset']:6} {row['method']:12} AUC {row['test_auc_mean']:.4f} +/- {row['test_auc_sd']:.4f} AP {row['test_ap_mean']:.4f} +/- {row['test_ap_sd']:.4f} time {row['elapsed_seconds_mean']:.2f}s"
        )
    print("Audit passed: 63 cases, 9 shared splits, recomputed saved-prediction metrics.")


if __name__ == "__main__":
    main()
