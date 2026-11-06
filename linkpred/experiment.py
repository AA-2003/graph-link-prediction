import csv
import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from linkpred.data import load_dataset
from linkpred.methods import ResourceLimit, classic_scores, hope_embedding, node2vec_embedding
from linkpred.metrics import evaluate
from linkpred.plotting import plot_results
from linkpred.split import split_edges

METHODS = ("jaccard", "adamic-adar", "hope", "node2vec", "gcn", "sage", "gat")


@dataclass
class RunConfig:
    dataset: str = "karate"
    seed: int = 42
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    dimensions: int = 32
    epochs: int = 100
    beta: float = 0.01
    max_hope_nodes: int = 4000
    walk_length: int = 10
    num_walks: int = 10
    window: int = 5
    tune: bool = False
    device: str = "cpu"
    threads: int = 2
    ppi_index: int = 0
    hope_backend: str = "dense"
    log_every: int = 0

    def validate(self):
        if (
            min(
                self.dimensions,
                self.epochs,
                self.max_hope_nodes,
                self.walk_length,
                self.num_walks,
                self.window,
                self.threads,
            )
            < 1
        ):
            raise ValueError("Dimensions, epochs, resource limits and walk settings must be positive.")
        if not np.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("Beta must be finite and positive.")
        if not 0 <= self.seed < 2**32 - 1:
            raise ValueError("Seed must be between 0 and 2**32 - 2.")
        if self.hope_backend not in {"dense", "sparse"}:
            raise ValueError("HOPE backend must be dense or sparse.")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Device must be cpu or cuda.")


def embedding_candidates(method, config):
    if method == "hope":
        base = {"dimensions": config.dimensions, "beta": config.beta}
        candidates = [base, {**base, "beta": config.beta / 2}] if config.tune else [base]
    else:
        base = {
            "dimensions": config.dimensions,
            "walk_length": config.walk_length,
            "num_walks": config.num_walks,
            "window": config.window,
        }
        candidates = [base, {**base, "window": config.window * 2}] if config.tune else [base]
    return candidates


def select_embedding(graph, split, method, config):
    validation_pairs, validation_labels = split.examples("val")
    best_embedding, best_params, best_metrics, best_key = None, None, None, (-1.0, -1.0)
    trials = []
    last_error = None
    for params in embedding_candidates(method, config):
        try:
            if method == "hope":
                if config.hope_backend == "sparse":
                    from linkpred.sparse_hope import sparse_hope_embedding

                    embedding = sparse_hope_embedding(
                        graph, **params, seed=config.seed, max_dense_nodes=config.max_hope_nodes
                    )
                else:
                    embedding = hope_embedding(graph, **params, seed=config.seed, max_nodes=config.max_hope_nodes)
            else:
                embedding = node2vec_embedding(graph, **params, seed=config.seed)
            metrics = evaluate(validation_labels, embedding.scores(validation_pairs))
            trials.append({"parameters": params, "validation": metrics, "status": "ok"})
            key = (metrics["auc_roc"], metrics["average_precision"])
            if key > best_key:
                best_embedding, best_params, best_metrics, best_key = embedding, params, metrics, key
        except (ValueError, RuntimeError) as error:
            trials.append({"parameters": params, "status": "error", "reason": str(error)})
            last_error = error
    if best_embedding is None:
        raise last_error or ValueError("No embedding candidates were configured.")
    return best_embedding, best_params, best_metrics, trials


def source_fingerprint():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob("*.py")) + [root.parent / "link_prediction.py"]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "networkx", "scipy", "scikit-learn", "node2vec", "gensim", "torch", "torch-geometric"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    return versions


def run_experiment(config=None, methods=("jaccard", "adamic-adar"), output="reports/run", cache="data"):
    config = config or RunConfig()
    config.validate()
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError("Choose distinct supported methods.")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new directory.")
    dataset = load_dataset(config.dataset, cache, config.ppi_index)
    split = split_edges(dataset.graph, config.seed, config.val_ratio, config.test_ratio)
    graph = split.train_graph
    output.mkdir(parents=True)
    (output / "split.json").write_text(json.dumps(split.to_dict(), indent=2) + "\n", encoding="utf-8")
    test_pairs, test_labels = split.examples("test")
    val_pairs, val_labels = split.examples("val")
    results, predictions = [], []
    for method in methods:
        started = None
        embedding, fitted = None, None
        result = {"method": method, "split_id": split.identifier}
        print(f"Running {method}...", flush=True)
        try:
            if method == "node2vec":
                import_module("node2vec")
            elif method in {"gcn", "sage", "gat"}:
                try:
                    import torch
                    from linkpred.gnn import fit_gnn
                except ImportError:
                    raise RuntimeError("Install the gnn extra to run GCN, GraphSAGE or GAT.") from None
                torch.set_num_threads(config.threads)
                if config.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            with threadpool_limits(limits=config.threads):
                started = time.perf_counter()
                if method in {"jaccard", "adamic-adar"}:
                    scores = classic_scores(graph, test_pairs, method)
                    result["validation"] = evaluate(val_labels, classic_scores(graph, val_pairs, method))
                    result["parameters"] = {}
                elif method in {"hope", "node2vec"}:
                    embedding, params, validation, trials = select_embedding(graph, split, method, config)
                    scores = embedding.scores(test_pairs)
                    result.update(parameters=params, validation=validation, trials=trials)
                    if method == "hope":
                        result["hope_backend"] = config.hope_backend
                else:
                    fitted = fit_gnn(
                        dataset.features,
                        split,
                        method,
                        config.epochs,
                        config.dimensions,
                        seed=config.seed,
                        device=config.device,
                        log_every=config.log_every,
                    )
                    scores = fitted.scores(test_pairs)
                    if config.device == "cuda":
                        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                        result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                    result.update(
                        parameters=fitted.model.config, validation=fitted.validation, best_epoch=fitted.best_epoch
                    )
                result.update(status="ok", test=evaluate(test_labels, scores))
                result["elapsed_seconds"] = time.perf_counter() - started
            if embedding is not None:
                np.savez_compressed(
                    output / f"{method}_embedding.npz", source=embedding.source, target=embedding.target
                )
            if fitted is not None:
                fitted.save(output / f"{method}_checkpoint.pth", split.identifier)
                (output / f"{method}_history.json").write_text(
                    json.dumps(fitted.history, indent=2) + "\n", encoding="utf-8"
                )
            predictions.extend(
                {"method": method, "u": int(u), "v": int(v), "label": int(label), "score": float(score)}
                for (u, v), label, score in zip(test_pairs, test_labels, scores)
            )
        except ResourceLimit as error:
            result.update(status="skipped", reason=str(error))
        except (ImportError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            result.update(status="error", reason=str(error))
        if "elapsed_seconds" not in result:
            result["elapsed_seconds"] = time.perf_counter() - started if started is not None else None
        results.append(result)
        print(f"{method}: {result['status']}" + (f" - {result['reason']}" if "reason" in result else ""), flush=True)
    metadata = {
        "configuration": asdict(config),
        "dataset": dataset.name,
        "description": dataset.description,
        "nodes": len(graph),
        "positive_edges": sum(len(getattr(split, part + "_pos")) for part in ("train", "val", "test")),
        "features": int(dataset.features.shape[1]),
        "split_id": split.identifier,
        "features_sha256": hashlib.sha256(np.ascontiguousarray(dataset.features).tobytes()).hexdigest(),
        "versions": _versions(),
        "source_sha256": source_fingerprint(),
        "negative_sampling": "Unique disjoint non-edges, 1:1 with positives in each split.",
        "timing": "Fit, validation selection and test scoring; excludes dependency imports, loading, splitting, file writes and plots.",
    }
    report = {"metadata": metadata, "results": results}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("method", "status", "validation_auc", "test_auc", "test_ap", "elapsed_seconds", "reason"),
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "method": row["method"],
                    "status": row["status"],
                    "validation_auc": row.get("validation", {}).get("auc_roc"),
                    "test_auc": row.get("test", {}).get("auc_roc"),
                    "test_ap": row.get("test", {}).get("average_precision"),
                    "elapsed_seconds": row["elapsed_seconds"],
                    "reason": row.get("reason", ""),
                }
            )
    with (output / "test_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("method", "u", "v", "label", "score"))
        writer.writeheader()
        writer.writerows(predictions)
    plot_results(results, output, metadata)
    return report
