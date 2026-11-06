import csv
import json
import statistics
from pathlib import Path

out = Path("benchmarks/2026-10-23")


def read_csv(name):
    with (out / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def mean(rows, key):
    return statistics.mean(float(r[key]) for r in rows)


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in rows]) + "\n"


def ms(row, key, precision=4):
    return f"{float(row[key + '_mean']):.{precision}f} +/- {float(row[key + '_sd']):.{precision}f}"


summary = read_csv("summary.csv")
raw = read_csv("raw-results.csv")
probe = read_csv("validation-probes.csv")
batch = read_csv("node2vec-batching-probe.csv")
decoder = read_csv("node2vec-decoder-probes.csv")
training = read_csv("training-diagnostics.csv")
sizes = json.loads((out / "dataset-sizes.json").read_text())
protocol = json.loads((out / "protocol.json").read_text())
numeric = json.loads((out / "numerical-checks.json").read_text())
names = {
    "jaccard": "Jaccard",
    "adamic-adar": "Adamic-Adar",
    "hope": "HOPE",
    "node2vec": "Node2Vec",
    "gcn": "GCN",
    "sage": "GraphSAGE",
    "gat": "GAT",
}
datasets = ("Cora", "PubMed", "PPI")
text = [
    """# Link prediction benchmark - 23 October 2026

All 63 method cases completed successfully: seven methods, three graphs and three seeds (42, 123, 2026). GNNs trained for 200 epochs per case. Checkpoints and embedding candidates were selected by validation AUC-ROC, breaking ties with validation AP. This compares these implementations under a shared random-edge protocol; it is not a state-of-the-art claim.

## Results

Values are mean +/- sample standard deviation over three seeded edge splits. The splits overlap within each graph; this measures seed sensitivity, not independent graph replication or statistical significance. AP means average precision on balanced sampled positives/non-edges.

"""
]
winners = []
for d in datasets:
    selected = [r for r in summary if r["dataset"] == d]
    auc = max(selected, key=lambda r: float(r["test_auc_mean"]))
    ap = max(selected, key=lambda r: float(r["test_ap_mean"]))
    winners.append(
        [
            d if d != "PPI" else "PPI graph 0",
            names[auc["method"]],
            ms(auc, "test_auc"),
            names[ap["method"]],
            ms(ap, "test_ap"),
        ]
    )
text.append(table(["Graph", "Highest mean AUC", "AUC", "Highest mean AP", "AP"], winners))
text.append("These are descriptive test rankings. Test results were not used to change this benchmark.\n")
text.append("### Interpretation and priorities\n")
text.append(
    "GCN has the highest mean AUC on all three selected graphs, while HOPE has the highest mean AP on PPI. Continue from GCN as the accuracy baseline under this protocol; retain inexpensive structural methods as speed/quality controls. Prioritize attribute-only and degree-product controls, Node2Vec decoder experiments, and validation-based GNN training settings before adding more complex architectures. Expand graph coverage and ranking evaluation before drawing broader conclusions.\n"
)
for d in datasets:
    text.append("### " + (d if d != "PPI" else "PPI training graph 0") + "\n")
    text.append(
        table(
            ["Method", "Test AUC-ROC", "Test AP", "Method seconds"],
            [
                [names[r["method"]], ms(r, "test_auc"), ms(r, "test_ap"), ms(r, "elapsed_seconds", 2)]
                for r in summary
                if r["dataset"] == d
            ],
        )
    )
text.append("![Mean scores and sample SD](metrics.png)\n")
text.append("## Protocol and scope\n")
text.append(
    table(
        ["Graph", "Nodes", "Unique edges", "Features", "Train / validation / test positives"],
        [
            [
                d,
                sizes[d]["nodes"],
                sizes[d]["positive_edges"],
                sizes[d]["features"],
                f"{sizes[d]['train_pos']} / {sizes[d]['val_pos']} / {sizes[d]['test_pos']}",
            ]
            for d in datasets
        ],
    )
)
text.append("""- Simple, unweighted, undirected graphs with isolated nodes retained; approximately 70/10/20% positive edges.
- Unique 1:1 negative samples, disjoint across all partitions and excluding every observed positive. Unobserved links are assumed negative.
- Identical training graph and evaluation pairs across methods for each graph/seed. Validation/test edges are absent from GNN message passing.
- GNNs additionally receive node attributes transformed by PyG NormalizeFeatures; the other methods use topology only. PPI is training graph index 0, not the full collection or an inductive evaluation.
- Embedding dimension 64. HOPE uses 32 source/32 target factors, sparse Katz decomposition, and beta 0.01 or 0.005 selected on validation.
- Node2Vec uses p=q=1, 20 walks per node, length 20, windows 5/10, one worker, batch_words=4, five Skip-Gram epochs and five noise words per context.
- GNNs use two encoder layers, hidden width 32, output width 64, dropout 0.3, Adam at 0.01, and a symmetric product/absolute-difference MLP decoder. GAT has eight first-layer heads; parameter counts are not matched.
- GNN supervision and message passing share positive training edges. Training negatives remain fixed over epochs.

## Runtime and memory

CPU methods used an Intel i5-11400H and three CPU threads (Node2Vec one worker). GNNs used an RTX 3050 Ti Laptop GPU, 4 GiB, CUDA 13.0 and PyTorch 2.14.1+cu130. System RAM is approximately 16 GiB. Benchmark cases ran serially in separate processes on Windows, on AC power.

Method time includes fitting, validation selection and test scoring; loading, imports, saving and plotting are excluded. Two embedding candidates and 200 GNN epochs represent different computation budgets. Whole-process wall time and memory are in the raw CSV. RSS was sampled once per second; PyTorch GPU peaks exclude context/driver allocations. Environment preparation overlapped the CPU stage. Clocks and other local applications were not controlled.

""")
text.append(
    f"Sum of case wall times: {sum(float(r['wall_seconds']) for r in raw) / 60:.1f} minutes, excluding setup and probes. Maximum sampled process RSS: {max(float(r['peak_rss_mib']) for r in raw):.0f} MiB; maximum PyTorch GPU allocation: {max(float(r['cuda_allocated_mib']) for r in raw):.0f} MiB.\n"
)
text.append("![Mean method time, CPU or CUDA](runtime.png)\n")
text.append("""## Improvements investigated on validation only

The following experiments do not replace the main benchmark. They use training/validation labels only and never produce test scores. Repeated validation inspection can overfit; generalization gains need a new prespecified evaluation.

### Node2Vec batching

""")
values = []
for d in datasets:
    rs = [r for r in batch if r["dataset"] == d]
    original, new = mean(rs, "baseline_method_seconds"), mean(rs, "probe_fit_validation_seconds")
    values.append(
        [
            d,
            f"{original:.1f}",
            f"{new:.1f}",
            f"{original / new:.1f}x",
            f"{mean(rs, 'baseline_validation_auc'):.4f}",
            f"{mean(rs, 'probe_validation_auc'):.4f}",
        ]
    )
text.append(
    table(
        ["Graph", "Batch 4 sec", "Batch 10000 sec", "Approx. time ratio", "Batch 4 val AUC", "Batch 10000 val AUC"],
        values,
    )
)
text.append(
    "Both use one worker and two window candidates. Original time also includes test scoring; probe time covers fitting and validation only, so the ratio is approximate. Larger batches change the optimization trajectory as well as overhead. Expose batching as a parameter and verify quality before changing the default. Also generate random walks once per graph/seed and reuse the corpus across window candidates: the current code regenerates the same seeded walks for each window. [Gensim describes batch_words as the batch target passed to worker/Cython routines.](https://radimrehurek.com/gensim/models/word2vec.html)\n"
)
text.append("### Node2Vec decoder\n")
values = []
for d in datasets:
    rs = [r for r in decoder if r["dataset"] == d]
    learned = [r for r in rs if r["decoder"] == "logistic-product-absdiff"]
    values.append(
        [
            d,
            f"{mean(rs, 'baseline_dot_validation_auc'):.4f}",
            f"{mean([r for r in rs if r['decoder'] == 'cosine'], 'validation_auc'):.4f}",
            f"{mean(learned, 'validation_auc'):.4f}",
            f"{mean(learned, 'additional_fit_validation_seconds'):.2f}",
        ]
    )
text.append(table(["Graph", "Dot val AUC", "Cosine val AUC", "Learned decoder val AUC", "Added sec"], values))
text.append(
    "The learned decoder is a StandardScaler and logistic regression (C=1) on product and absolute-difference pair features, fitted on training labels. It reuses the original validation-selected embeddings. This adds supervised link information, as the GNN decoder does, and is a distinct method. These trials were added after preliminary validation review: [exploration plan](decoder-exploration-plan.json).\n"
)
text.append("### Inexpensive controls\n")
values = []
for d in datasets:
    aa = next(r for r in summary if r["dataset"] == d and r["method"] == "adamic-adar")
    values.append(
        [d, f"{float(aa['validation_auc_mean']):.4f}"]
        + [
            f"{mean([r for r in probe if r['dataset'] == d and r['method'] == m], 'validation_auc'):.4f}"
            for m in ("common-neighbors", "resource-allocation", "preferential-attachment", "feature-cosine")
        ]
    )
text.append(
    table(
        ["Graph", "Adamic-Adar", "Common neighbors", "Resource Allocation", "Degree product", "Feature cosine"], values
    )
)
text.append(
    "Values are mean validation AUC, not test results. Feature cosine is an attribute-only control. Add a learned feature-only MLP and GNN feature ablations to separate attribute gains from message passing. [Graph diagnostics](graph-diagnostics.csv) record isolated nodes, components and common-neighbor coverage.\n"
)
text.append("### GNN training\n")
values = []
for d in datasets:
    for m in ("gcn", "sage", "gat"):
        rs = [r for r in training if r["dataset"] == d and r["method"] == m]
        epochs = [int(r["best_epoch"]) for r in rs]
        values.append(
            [
                d,
                names[m],
                f"{int(statistics.median(epochs))} ({min(epochs)}-{max(epochs)})",
                f"{mean(rs, 'final_auc_drop'):.4f}",
            ]
        )
text.append(table(["Graph", "Model", "Best epoch median (range)", "Final val AUC drop from best"], values))
text.append(
    "Use validation early stopping beyond a plateau. Where checkpoints remain near epoch 200, test a larger budget under a new protocol. Compare learning rate, dropout and width with equal validation budgets. Investigate resampled training negatives and disjoint message-passing/supervision edges; [PyG supports disjoint_train_ratio](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.RandomLinkSplit.html). None guarantees an improvement without testing.\n"
)
text.append("![Validation learning curves](learning-curves.png)\n")
text.append("""### Stronger evaluation

Add multiple PPI graphs and a separately defined held-out-graph evaluation. For recommendation-style use, preregister harder/larger candidate pools and Hits@K/MRR; temporal splits require suitable data. [OGB illustrates ranking metrics and temporal/domain splits](https://ogb.stanford.edu/docs/linkprop/). Balanced AP here does not estimate precision among all possible graph pairs.

## Verification and artifacts

""")
text.append(
    f"All 63 prediction files reproduce reported AUC/AP. Nine graph/seed groups have identical split dictionaries, test pairs/labels and feature hashes across methods. All six edge partitions are unique and mutually disjoint. Nine sparse HOPE eigenpair checks have maximum relative residual {max(r['relative_adjacency_eigenpair_residual'] for r in numeric['hope_eigenpair_checks']):.2e}. All 27 GNN checkpoints were reloaded on CPU and replayed; maximum absolute difference from saved CUDA predictions was {max(r['maximum_absolute_prediction_error_vs_saved_cuda_scores'] for r in numeric['gnn_checkpoint_replays']):.2e}.\n"
)
text.append(
    "The unit suite passed 21 tests, and Ruff passed. Small-graph tests verify full/truncated sparse HOPE against dense Katz and forbid densification on its sparse path. Seeds and disabled TF32 reduce variation; strict deterministic CUDA algorithms are not enabled. [PyTorch does not guarantee identical results across devices, versions or platforms.](https://docs.pytorch.org/docs/2.14/notes/randomness.html)\n"
)
text.append(f"Algorithm-source SHA-256: **{protocol['source_sha256']}**.\n")
text.append(
    "Artifacts: [raw results](raw-results.csv), [summary](summary.csv), [training diagnostics](training-diagnostics.csv), [split audit](audit.json), [numerical checks](numerical-checks.json), [protocol](protocol.json), [source hashes](source-files-sha256.json), [hardware](hardware.json), [versions](environment-versions.json). Exact pairs, logs, arrays and checkpoints remain locally under ignored reports/full-benchmark/.\n"
)
text.append("""## Reproduce

CPU-only run, using a new output directory:

~~~powershell
uv sync --locked --extra embeddings --extra gnn --extra benchmark
.venv\\Scripts\\python.exe benchmark.py --device cpu --output reports/reproduction-cpu
~~~

The recorded run used the CPU environment for four methods and a separate CUDA environment for GNNs:

~~~powershell
uv venv --python 3.11 --no-project .venv-benchmark
uv pip install --python .venv-benchmark\\Scripts\\python.exe --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.1"
uv pip install --python .venv-benchmark\\Scripts\\python.exe --no-deps -r benchmarks/2026-10-23/environment-requirements-without-torch.txt -e .
uv pip check --python .venv-benchmark\\Scripts\\python.exe
.venv\\Scripts\\python.exe benchmark.py --stage cpu --device cuda --output reports/reproduction-cuda
.venv\\Scripts\\python.exe benchmark.py --stage gnn --device cuda --python .venv-benchmark\\Scripts\\python.exe --output reports/reproduction-cuda --resume
~~~

In the CPU stage, the device flag reserves the GNN protocol for the next stage; the four methods still compute on CPU. Resume reuses successful cases with matching protocol/source hash. The environment requirement file includes Windows packages; other platforms may need a newly resolved environment.

Analysis and validation-probe scripts are retained under [scripts/](scripts/). No remote repository was created or updated.
""")
(out / "README.md").write_text("\n".join(text), encoding="utf-8", newline="\n")
print("Wrote", out / "README.md")
