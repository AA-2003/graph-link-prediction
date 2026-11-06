# Comparison of Graph Link Prediction Methods

A university final project comparing **Jaccard, Adamic–Adar, HOPE, Node2Vec, GCN, GraphSAGE and GAT** using AUC-ROC and average precision.

This version provides an executable Python entry point, a small offline example, shared edge splits and saved experiment reports.

## Quick start

Python 3.11 or 3.12:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python link_prediction.py --dry-run
```

This runs the two classical baselines on NetworkX's karate-club graph without downloading data. Reports are written to a new timestamped directory under `reports/`.

For all seven methods, install the optional dependencies:

```powershell
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[embeddings,gnn,dev]"
python link_prediction.py --dry-run --methods all
```

The extended dry run uses two GNN epochs and short random walks. It verifies the pipeline; its scores are not a benchmark.

For uv, `uv sync --extra embeddings --extra gnn --extra dev` uses the CPU PyTorch index configured in `pyproject.toml`.

## Experiments

```console
python link_prediction.py --dataset Cora --methods all --epochs 100 --dimensions 32
python link_prediction.py --dataset PubMed --methods jaccard adamic-adar gcn sage gat --epochs 100
python link_prediction.py --dataset PPI --ppi-index 0 --methods all --epochs 100
python link_prediction.py --dry-run --methods hope node2vec --tune
```

Cora, PubMed and PPI are downloaded explicitly through PyTorch Geometric and cached in `data/`. PPI uses one graph from its training collection, selected by `--ppi-index`, for transductive link prediction.

Use `--seed`, `--validation-ratio`, `--test-ratio`, `--dimensions`, `--beta`, `--walk-length`, `--num-walks` and `--window` to configure experiments. CPU is the default; `--device cuda` requires a CUDA-enabled PyTorch installation. Existing output directories are refused.

## Evaluation protocol

- Convert each graph to simple, undirected, unweighted edges and retain isolated nodes.
- Split positive edges once: approximately 70% training, 10% validation and 20% test by default.
- Sample unique non-edges at a 1:1 ratio, disjoint across training, validation and test, excluding every observed positive edge.
- Give every method the same training graph and evaluation pairs. GNN message passing uses training edges in both directions.
- Select embedding candidates and GNN checkpoints using validation AUC-ROC, breaking ties with validation AP. Score the selected models on the test set.
- Use a symmetric GNN decoder and symmetric source–target embedding scores for undirected links.

`--tune` compares two embedding configurations: the specified beta and half that beta for HOPE, or the specified context window and twice that window for Node2Vec. Node2Vec uses one worker and a stable hash for repeatability. Results may still vary across hardware or library versions.

HOPE defaults to a dense Katz matrix and truncated SVD, limited to 4,000 nodes. Use `--hope-backend sparse` for larger undirected graphs. The sparse backend computes the leading Katz factors from extreme adjacency eigenpairs, avoiding a dense proximity matrix. Both backends check `beta × spectral_radius < 1` and retain separate source/target factors. Numerical equivalence to the dense low-rank objective is checked on small graphs.

Balanced sampled non-edges and a single seeded split do not establish real-world precision or statistical significance. GNNs use original node features on the downloaded datasets and one-hot node IDs on karate. Their additional feature input should be considered when comparing them with topology-only methods.

## Multi-seed benchmark

The fixed benchmark runs all seven methods on Cora, PubMed and PPI training graph 0 with seeds 42, 123 and 2026: 63 cases. GNNs train for 200 epochs with 64 output dimensions; embeddings use 64 dimensions and two validation-selected configurations. Methods run serially in separate processes.

```console
python -m pip install -e ".[embeddings,gnn,benchmark]"
python benchmark.py --device cpu --output reports/full-benchmark-cpu
python benchmark.py --device cpu --output reports/full-benchmark-cpu --resume
```

For CUDA, select `--device cuda` and point `--python` to an environment with CUDA-enabled PyTorch. The fixed protocol and source fingerprint must match when resuming. CPU and GNN stages can be run separately with `--stage cpu` and `--stage gnn`.

Dataset downloads, model checkpoints and per-case predictions stay under ignored `data/` and `reports/` directories. Curated summaries and charts are in the [23 October 2026 benchmark report](benchmarks/2026-10-23/README.md), including all 63 cases and validation-only improvement probes.

## Outputs

Each run saves:

- `results.json` and `results.csv`: configuration, selected parameters, validation/test metrics, runtime and dependency versions.
- `split.json` and `test_scores.csv`: exact edge pairs and predictions for auditing.
- `scores.png/.svg` and `runtime.png/.svg`: charts for successful methods.
- Embedding arrays and GNN state-dict checkpoints when those methods are run.

Failed or skipped methods have no fabricated zero scores. The notebook `link-prediction.ipynb` runs the small example by default; dataset downloads and full training are opt-in.

## Historical results

The original figures are preserved in [legacy/](legacy/README.md). The original console log is kept locally and excluded from Git. They predate the shared-split evaluation and the HOPE correction and have not been reproduced under this protocol.

## Checks

After installing CPU PyTorch as above:

```console
python -m pip install -e ".[dev,embeddings,gnn]"
python -m unittest discover -s tests -v
python -m ruff check link_prediction.py linkpred tests
python -m nbconvert --execute --to notebook --output-dir reports link-prediction.ipynb
```

## References

- [HOPE: Asymmetric Transitivity Preserving Graph Embedding](https://doi.org/10.1145/2939672.2939751)
- [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)
- [PyTorch Geometric datasets](https://pytorch-geometric.readthedocs.io/en/stable/modules/datasets.html)
