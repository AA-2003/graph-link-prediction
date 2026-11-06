from pathlib import Path


def plot_results(results, output, metadata=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    successful = [row for row in results if row["status"] == "ok"]
    if not successful:
        return
    output = Path(output)
    metadata = metadata or {}
    dataset_name = metadata.get("dataset", "Graph")
    configuration = metadata.get("configuration", {})
    note = f"Seed {configuration.get('seed', '?')}; balanced positive/negative test samples"
    if any(row["method"] in {"gcn", "sage", "gat"} for row in successful):
        note += f"; GNN training: at most {configuration.get('epochs', '?')} epochs"
    names = [row["method"] for row in successful]
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for offset, metric, title, color in (
        (-0.2, "auc_roc", "AUC-ROC", "#3267a8"),
        (0.2, "average_precision", "Average precision", "#c45d44"),
    ):
        bars = ax.bar(
            positions + offset, [row["test"][metric] for row in successful], width=0.38, label=title, color=color
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set(
        xticks=positions,
        xticklabels=names,
        ylim=(0, 1.12),
        ylabel="Score",
        title=f"{dataset_name}: link prediction on the shared test split",
    )
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(loc="upper left", ncols=2)
    if metadata:
        fig.supxlabel(note, fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(output / ("scores." + extension), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    bars = ax.bar(names, [row["elapsed_seconds"] for row in successful], color="#38785e")
    ax.bar_label(bars, fmt="%.3fs", padding=3, fontsize=9)
    ax.set(
        yscale="log",
        ylabel="Elapsed time (seconds, log scale)",
        title=f"{dataset_name}: fit, validation selection and test scoring time",
    )
    ax.margins(y=0.2)
    if metadata:
        fig.supxlabel(note, fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(output / ("runtime." + extension), dpi=180)
    plt.close(fig)
