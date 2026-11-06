import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite vector aligned with the labels.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires both positive and negative labels.")
    return {
        "auc_roc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }
