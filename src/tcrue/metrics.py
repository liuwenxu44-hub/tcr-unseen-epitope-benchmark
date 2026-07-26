from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    metrics: dict[str, float] = {}
    if len(np.unique(labels)) > 1:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
        metrics["auprc"] = float(average_precision_score(labels, scores))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    return metrics


def recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    top = order[:k]
    return float(labels[top].sum() / labels.sum())
