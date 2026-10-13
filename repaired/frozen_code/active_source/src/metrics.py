from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_classification_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    label_array = np.asarray(labels, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if label_array.ndim != 1 or probability_array.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional")
    if label_array.size == 0 or label_array.size != probability_array.size:
        raise ValueError("labels and probabilities must have equal non-zero length")
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must be finite")
    if not set(np.unique(label_array)).issubset({0, 1}):
        raise ValueError("labels must contain only 0 and 1")
    if np.unique(label_array).size != 2:
        raise ValueError("AUROC and AUPRC require both label classes")

    return {
        "auroc": float(roc_auc_score(label_array, probability_array)),
        "auprc": float(average_precision_score(label_array, probability_array)),
    }


def query_level_classification_metrics(
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    if not predictions:
        raise ValueError("query-level metrics require non-empty predictions")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        query_id = str(row.get("pmhc_id", "")).strip().upper()
        if "|" not in query_id:
            raise ValueError(f"invalid or missing pMHC query ID: {query_id!r}")
        grouped[query_id].append(row)

    per_query = []
    for query_id in sorted(grouped):
        rows = grouped[query_id]
        labels = [int(row["label"]) for row in rows]
        probabilities = [float(row["probability"]) for row in rows]
        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        if positive_count == 0 or negative_count == 0:
            raise ValueError(f"query lacks both label classes: {query_id}")
        epitope, hla = query_id.rsplit("|", 1)
        metrics = binary_classification_metrics(labels, probabilities)
        per_query.append(
            {
                "query_id": query_id,
                "epitope": epitope,
                "hla": hla,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "sample_count": len(rows),
                **metrics,
            }
        )

    return (
        {
            "query_count": len(per_query),
            "auroc": float(np.mean([row["auroc"] for row in per_query])),
            "auprc": float(np.mean([row["auprc"] for row in per_query])),
        },
        per_query,
    )
