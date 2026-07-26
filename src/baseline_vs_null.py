from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .baseline_paired_statistics import (
    METHODS,
    _audit_and_join,
    _load_json,
    _load_yaml,
    _relative,
    _resolve,
    _verified_explicit_csv,
    _verified_summary_output,
    _verify_summary_input_hashes,
)
from .five_fold_summary import file_sha256
from .metrics import binary_classification_metrics, query_level_classification_metrics


METRIC_NAMES = ("pooled_auroc", "pooled_auprc", "macro_auroc", "macro_auprc")
P_VALUE_COMPARISON_TOLERANCE = 1e-12
CSV_COLUMNS = (
    "method",
    "metric",
    "query_count",
    "sample_count",
    "observed",
    "null_mean",
    "observed_minus_null_mean",
    "p_value_one_sided",
    "p_value_holm",
    "null_ci_lower",
    "null_ci_upper",
    "permutation_iterations",
    "permutation_seed",
    "confidence_level",
    "alternative",
    "permutation_unit",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        raise ValueError("Holm adjustment requires at least one p-value")
    for name, value in p_values.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"invalid p-value for {name}: {value}")
    ordered = sorted(p_values.items(), key=lambda item: (float(item[1]), item[0]))
    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, float(value) * (total - rank))
        running_maximum = max(running_maximum, candidate)
        adjusted[name] = running_maximum
    return adjusted


def permuted_correspondence_labels(
    labels: np.ndarray,
    query_indices: Sequence[np.ndarray],
    *,
    rng: np.random.Generator,
    batch_size: int,
) -> np.ndarray:
    if labels.ndim != 1 or batch_size <= 0:
        raise ValueError("labels must be one-dimensional and batch_size positive")
    permuted = np.zeros((batch_size, labels.size), dtype=np.uint8)
    covered = np.zeros(labels.size, dtype=np.uint8)
    for indices in query_indices:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0 or np.any(covered[indices]):
            raise ValueError("query index groups must be non-empty and disjoint")
        covered[indices] = 1
        positive_count = int(labels[indices].sum())
        if positive_count <= 0 or positive_count >= indices.size:
            raise ValueError("every query must contain both label classes")
        random_keys = rng.random((batch_size, indices.size))
        selected = np.argpartition(
            random_keys, positive_count - 1, axis=1
        )[:, :positive_count]
        local = np.zeros((batch_size, indices.size), dtype=np.uint8)
        np.put_along_axis(local, selected, 1, axis=1)
        permuted[:, indices] = local
    if not np.all(covered == 1):
        raise ValueError("query index groups must cover every sample exactly once")
    return permuted


def _score_groups(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    sizes = np.diff(np.r_[starts, scores.size]).astype(np.float64)
    return order, starts, sizes


def _metrics_from_label_matrix(
    labels: np.ndarray,
    score_groups: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    order, starts, sizes = score_groups
    sorted_labels = labels[:, order].astype(np.float64)
    positive_by_score = np.add.reduceat(sorted_labels, starts, axis=1)
    negative_by_score = sizes[None, :] - positive_by_score
    cumulative_positive = np.cumsum(positive_by_score, axis=1)
    cumulative_negative = np.cumsum(negative_by_score, axis=1)
    total_positive = cumulative_positive[:, -1]
    total_negative = cumulative_negative[:, -1]
    if np.any(total_positive <= 0) or np.any(total_negative <= 0):
        raise ValueError("permutation replicate lacks a label class")
    precision = cumulative_positive / (cumulative_positive + cumulative_negative)
    auprc = np.sum(positive_by_score * precision, axis=1) / total_positive
    negative_below = total_negative[:, None] - cumulative_negative
    auroc = np.sum(
        positive_by_score * (negative_below + 0.5 * negative_by_score), axis=1
    ) / (total_positive * total_negative)
    return auroc, auprc


def query_permutation_null_test(
    predictions: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = METHODS,
    seed: int,
    iterations: int,
    batch_size: int,
    confidence_level: float,
) -> dict[str, Any]:
    if not predictions or iterations <= 0 or batch_size <= 0:
        raise ValueError("predictions, iterations, and batch size must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between zero and one")
    labels = np.asarray([int(row["label"]) for row in predictions], dtype=np.uint8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("predictions must contain both binary label classes")
    query_positions: dict[str, list[int]] = defaultdict(list)
    query_folds: dict[str, str] = {}
    for index, row in enumerate(predictions):
        query_id = str(row["query_id"])
        fold_id = str(row["fold_id"])
        if query_id in query_folds and query_folds[query_id] != fold_id:
            raise ValueError(f"query appears in multiple folds: {query_id}")
        query_folds[query_id] = fold_id
        query_positions[query_id].append(index)
    ordered_query_ids = sorted(query_positions, key=lambda query_id: (query_folds[query_id], query_id))
    query_indices = [np.asarray(query_positions[query_id], dtype=np.int64) for query_id in ordered_query_ids]
    method_scores = {
        method: np.asarray(
            [float(row[f"{method}_score"]) for row in predictions], dtype=np.float64
        )
        for method in methods
    }
    if any(not np.isfinite(scores).all() for scores in method_scores.values()):
        raise ValueError("method scores must be finite")

    observed: dict[str, dict[str, float]] = {}
    global_groups = {}
    local_groups: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for method, scores in method_scores.items():
        pooled = binary_classification_metrics(labels, scores)
        macro, _ = query_level_classification_metrics(
            [
                {
                    "pmhc_id": row["query_id"],
                    "label": row["label"],
                    "probability": row[f"{method}_score"],
                }
                for row in predictions
            ]
        )
        observed[method] = {
            "pooled_auroc": pooled["auroc"],
            "pooled_auprc": pooled["auprc"],
            "macro_auroc": float(macro["auroc"]),
            "macro_auprc": float(macro["auprc"]),
        }
        global_groups[method] = _score_groups(scores)
        local_groups[method] = [_score_groups(scores[indices]) for indices in query_indices]

    null = {
        method: {metric: np.empty(iterations, dtype=np.float64) for metric in METRIC_NAMES}
        for method in methods
    }
    rng = np.random.default_rng(seed)
    query_count = len(query_indices)
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        current_size = stop - start
        permuted_labels = permuted_correspondence_labels(
            labels, query_indices, rng=rng, batch_size=current_size
        )
        for method in methods:
            pooled_auroc, pooled_auprc = _metrics_from_label_matrix(
                permuted_labels, global_groups[method]
            )
            null[method]["pooled_auroc"][start:stop] = pooled_auroc
            null[method]["pooled_auprc"][start:stop] = pooled_auprc
            macro_auroc = np.zeros(current_size, dtype=np.float64)
            macro_auprc = np.zeros(current_size, dtype=np.float64)
            for indices, groups in zip(query_indices, local_groups[method], strict=True):
                query_auroc, query_auprc = _metrics_from_label_matrix(
                    permuted_labels[:, indices], groups
                )
                macro_auroc += query_auroc / query_count
                macro_auprc += query_auprc / query_count
            null[method]["macro_auroc"][start:stop] = macro_auroc
            null[method]["macro_auprc"][start:stop] = macro_auprc

    alpha = 1.0 - confidence_level
    raw_p_values = {}
    statistics: dict[str, dict[str, Any]] = {}
    for method in methods:
        statistics[method] = {}
        for metric in METRIC_NAMES:
            values = null[method][metric]
            null_mean = float(values.mean())
            observed_value = observed[method][metric]
            p_value = float(
                (
                    1
                    + np.sum(
                        values >= observed_value - P_VALUE_COMPARISON_TOLERANCE
                    )
                )
                / (iterations + 1)
            )
            ci_lower, ci_upper = np.quantile(
                values, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear"
            )
            key = f"{method}:{metric}"
            raw_p_values[key] = p_value
            statistics[method][metric] = {
                "observed": observed_value,
                "null_mean": null_mean,
                "observed_minus_null_mean": observed_value - null_mean,
                "p_value_one_sided": p_value,
                "null_ci_lower": float(ci_lower),
                "null_ci_upper": float(ci_upper),
            }
    adjusted = holm_adjust(raw_p_values)
    for method in methods:
        for metric in METRIC_NAMES:
            statistics[method][metric]["p_value_holm"] = adjusted[f"{method}:{metric}"]
    return {
        "statistics": statistics,
        "query_count": query_count,
        "sample_count": len(predictions),
        "query_counts_by_fold": dict(sorted(defaultdict(int, {fold: list(query_folds.values()).count(fold) for fold in set(query_folds.values())}).items())),
        "iterations": iterations,
        "seed": seed,
        "batch_size": batch_size,
        "confidence_level": confidence_level,
        "alternative": "greater",
        "permutation_unit": "query",
        "score_distributions_preserved": True,
        "labels_and_sample_counts_preserved": True,
        "shared_permutations_across_methods": True,
        "p_value_comparison_tolerance": P_VALUE_COMPARISON_TOLERANCE,
    }


def _csv_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    permutation = result["permutation"]
    rows = []
    for method in METHODS:
        for metric in METRIC_NAMES:
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "query_count": permutation["query_count"],
                    "sample_count": permutation["sample_count"],
                    **result["statistics"][method][metric],
                    "permutation_iterations": permutation["iterations"],
                    "permutation_seed": permutation["seed"],
                    "confidence_level": permutation["confidence_level"],
                    "alternative": permutation["alternative"],
                    "permutation_unit": permutation["permutation_unit"],
                }
            )
    return rows


def _render_report(result: Mapping[str, Any]) -> str:
    permutation = result["permutation"]
    lines = [
        "# Baselines versus permutation null",
        "",
        f"- Common evaluable queries: {permutation['query_count']}",
        f"- Common samples: {permutation['sample_count']}",
        f"- Permutations: {permutation['iterations']:,}, fixed seed {permutation['seed']}",
        "- Scores are permuted within each query; labels, sample counts, and each method's query-level score distribution remain fixed.",
        "- P-values are one-sided for performance greater than the permutation null and are adjusted jointly across 12 tests using Holm's method.",
    ]
    for method in METHODS:
        lines.extend(
            [
                "",
                f"## {method}",
                "",
                "| Metric | Observed | Null mean | Difference | One-sided p | Holm p | 95% null interval |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for metric in METRIC_NAMES:
            row = result["statistics"][method][metric]
            lines.append(
                f"| {metric} | {row['observed']:.6f} | {row['null_mean']:.6f} | "
                f"{row['observed_minus_null_mean']:+.6f} | {row['p_value_one_sided']:.6g} | "
                f"{row['p_value_holm']:.6g} | [{row['null_ci_lower']:.6f}, {row['null_ci_upper']:.6f}] |"
            )
    lines.extend(
        [
            "",
            "No training, rescoring, or modification of existing baseline results was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def build_baseline_vs_null(config_path: Path, project_dir: Path) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    permutation_config = config["permutation"]
    if permutation_config["operation"] != "permute_score_label_correspondence_within_query":
        raise ValueError("permutation operation must be query-conditioned")
    if config["multiple_testing"] != {"method": "holm", "family": "3_methods_x_4_metrics"}:
        raise ValueError("multiple-testing configuration must be Holm across 12 tests")
    output_directory = _resolve(project_dir, config["outputs"]["directory"])
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_directory}")

    manifest_path = _resolve(project_dir, config["inputs"]["manifest"])
    cnn_summary_path = _resolve(project_dir, config["inputs"]["cnn_summary"])
    logistic_summary_path = _resolve(project_dir, config["inputs"]["logistic_summary"])
    similarity_summary_path = _resolve(project_dir, config["inputs"]["similarity_summary"])
    manifest = _load_json(manifest_path)
    cnn_summary = _load_json(cnn_summary_path)
    logistic_summary = _load_json(logistic_summary_path)
    similarity_summary = _load_json(similarity_summary_path)
    manifest_sha = file_sha256(manifest_path)
    for method, summary in (("CNN", cnn_summary), ("Logistic", logistic_summary), ("Similarity", similarity_summary)):
        if summary["audit"]["manifest"]["sha256"] != manifest_sha:
            raise ValueError(f"{method} manifest SHA-256 mismatch")
    _verify_summary_input_hashes(manifest, cnn_summary, logistic_summary, similarity_summary)

    cnn_oof_path, cnn_rows, cnn_oof_audit = _verified_summary_output(
        project_dir, cnn_summary, "oof_predictions_csv", "CNN OOF predictions"
    )
    cnn_query_path, cnn_per_query, cnn_query_audit = _verified_summary_output(
        project_dir, cnn_summary, "oof_per_query_csv", "CNN per-query metrics"
    )
    logistic_oof_path, logistic_rows, logistic_oof_audit = _verified_explicit_csv(
        project_dir, config["inputs"]["logistic_oof_predictions"], "concatenated fold test predictions"
    )
    logistic_query_path, logistic_per_query, logistic_query_audit = _verified_explicit_csv(
        project_dir, config["inputs"]["logistic_per_query_metrics"], "concatenated fold per-query metrics"
    )
    similarity_oof_path, similarity_rows, similarity_oof_audit = _verified_summary_output(
        project_dir, similarity_summary, "oof_predictions_csv", "Similarity OOF predictions"
    )
    similarity_query_path, similarity_per_query, similarity_query_audit = _verified_summary_output(
        project_dir, similarity_summary, "per_query_csv", "Similarity per-query metrics"
    )
    canonical, common_per_query, paired_audit = _audit_and_join(
        project_dir=project_dir,
        manifest=manifest,
        cnn_rows=cnn_rows,
        cnn_per_query=cnn_per_query,
        logistic_rows=logistic_rows,
        logistic_per_query=logistic_per_query,
        similarity_rows=similarity_rows,
        similarity_per_query=similarity_per_query,
    )
    available_ids = {row["query_id"] for row in common_per_query}
    common_rows = [row for row in canonical if row["query_id"] in available_ids]
    if len(available_ids) != 153:
        raise ValueError("permutation test requires exactly 153 common queries")
    permutation = query_permutation_null_test(
        common_rows,
        seed=int(permutation_config["seed"]),
        iterations=int(permutation_config["iterations"]),
        batch_size=int(permutation_config["batch_size"]),
        confidence_level=float(permutation_config["confidence_level"]),
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "statistics": permutation.pop("statistics"),
        "permutation": permutation,
        "multiple_testing": {"method": "holm", "test_count": 12, "family": "3_methods_x_4_metrics"},
        "audit": {
            "status": "passed",
            "manifest": {"path": _relative(project_dir, manifest_path), "sha256": manifest_sha},
            "paired_inputs": paired_audit,
            "input_artifacts": {
                "cnn_summary": {"path": _relative(project_dir, cnn_summary_path), "sha256": file_sha256(cnn_summary_path)},
                "cnn_oof_predictions": cnn_oof_audit,
                "cnn_per_query_metrics": cnn_query_audit,
                "logistic_summary": {"path": _relative(project_dir, logistic_summary_path), "sha256": file_sha256(logistic_summary_path)},
                "logistic_oof_predictions": logistic_oof_audit,
                "logistic_per_query_metrics": logistic_query_audit,
                "similarity_summary": {"path": _relative(project_dir, similarity_summary_path), "sha256": file_sha256(similarity_summary_path)},
                "similarity_oof_predictions": similarity_oof_audit,
                "similarity_per_query_metrics": similarity_query_audit,
            },
            "source_paths": {
                "cnn_oof": _relative(project_dir, cnn_oof_path),
                "cnn_per_query": _relative(project_dir, cnn_query_path),
                "logistic_oof": _relative(project_dir, logistic_oof_path),
                "logistic_per_query": _relative(project_dir, logistic_query_path),
                "similarity_oof": _relative(project_dir, similarity_oof_path),
                "similarity_per_query": _relative(project_dir, similarity_query_path),
            },
            "query_ids_match": True,
            "fold_ids_match": True,
            "labels_match": True,
            "interaction_keys_match": True,
            "input_sha256_consistent_with_manifest": True,
            "common_query_count": len(available_ids),
            "training_performed": False,
            "rescoring_performed": False,
            "source_results_modified": False,
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / config["outputs"]["statistics_csv"]
    json_path = output_directory / config["outputs"]["statistics_json"]
    report_path = output_directory / config["outputs"]["report_md"]
    _write_csv(csv_path, _csv_rows(result))
    report_path.write_text(_render_report(result), encoding="utf-8")
    result["outputs"] = {
        "statistics_csv": {"path": _relative(project_dir, csv_path), "sha256": file_sha256(csv_path)},
        "report_md": {"path": _relative(project_dir, report_path), "sha256": file_sha256(report_path)},
    }
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
