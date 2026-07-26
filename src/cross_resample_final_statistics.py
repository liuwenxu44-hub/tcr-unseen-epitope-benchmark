from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .baseline_vs_null import holm_adjust
from .five_fold_summary import file_sha256
from .interaction_resample_robustness import METHODS, RESAMPLE_IDS
from .kmer_logistic import METRIC_NAMES, _load_json, _load_yaml, _relative, _resolve
from .metrics import binary_classification_metrics
from .query_balanced_paired_statistics import (
    _quantile_interval,
    _weighted_pooled_replicates,
    stratified_query_bootstrap_counts,
)


COMPARISON = "orthogonalized_minus_additive_only"
FOLD_IDS = tuple(f"fold_{index}" for index in range(1, 6))
CSV_COLUMNS = (
    "record_type",
    "excluded_resample",
    "metric",
    "difference",
    "ci_lower",
    "ci_upper",
    "p_value_two_sided",
    "p_value_holm",
    "positive_combination_count",
    "combination_count",
    "all_leave_one_out_differences_positive",
    "bootstrap_iterations",
    "bootstrap_seed",
    "randomization_iterations",
    "randomization_seed",
)


def _locked_json(
    project_dir: Path, source: Mapping[str, str], description: str
) -> tuple[Path, dict[str, Any]]:
    path = _resolve(project_dir, source["path"])
    observed = file_sha256(path)
    if observed != source["sha256"]:
        raise ValueError(f"{description} SHA-256 mismatch")
    return path, _load_json(path)


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def _read_gzip_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    metadata = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": contiguous.shape},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(metadata + b"\0" + contiguous.tobytes()).hexdigest()


def generate_randomization_masks(
    query_count: int, *, seed: int, iterations: int
) -> np.ndarray:
    if query_count <= 0 or iterations <= 0:
        raise ValueError("query count and randomization iterations must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(iterations, query_count), dtype=np.uint8).astype(
        np.bool_
    )


def _batched_unweighted_metrics(
    labels: np.ndarray, score_rows: np.ndarray
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(score_rows, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 2 or scores.shape[1] != labels.size:
        raise ValueError("batched score matrix and labels are incompatible")
    if not np.isfinite(scores).all() or not set(np.unique(labels)).issubset({0.0, 1.0}):
        raise ValueError("scores must be finite and labels binary")
    total_positive = float(labels.sum())
    total_negative = float(labels.size - total_positive)
    if total_positive <= 0.0 or total_negative <= 0.0:
        raise ValueError("both classes are required")

    order = np.argsort(-scores, axis=1, kind="stable")
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    sorted_labels = labels[order]
    cumulative_positive = np.cumsum(sorted_labels, axis=1)
    positions = np.arange(1, labels.size + 1, dtype=np.float64)[None, :]
    cumulative_negative = positions - cumulative_positive

    group_start = np.empty(sorted_scores.shape, dtype=np.bool_)
    group_start[:, 0] = True
    group_start[:, 1:] = sorted_scores[:, 1:] != sorted_scores[:, :-1]
    group_end = np.empty(sorted_scores.shape, dtype=np.bool_)
    group_end[:, -1] = True
    group_end[:, :-1] = sorted_scores[:, :-1] != sorted_scores[:, 1:]
    position_indices = np.arange(labels.size, dtype=np.int64)[None, :]
    group_start_indices = np.where(group_start, position_indices, 0)
    np.maximum.accumulate(group_start_indices, axis=1, out=group_start_indices)
    previous_indices = np.maximum(group_start_indices - 1, 0)
    previous_positive = np.take_along_axis(
        cumulative_positive, previous_indices, axis=1
    )
    previous_positive[group_start_indices == 0] = 0.0
    positive_by_score = cumulative_positive - previous_positive
    group_size = positions - group_start_indices
    negative_by_score = group_size - positive_by_score

    precision = cumulative_positive / positions
    auprc = np.sum(
        np.where(group_end, positive_by_score * precision, 0.0), axis=1
    ) / total_positive
    negative_below = total_negative - cumulative_negative
    auroc = np.sum(
        np.where(
            group_end,
            positive_by_score * (negative_below + 0.5 * negative_by_score),
            0.0,
        ),
        axis=1,
    ) / (total_positive * total_negative)
    return {"auroc": auroc, "auprc": auprc}


def paired_query_swap_replicates(
    labels: np.ndarray,
    query_indices: np.ndarray,
    additive_scores: np.ndarray,
    orthogonalized_scores: np.ndarray,
    swap_masks: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    additive_scores = np.asarray(additive_scores, dtype=np.float64)
    orthogonalized_scores = np.asarray(orthogonalized_scores, dtype=np.float64)
    masks = np.asarray(swap_masks, dtype=np.bool_)
    if batch_size <= 0:
        raise ValueError("randomization batch size must be positive")
    if not (
        labels.shape
        == query_indices.shape
        == additive_scores.shape
        == orthogonalized_scores.shape
    ):
        raise ValueError("paired randomization row arrays differ")
    if masks.ndim != 2 or np.max(query_indices) >= masks.shape[1]:
        raise ValueError("randomization masks do not cover every query")
    differences = {
        "auroc": np.empty(masks.shape[0], dtype=np.float64),
        "auprc": np.empty(masks.shape[0], dtype=np.float64),
    }
    for start in range(0, masks.shape[0], batch_size):
        stop = min(start + batch_size, masks.shape[0])
        row_swaps = masks[start:stop, query_indices]
        pseudo_orthogonalized = np.where(
            row_swaps, additive_scores[None, :], orthogonalized_scores[None, :]
        )
        pseudo_additive = np.where(
            row_swaps, orthogonalized_scores[None, :], additive_scores[None, :]
        )
        combined = np.concatenate((pseudo_orthogonalized, pseudo_additive), axis=0)
        metrics = _batched_unweighted_metrics(labels, combined)
        width = stop - start
        for metric in differences:
            differences[metric][start:stop] = (
                metrics[metric][:width] - metrics[metric][width:]
            )
    return differences


def _per_query_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    row_query_indices: np.ndarray,
    query_count: int,
) -> dict[str, np.ndarray]:
    values = {
        "auroc": np.empty(query_count, dtype=np.float64),
        "auprc": np.empty(query_count, dtype=np.float64),
    }
    for query_index in range(query_count):
        selected = row_query_indices == query_index
        metrics = binary_classification_metrics(labels[selected], scores[selected])
        values["auroc"][query_index] = metrics["auroc"]
        values["auprc"][query_index] = metrics["auprc"]
    return values


def _cell_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    labels = payload["labels"]
    query_indices = payload["query_indices"]
    counts = payload["bootstrap_counts"]
    masks = payload["randomization_masks"]
    additive = payload["additive_scores"]
    orthogonalized = payload["orthogonalized_scores"]
    additive_bootstrap = _weighted_pooled_replicates(
        counts,
        query_indices,
        labels.astype(np.float64),
        additive,
        batch_size=payload["bootstrap_batch_size"],
    )
    orthogonalized_bootstrap = _weighted_pooled_replicates(
        counts,
        query_indices,
        labels.astype(np.float64),
        orthogonalized,
        batch_size=payload["bootstrap_batch_size"],
    )
    randomization_pooled = paired_query_swap_replicates(
        labels,
        query_indices,
        additive,
        orthogonalized,
        masks,
        batch_size=payload["randomization_batch_size"],
    )
    per_query_difference = payload["per_query_difference"]
    query_count = counts.shape[1]
    signs = 1.0 - 2.0 * masks.astype(np.float64)
    return {
        "train_resample": payload["train_resample"],
        "test_resample": payload["test_resample"],
        "bootstrap": {
            "pooled_auroc": orthogonalized_bootstrap["auroc"]
            - additive_bootstrap["auroc"],
            "pooled_auprc": orthogonalized_bootstrap["auprc"]
            - additive_bootstrap["auprc"],
            "macro_auroc": counts @ per_query_difference["auroc"] / query_count,
            "macro_auprc": counts @ per_query_difference["auprc"] / query_count,
        },
        "randomization": {
            "pooled_auroc": randomization_pooled["auroc"],
            "pooled_auprc": randomization_pooled["auprc"],
            "macro_auroc": signs @ per_query_difference["auroc"] / query_count,
            "macro_auprc": signs @ per_query_difference["auprc"] / query_count,
        },
    }


def _metric_dict(labels: np.ndarray, scores: np.ndarray, per_query: Mapping[str, np.ndarray]) -> dict[str, float]:
    pooled = binary_classification_metrics(labels, scores)
    return {
        "pooled_auroc": pooled["auroc"],
        "pooled_auprc": pooled["auprc"],
        "macro_auroc": float(np.mean(per_query["auroc"])),
        "macro_auprc": float(np.mean(per_query["auprc"])),
    }


def _load_and_audit_inputs(
    project_dir: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[str], list[str], dict[str, Any]]:
    manifest_path, manifest = _locked_json(
        project_dir, config["inputs"]["prediction_manifest"], "prediction manifest"
    )
    summary_path, summary = _locked_json(
        project_dir, config["inputs"]["rescoring_summary"], "rescoring summary"
    )
    original_path, original = _locked_json(
        project_dir,
        config["inputs"]["original_cross_resample_summary"],
        "original cross-resample summary",
    )
    per_query_path = _resolve(project_dir, config["inputs"]["per_query_metrics"]["path"])
    if file_sha256(per_query_path) != config["inputs"]["per_query_metrics"]["sha256"]:
        raise ValueError("per-query metrics SHA-256 mismatch")
    per_query_columns, per_query_rows = _read_csv(per_query_path)
    if len(per_query_rows) != 8_400:
        raise ValueError("per-query metric row count must be 8,400")
    if manifest["prediction_file_count"] != 250 or manifest["prediction_row_count"] != 1_028_900:
        raise ValueError("prediction manifest count mismatch")
    if summary["prediction_file_count"] != 250 or summary["prediction_row_count"] != 1_028_900:
        raise ValueError("rescoring summary count mismatch")
    if summary["metric_reproduction"]["maximum_absolute_difference"] != 0.0:
        raise ValueError("prior metric reproduction was not exact")

    expected_keys = {
        (model, train, test, fold)
        for model in METHODS
        for train in RESAMPLE_IDS
        for test in RESAMPLE_IDS
        for fold in FOLD_IDS
    }
    entries = {
        (row["model"], row["train_resample"], row["test_resample"], row["fold"]): row
        for row in manifest["files"]
    }
    if set(entries) != expected_keys or len(manifest["files"]) != 250:
        raise ValueError("prediction manifest is incomplete or duplicated")
    prediction_root = _resolve(project_dir, config["inputs"]["prediction_manifest"]["path"]).parent / "predictions"
    disk_files = {path.resolve() for path in prediction_root.rglob("predictions.csv.gz")}
    manifest_files = {_resolve(project_dir, row["path"]).resolve() for row in manifest["files"]}
    if disk_files != manifest_files:
        raise ValueError("prediction directory and manifest file sets differ")

    per_query_index = {
        (row["model"], row["train_resample"], row["test_resample"], row["query_id"]): row
        for row in per_query_rows
    }
    if len(per_query_index) != len(per_query_rows):
        raise ValueError("duplicate per-query metric row")
    summary_cells = {
        (row["train_resample"], row["test_resample"]): row for row in summary["cells"]
    }
    original_cells = {
        (row["train_resample"], row["test_resample"]): row for row in original["cells"]
    }
    expected_cells = {(train, test) for train in RESAMPLE_IDS for test in RESAMPLE_IDS}
    if set(summary_cells) != expected_cells or set(original_cells) != expected_cells:
        raise ValueError("cross-resample summaries are incomplete")

    query_fold: dict[str, str] = {}
    cells = []
    total_rows = 0
    paired_fold_checks = 0
    maximum_difference = 0.0
    prediction_hashes = []
    for train_resample in RESAMPLE_IDS:
        for test_resample in RESAMPLE_IDS:
            model_rows: dict[str, list[dict[str, str]]] = {model: [] for model in METHODS}
            for fold_id in FOLD_IDS:
                fold_rows = {}
                for model in METHODS:
                    entry = entries[(model, train_resample, test_resample, fold_id)]
                    path = _resolve(project_dir, entry["path"])
                    observed_sha = file_sha256(path)
                    if observed_sha != entry["sha256"] or observed_sha != entry["repeat_scoring_sha256"]:
                        raise ValueError("prediction SHA-256 or repeat-scoring SHA-256 mismatch")
                    columns, rows = _read_gzip_csv(path)
                    if columns != tuple(manifest["prediction_columns"]):
                        raise ValueError("prediction columns differ from manifest")
                    if len(rows) != int(entry["rows"]):
                        raise ValueError("prediction row count differs from manifest")
                    required = {
                        "model": model,
                        "train_resample": train_resample,
                        "test_resample": test_resample,
                        "fold": fold_id,
                        "checkpoint_sha256": entry["checkpoint_sha256"],
                        "test_data_sha256": entry["test_data_sha256"],
                    }
                    for field, value in required.items():
                        if {row[field] for row in rows} != {value}:
                            raise ValueError(f"prediction metadata mismatch: {field}")
                    fold_rows[model] = rows
                    model_rows[model].extend(rows)
                    prediction_hashes.append(observed_sha)
                    total_rows += len(rows)
                additive_identity = [
                    (row["query_id"], row["interaction_key"], row["label"], row["fold"])
                    for row in fold_rows["additive_only"]
                ]
                orthogonalized_identity = [
                    (row["query_id"], row["interaction_key"], row["label"], row["fold"])
                    for row in fold_rows["orthogonalized_interaction"]
                ]
                if additive_identity != orthogonalized_identity:
                    raise ValueError("model prediction rows are not exactly paired")
                paired_fold_checks += 1

            additive_rows = model_rows["additive_only"]
            orthogonalized_rows = model_rows["orthogonalized_interaction"]
            labels = np.asarray([int(row["label"]) for row in additive_rows], dtype=np.int8)
            additive_scores = np.asarray([float(row["probability"]) for row in additive_rows])
            orthogonalized_scores = np.asarray(
                [float(row["probability"]) for row in orthogonalized_rows]
            )
            for row in additive_rows:
                prior = query_fold.setdefault(row["query_id"], row["fold"])
                if prior != row["fold"]:
                    raise ValueError("query appears in multiple folds")

            cells.append(
                {
                    "train_resample": train_resample,
                    "test_resample": test_resample,
                    "labels": labels,
                    "query_ids_by_row": [row["query_id"] for row in additive_rows],
                    "additive_scores": additive_scores,
                    "orthogonalized_scores": orthogonalized_scores,
                }
            )

    # Match query_level_classification_metrics exactly, including floating-point
    # summation order, so persisted and recomputed macro metrics are bit-identical.
    query_ids = sorted(query_fold)
    if len(query_ids) != int(config["analysis"]["query_count"]):
        raise ValueError("common query count mismatch")
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    query_folds = [query_fold[query_id] for query_id in query_ids]
    for cell in cells:
        cell["query_indices"] = np.asarray(
            [query_index[value] for value in cell.pop("query_ids_by_row")], dtype=np.int16
        )
        per_query = {}
        method_metrics = {}
        for model, score_key in (
            ("additive_only", "additive_scores"),
            ("orthogonalized_interaction", "orthogonalized_scores"),
        ):
            values = _per_query_metrics(
                cell["labels"], cell[score_key], cell["query_indices"], len(query_ids)
            )
            per_query[model] = values
            method_metrics[model] = _metric_dict(cell["labels"], cell[score_key], values)
            for index, query_id in enumerate(query_ids):
                source = per_query_index[(model, cell["train_resample"], cell["test_resample"], query_id)]
                if source["fold"] != query_folds[index]:
                    raise ValueError("per-query fold mismatch")
                for metric in ("auroc", "auprc"):
                    if not math.isclose(values[metric][index], float(source[metric]), rel_tol=0.0, abs_tol=1e-12):
                        raise ValueError("persisted per-query metric mismatch")
            for metric in METRIC_NAMES:
                expected = float(summary_cells[(cell["train_resample"], cell["test_resample"])][model][metric])
                original_value = float(original_cells[(cell["train_resample"], cell["test_resample"])][model][metric])
                differences = (
                    abs(method_metrics[model][metric] - expected),
                    abs(expected - original_value),
                )
                maximum_difference = max(maximum_difference, *differences)
                if any(value != 0.0 for value in differences):
                    raise ValueError("25-combination metric reproduction is not exact")
        cell["per_query_difference"] = {
            metric: per_query["orthogonalized_interaction"][metric]
            - per_query["additive_only"][metric]
            for metric in ("auroc", "auprc")
        }
        cell["point_difference"] = {
            metric: method_metrics["orthogonalized_interaction"][metric]
            - method_metrics["additive_only"][metric]
            for metric in METRIC_NAMES
        }

    if total_rows != 1_028_900:
        raise ValueError("prediction total row count mismatch")
    return cells, query_ids, query_folds, {
        "prediction_manifest": {
            "path": _relative(project_dir, manifest_path),
            "sha256": file_sha256(manifest_path),
        },
        "per_query_metrics": {
            "path": _relative(project_dir, per_query_path),
            "sha256": file_sha256(per_query_path),
            "rows": len(per_query_rows),
            "columns": list(per_query_columns),
        },
        "rescoring_summary": {
            "path": _relative(project_dir, summary_path),
            "sha256": file_sha256(summary_path),
        },
        "original_cross_resample_summary": {
            "path": _relative(project_dir, original_path),
            "sha256": file_sha256(original_path),
        },
        "prediction_file_count": len(prediction_hashes),
        "prediction_row_count": total_rows,
        "prediction_file_sha256_set_sha256": hashlib.sha256(
            "\n".join(sorted(prediction_hashes)).encode("ascii")
        ).hexdigest(),
        "paired_fold_check_count": paired_fold_checks,
        "all_model_prediction_rows_paired": True,
        "common_query_count": len(query_ids),
        "all_queries_common_across_25_combinations": True,
        "maximum_metric_reproduction_absolute_difference": maximum_difference,
        "all_25_combination_metrics_reproduced_exactly": maximum_difference == 0.0,
    }


def _leave_one_out(cell_differences: np.ndarray) -> dict[str, Any]:
    result = {"train_resample": {}, "test_resample": {}}
    for axis_name, axis in (("train_resample", 0), ("test_resample", 1)):
        for excluded_index, excluded in enumerate(RESAMPLE_IDS):
            selected = np.delete(cell_differences, excluded_index, axis=axis)
            result[axis_name][excluded] = {
                metric: float(selected[:, :, metric_index].mean())
                for metric_index, metric in enumerate(METRIC_NAMES)
            }
    result["all_positive"] = {
        metric: all(
            result[scope][excluded][metric] > 0.0
            for scope in ("train_resample", "test_resample")
            for excluded in RESAMPLE_IDS
        )
        for metric in METRIC_NAMES
    }
    return result


def _render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Final cross-negative-resample paired statistics",
        "",
        "The primary estimand is the equal-weight mean of the 25 train-resample by test-resample metric differences. Predictions were not concatenated into a single pseudo-dataset.",
        "",
        "## Primary results",
        "",
        "| Metric | Orthogonalized - Additive-only | 95% CI | Raw two-sided p | Holm p | Positive cells |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in METRIC_NAMES:
        item = result["metrics"][metric]
        lines.append(
            f"| {metric} | {item['difference']:+.8f} | "
            f"[{item['ci_lower']:+.8f}, {item['ci_upper']:+.8f}] | "
            f"{item['p_value_two_sided']:.8g} | {item['p_value_holm']:.8g} | "
            f"{item['positive_combination_count']}/25 |"
        )
    lines.extend(
        [
            "",
            "## Leave-one-train-resample-out",
            "",
            "| Excluded | Pooled AUROC | Pooled AUPRC | Macro AUROC | Macro AUPRC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for excluded, values in result["leave_one_out"]["train_resample"].items():
        lines.append(
            f"| {excluded} | " + " | ".join(f"{values[m]:+.8f}" for m in METRIC_NAMES) + " |"
        )
    lines.extend(
        [
            "",
            "## Leave-one-test-resample-out",
            "",
            "| Excluded | Pooled AUROC | Pooled AUPRC | Macro AUROC | Macro AUPRC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for excluded, values in result["leave_one_out"]["test_resample"].items():
        lines.append(
            f"| {excluded} | " + " | ".join(f"{values[m]:+.8f}" for m in METRIC_NAMES) + " |"
        )
    lines.extend(["", "## Audit", ""])
    audit = result["audit"]
    lines.extend(
        [
            f"- Prediction files: {audit['prediction_file_count']}",
            f"- Prediction rows: {audit['prediction_row_count']:,}",
            f"- Common queries: {audit['common_query_count']}",
            f"- Maximum reproduction difference: {audit['maximum_metric_reproduction_absolute_difference']}",
            f"- Bootstrap: {result['bootstrap']['iterations']:,} fold-stratified query-cluster resamples",
            f"- Randomization: {result['randomization']['iterations']:,} paired query score swaps across all 25 combinations",
            "- No training, rescoring, tuning, or data modification was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_cross_resample_final_statistics(
    project_dir: Path, config_path: Path
) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    bootstrap_config = config["bootstrap"]
    randomization_config = config["randomization"]
    analysis_config = config["analysis"]
    if int(bootstrap_config["iterations"]) != 10_000 or int(randomization_config["iterations"]) != 10_000:
        raise ValueError("final analysis requires exactly 10,000 bootstrap and randomization iterations")
    if bootstrap_config["resampling_unit"] != "query_cluster" or bootstrap_config["stratified_by"] != "fold":
        raise ValueError("bootstrap must be fold-stratified query-cluster sampling")
    if randomization_config["operation"] != "swap_model_scores_for_query_across_all_25_combinations":
        raise ValueError("randomization operation changed")
    if analysis_config["estimand"] != "equal_weight_mean_of_25_train_by_test_resample_differences":
        raise ValueError("primary estimand changed")

    output_directory = _resolve(project_dir, config["outputs"]["directory"])
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_directory}")

    cells, query_ids, query_folds, input_audit = _load_and_audit_inputs(project_dir, config)
    counts, stratum_sizes = stratified_query_bootstrap_counts(
        query_folds,
        seed=int(bootstrap_config["seed"]),
        iterations=int(bootstrap_config["iterations"]),
    )
    masks = generate_randomization_masks(
        len(query_ids),
        seed=int(randomization_config["seed"]),
        iterations=int(randomization_config["iterations"]),
    )
    payloads = []
    for cell in cells:
        payloads.append(
            {
                **cell,
                "bootstrap_counts": counts,
                "randomization_masks": masks,
                "bootstrap_batch_size": int(bootstrap_config["batch_size"]),
                "randomization_batch_size": int(randomization_config["batch_size"]),
            }
        )
    worker_count = min(int(analysis_config["worker_count"]), len(payloads))
    if worker_count == 1:
        cell_replicates = [_cell_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            cell_replicates = list(executor.map(_cell_worker, payloads))

    combination_count = len(cells)
    bootstrap_replicates = {
        metric: sum(cell["bootstrap"][metric] for cell in cell_replicates) / combination_count
        for metric in METRIC_NAMES
    }
    randomization_replicates = {
        metric: sum(cell["randomization"][metric] for cell in cell_replicates) / combination_count
        for metric in METRIC_NAMES
    }
    cell_differences = np.empty((5, 5, len(METRIC_NAMES)), dtype=np.float64)
    for cell in cells:
        train_index = RESAMPLE_IDS.index(cell["train_resample"])
        test_index = RESAMPLE_IDS.index(cell["test_resample"])
        for metric_index, metric in enumerate(METRIC_NAMES):
            cell_differences[train_index, test_index, metric_index] = cell["point_difference"][metric]
    observed = {
        metric: float(cell_differences[:, :, index].mean())
        for index, metric in enumerate(METRIC_NAMES)
    }
    raw_p_values = {
        metric: float(
            (1 + np.sum(np.abs(randomization_replicates[metric]) >= abs(observed[metric]) - 1e-15))
            / (int(randomization_config["iterations"]) + 1)
        )
        for metric in METRIC_NAMES
    }
    adjusted_p_values = holm_adjust(raw_p_values)
    metrics = {}
    for metric_index, metric in enumerate(METRIC_NAMES):
        lower, upper = _quantile_interval(
            bootstrap_replicates[metric], float(bootstrap_config["confidence_level"])
        )
        metrics[metric] = {
            "difference": observed[metric],
            "ci_lower": lower,
            "ci_upper": upper,
            "p_value_two_sided": raw_p_values[metric],
            "p_value_holm": adjusted_p_values[metric],
            "positive_combination_count": int(np.sum(cell_differences[:, :, metric_index] > 0.0)),
            "combination_count": combination_count,
            "bootstrap_replicates_sha256": _array_sha256(bootstrap_replicates[metric]),
            "randomization_replicates_sha256": _array_sha256(randomization_replicates[metric]),
            "randomization_null_mean": float(np.mean(randomization_replicates[metric])),
        }
    leave_one_out = _leave_one_out(cell_differences)
    result: dict[str, Any] = {
        "schema_version": 1,
        "comparison": COMPARISON,
        "estimand": analysis_config["estimand"],
        "query_count": len(query_ids),
        "combination_count": combination_count,
        "metrics": metrics,
        "leave_one_out": leave_one_out,
        "bootstrap": {
            "seed": int(bootstrap_config["seed"]),
            "iterations": int(bootstrap_config["iterations"]),
            "confidence_level": float(bootstrap_config["confidence_level"]),
            "resampling_unit": "query_cluster",
            "stratified_by": "fold",
            "stratum_query_counts": stratum_sizes,
            "same_query_counts_used_for_both_models_and_all_25_combinations": True,
            "sampling_counts_sha256": _array_sha256(counts),
        },
        "randomization": {
            "seed": int(randomization_config["seed"]),
            "iterations": int(randomization_config["iterations"]),
            "alternative": "two_sided",
            "unit": "query_cluster",
            "same_query_swap_used_for_both_models_and_all_25_combinations": True,
            "swap_masks_sha256": _array_sha256(masks),
            "p_value_formula": "(1 + count(abs(null) >= abs(observed) - 1e-15)) / (iterations + 1)",
        },
        "multiple_testing": {
            "method": "holm",
            "family": "four_primary_metrics",
        },
        "audit": {
            "status": "passed",
            **input_audit,
            "config": {
                "path": _relative(project_dir, config_path),
                "sha256": file_sha256(config_path),
            },
            "point_estimator_computed_per_combination_then_equal_weighted": True,
            "predictions_concatenated_across_combinations_for_primary_estimate": False,
            "models_trained": False,
            "predictions_rescored": False,
            "data_resampled": False,
        },
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / config["outputs"]["statistics_csv"]
    json_path = output_directory / config["outputs"]["statistics_json"]
    report_path = output_directory / config["outputs"]["report_md"]
    csv_rows = []
    for metric in METRIC_NAMES:
        item = metrics[metric]
        csv_rows.append(
            {
                "record_type": "overall",
                "excluded_resample": "",
                "metric": metric,
                "difference": item["difference"],
                "ci_lower": item["ci_lower"],
                "ci_upper": item["ci_upper"],
                "p_value_two_sided": item["p_value_two_sided"],
                "p_value_holm": item["p_value_holm"],
                "positive_combination_count": item["positive_combination_count"],
                "combination_count": combination_count,
                "all_leave_one_out_differences_positive": leave_one_out["all_positive"][metric],
                "bootstrap_iterations": bootstrap_config["iterations"],
                "bootstrap_seed": bootstrap_config["seed"],
                "randomization_iterations": randomization_config["iterations"],
                "randomization_seed": randomization_config["seed"],
            }
        )
    for scope in ("train_resample", "test_resample"):
        for excluded in RESAMPLE_IDS:
            for metric in METRIC_NAMES:
                csv_rows.append(
                    {
                        "record_type": f"leave_one_{scope}_out",
                        "excluded_resample": excluded,
                        "metric": metric,
                        "difference": leave_one_out[scope][excluded][metric],
                        "ci_lower": "",
                        "ci_upper": "",
                        "p_value_two_sided": "",
                        "p_value_holm": "",
                        "positive_combination_count": "",
                        "combination_count": 20,
                        "all_leave_one_out_differences_positive": leave_one_out["all_positive"][metric],
                        "bootstrap_iterations": "",
                        "bootstrap_seed": "",
                        "randomization_iterations": "",
                        "randomization_seed": "",
                    }
                )
    _write_csv(csv_path, csv_rows)
    report_path.write_text(_render_report(result), encoding="utf-8")
    result["outputs"] = {
        "statistics_csv": {
            "path": _relative(project_dir, csv_path),
            "sha256": file_sha256(csv_path),
            "rows": len(csv_rows),
        },
        "report": {
            "path": _relative(project_dir, report_path),
            "sha256": file_sha256(report_path),
        },
    }
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
