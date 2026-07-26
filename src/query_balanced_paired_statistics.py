from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .five_fold_summary import file_sha256
from .metrics import binary_classification_metrics, query_level_classification_metrics


METRIC_NAMES = ("pooled_auroc", "pooled_auprc", "macro_auroc", "macro_auprc")
CSV_COLUMNS = (
    "record_type",
    "metric",
    "similarity_estimate",
    "cnn_estimate",
    "difference",
    "ci_lower",
    "ci_upper",
    "wins",
    "ties",
    "losses",
    "query_count",
    "bootstrap_iterations",
    "bootstrap_seed",
    "confidence_level",
    "resampling_unit",
    "stratified_by",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _resolve(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def _relative(project_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(project_dir.resolve()).as_posix()


def _assert_close(observed: float, expected: float, description: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{description} mismatch: {observed} != {expected}")


def _verified_summary_output(
    project_dir: Path,
    summary: Mapping[str, Any],
    output_key: str,
    description: str,
) -> tuple[Path, list[dict[str, str]], dict[str, Any]]:
    source = summary["outputs"][output_key]
    path = _resolve(project_dir, source["path"])
    observed_sha = file_sha256(path)
    if observed_sha != source["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {description}")
    _, rows = _read_csv(path)
    return path, rows, {
        "path": _relative(project_dir, path),
        "sha256": observed_sha,
        "rows": len(rows),
    }


def stratified_query_bootstrap_counts(
    query_folds: Sequence[str], *, seed: int, iterations: int
) -> tuple[np.ndarray, dict[str, int]]:
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if not query_folds:
        raise ValueError("query folds must be non-empty")
    folds = np.asarray(query_folds, dtype=object)
    counts = np.zeros((iterations, len(query_folds)), dtype=np.int16)
    rng = np.random.default_rng(seed)
    stratum_sizes = {}
    for fold_id in sorted(set(query_folds)):
        positions = np.flatnonzero(folds == fold_id)
        stratum_sizes[fold_id] = int(len(positions))
        local_draws = rng.integers(0, len(positions), size=(iterations, len(positions)))
        selected = positions[local_draws]
        np.add.at(
            counts,
            (np.repeat(np.arange(iterations), len(positions)), selected.reshape(-1)),
            1,
        )
    return counts, stratum_sizes


def _weighted_pooled_replicates(
    counts: np.ndarray,
    row_query_indices: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("bootstrap batch size must be positive")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_query_indices = row_query_indices[order]
    group_starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    iterations = counts.shape[0]
    replicates = {
        "auroc": np.empty(iterations, dtype=np.float64),
        "auprc": np.empty(iterations, dtype=np.float64),
    }
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        row_weights = counts[start:stop, sorted_query_indices].astype(np.float64)
        positive_by_score = np.add.reduceat(
            row_weights * sorted_labels, group_starts, axis=1
        )
        negative_by_score = np.add.reduceat(
            row_weights * (1.0 - sorted_labels), group_starts, axis=1
        )
        cumulative_positive = np.cumsum(positive_by_score, axis=1)
        cumulative_negative = np.cumsum(negative_by_score, axis=1)
        total_positive = cumulative_positive[:, -1]
        total_negative = cumulative_negative[:, -1]
        precision = np.divide(
            cumulative_positive,
            cumulative_positive + cumulative_negative,
            out=np.zeros_like(cumulative_positive),
            where=(cumulative_positive + cumulative_negative) > 0,
        )
        replicates["auprc"][start:stop] = (
            np.sum(positive_by_score * precision, axis=1) / total_positive
        )
        negative_below = total_negative[:, None] - cumulative_negative
        replicates["auroc"][start:stop] = np.sum(
            positive_by_score * (negative_below + 0.5 * negative_by_score), axis=1
        ) / (total_positive * total_negative)
    return replicates


def _quantile_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        values, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear"
    )
    return float(lower), float(upper)


def paired_stratified_query_bootstrap(
    predictions: Sequence[Mapping[str, Any]],
    per_query: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    iterations: int,
    confidence_level: float,
    batch_size: int,
    tie_tolerance: float,
) -> dict[str, Any]:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between zero and one")
    if tie_tolerance < 0.0:
        raise ValueError("tie tolerance must be non-negative")
    query_rows = sorted(per_query, key=lambda row: (str(row["fold_id"]), str(row["query_id"])))
    query_ids = [str(row["query_id"]) for row in query_rows]
    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError("paired bootstrap queries must be non-empty and unique")
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    query_folds = [str(row["fold_id"]) for row in query_rows]
    counts, stratum_sizes = stratified_query_bootstrap_counts(
        query_folds, seed=seed, iterations=iterations
    )
    for fold_id, size in stratum_sizes.items():
        positions = [index for index, value in enumerate(query_folds) if value == fold_id]
        if not np.all(counts[:, positions].sum(axis=1) == size):
            raise ValueError(f"bootstrap stratum size changed for {fold_id}")

    prediction_query_ids = [str(row["query_id"]) for row in predictions]
    if set(prediction_query_ids) != set(query_ids):
        raise ValueError("prediction and per-query query sets differ")
    row_query_indices = np.asarray(
        [query_index[query_id] for query_id in prediction_query_ids], dtype=np.int64
    )
    labels = np.asarray([int(row["label"]) for row in predictions], dtype=np.int64)
    similarity_scores = np.asarray(
        [float(row["similarity_score"]) for row in predictions], dtype=np.float64
    )
    cnn_scores = np.asarray(
        [float(row["cnn_probability"]) for row in predictions], dtype=np.float64
    )
    if not np.isfinite(similarity_scores).all() or not np.isfinite(cnn_scores).all():
        raise ValueError("paired prediction scores must be finite")

    similarity_pooled = binary_classification_metrics(labels, similarity_scores)
    cnn_pooled = binary_classification_metrics(labels, cnn_scores)
    similarity_replicates = _weighted_pooled_replicates(
        counts,
        row_query_indices,
        labels.astype(np.float64),
        similarity_scores,
        batch_size=batch_size,
    )
    cnn_replicates = _weighted_pooled_replicates(
        counts,
        row_query_indices,
        labels.astype(np.float64),
        cnn_scores,
        batch_size=batch_size,
    )

    per_query_values = {
        "similarity": {
            metric: np.asarray(
                [float(row[f"similarity_{metric}"]) for row in query_rows],
                dtype=np.float64,
            )
            for metric in ("auroc", "auprc")
        },
        "cnn": {
            metric: np.asarray(
                [float(row[f"cnn_{metric}"]) for row in query_rows], dtype=np.float64
            )
            for metric in ("auroc", "auprc")
        },
    }
    query_count = len(query_rows)
    macro_replicates = {
        method: {
            metric: counts @ values / query_count
            for metric, values in method_values.items()
        }
        for method, method_values in per_query_values.items()
    }

    metrics = {}
    for metric_name, scope, metric in (
        ("pooled_auroc", "pooled", "auroc"),
        ("pooled_auprc", "pooled", "auprc"),
        ("macro_auroc", "macro", "auroc"),
        ("macro_auprc", "macro", "auprc"),
    ):
        if scope == "pooled":
            similarity_estimate = similarity_pooled[metric]
            cnn_estimate = cnn_pooled[metric]
            differences = similarity_replicates[metric] - cnn_replicates[metric]
        else:
            similarity_estimate = float(per_query_values["similarity"][metric].mean())
            cnn_estimate = float(per_query_values["cnn"][metric].mean())
            differences = (
                macro_replicates["similarity"][metric]
                - macro_replicates["cnn"][metric]
            )
        lower, upper = _quantile_interval(differences, confidence_level)
        metrics[metric_name] = {
            "similarity": float(similarity_estimate),
            "cnn": float(cnn_estimate),
            "difference": float(similarity_estimate - cnn_estimate),
            "ci_lower": lower,
            "ci_upper": upper,
        }

    query_outcomes = {}
    for metric in ("auroc", "auprc"):
        differences = per_query_values["similarity"][metric] - per_query_values["cnn"][metric]
        query_outcomes[metric] = {
            "wins": int(np.sum(differences > tie_tolerance)),
            "ties": int(np.sum(np.abs(differences) <= tie_tolerance)),
            "losses": int(np.sum(differences < -tie_tolerance)),
        }
    return {
        "metrics": metrics,
        "query_outcomes": query_outcomes,
        "query_count": query_count,
        "stratum_query_counts": stratum_sizes,
        "seed": seed,
        "iterations": iterations,
        "confidence_level": confidence_level,
        "tie_tolerance": tie_tolerance,
        "resampling_unit": "query_cluster",
        "stratified_by": "fold_id",
        "rows_within_query_resampled_together": True,
    }


def _interaction_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        [
            [
                row["fold_id"],
                row["sample_id"],
                row["tcr_cdr3_beta"],
                row["epitope"],
                row["hla"],
            ]
            for row in rows
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _audit_and_join_inputs(
    *,
    project_dir: Path,
    manifest: Mapping[str, Any],
    cnn_rows: Sequence[Mapping[str, str]],
    cnn_per_query_rows: Sequence[Mapping[str, str]],
    similarity_rows: Sequence[Mapping[str, str]],
    similarity_per_query_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(cnn_rows) != len(similarity_rows):
        raise ValueError("CNN and similarity OOF row counts differ")
    identity_fields = ("fold_id", "sample_id", "source", "pmhc_id", "label")
    for field in identity_fields:
        if [row[field] for row in cnn_rows] != [row[field] for row in similarity_rows]:
            raise ValueError(f"CNN and similarity OOF {field} differs")
    for cnn_row, similarity_row in zip(cnn_rows, similarity_rows, strict=True):
        _assert_close(
            float(similarity_row["cnn_probability"]),
            float(cnn_row["probability"]),
            f"CNN score {cnn_row['sample_id']}",
        )

    cnn_query_index = {row["query_id"]: row for row in cnn_per_query_rows}
    similarity_query_index = {row["query_id"]: row for row in similarity_per_query_rows}
    if len(cnn_query_index) != len(cnn_per_query_rows):
        raise ValueError("duplicate CNN per-query row")
    if len(similarity_query_index) != len(similarity_per_query_rows):
        raise ValueError("duplicate similarity per-query row")
    if set(cnn_query_index) != set(similarity_query_index):
        raise ValueError("CNN and similarity query IDs differ")
    for query_id in sorted(cnn_query_index):
        cnn_query = cnn_query_index[query_id]
        similarity_query = similarity_query_index[query_id]
        for field in (
            "fold_id",
            "epitope",
            "hla",
            "positive_count",
            "negative_count",
            "sample_count",
        ):
            if cnn_query[field] != similarity_query[field]:
                raise ValueError(f"per-query {field} mismatch for {query_id}")

    expected_rows = []
    input_audit = {}
    for fold_id in sorted(manifest["folds"]):
        partition = manifest["folds"][fold_id]["partitions"]["test"]
        fold_audit = {}
        for kind, output_name, identifier in (
            ("positive", "positive_output", "record_id"),
            ("negative", "negative_output", "negative_id"),
        ):
            source = partition[output_name]
            path = _resolve(project_dir, source["path"])
            observed_sha = file_sha256(path)
            if observed_sha != source["sha256"]:
                raise ValueError(f"manifest SHA-256 mismatch for {fold_id} test {kind}")
            _, rows = _read_csv(path)
            if len(rows) != int(source["rows"]):
                raise ValueError(f"manifest row count mismatch for {fold_id} test {kind}")
            for row in rows:
                expected_rows.append(
                    {
                        "fold_id": fold_id,
                        "sample_id": row[identifier],
                        "label": row["label"],
                        "pmhc_id": row["pmhc_id"],
                        "tcr_cdr3_beta": row["tcr_cdr3_beta"],
                        "epitope": row["epitope"],
                        "hla": row["hla"],
                    }
                )
            fold_audit[kind] = {
                "path": _relative(project_dir, path),
                "sha256": observed_sha,
                "rows": len(rows),
            }
        input_audit[fold_id] = fold_audit
    if len(expected_rows) != len(similarity_rows):
        raise ValueError("manifest and OOF row counts differ")
    for expected, similarity_row in zip(expected_rows, similarity_rows, strict=True):
        for field in (
            "fold_id",
            "sample_id",
            "label",
            "pmhc_id",
            "tcr_cdr3_beta",
            "epitope",
            "hla",
        ):
            if str(expected[field]) != str(similarity_row[field]):
                raise ValueError(f"manifest interaction field {field} mismatch")

    available_query_ids = {
        row["query_id"]
        for row in similarity_per_query_rows
        if row["status"] == "available"
    }
    if len(available_query_ids) != 153:
        raise ValueError("expected exactly 153 commonly evaluable queries")
    common_predictions = []
    for cnn_row, similarity_row in zip(cnn_rows, similarity_rows, strict=True):
        if similarity_row["pmhc_id"] not in available_query_ids:
            continue
        if similarity_row["status"] != "available" or not similarity_row["similarity_score"]:
            raise ValueError("available query has missing similarity score")
        common_predictions.append(
            {
                "fold_id": similarity_row["fold_id"],
                "query_id": similarity_row["pmhc_id"],
                "sample_id": similarity_row["sample_id"],
                "label": int(similarity_row["label"]),
                "similarity_score": float(similarity_row["similarity_score"]),
                "cnn_probability": float(cnn_row["probability"]),
                "tcr_cdr3_beta": similarity_row["tcr_cdr3_beta"],
                "epitope": similarity_row["epitope"],
                "hla": similarity_row["hla"],
            }
        )
    common_per_query = []
    for query_id in sorted(available_query_ids):
        cnn_row = cnn_query_index[query_id]
        similarity_row = similarity_query_index[query_id]
        for metric in ("auroc", "auprc"):
            _assert_close(
                float(cnn_row[metric]),
                float(similarity_row[f"cnn_{metric}"]),
                f"CNN per-query {metric} {query_id}",
            )
        common_per_query.append(
            {
                "fold_id": similarity_row["fold_id"],
                "query_id": query_id,
                "similarity_auroc": float(similarity_row["similarity_auroc"]),
                "similarity_auprc": float(similarity_row["similarity_auprc"]),
                "cnn_auroc": float(cnn_row["auroc"]),
                "cnn_auprc": float(cnn_row["auprc"]),
            }
        )

    similarity_macro, similarity_recomputed = query_level_classification_metrics(
        [
            {
                "pmhc_id": row["query_id"],
                "label": row["label"],
                "probability": row["similarity_score"],
            }
            for row in common_predictions
        ]
    )
    cnn_macro, cnn_recomputed = query_level_classification_metrics(
        [
            {
                "pmhc_id": row["query_id"],
                "label": row["label"],
                "probability": row["cnn_probability"],
            }
            for row in common_predictions
        ]
    )
    similarity_recomputed_index = {row["query_id"]: row for row in similarity_recomputed}
    cnn_recomputed_index = {row["query_id"]: row for row in cnn_recomputed}
    for row in common_per_query:
        query_id = row["query_id"]
        for metric in ("auroc", "auprc"):
            _assert_close(
                row[f"similarity_{metric}"],
                float(similarity_recomputed_index[query_id][metric]),
                f"similarity recomputed {metric} {query_id}",
            )
            _assert_close(
                row[f"cnn_{metric}"],
                float(cnn_recomputed_index[query_id][metric]),
                f"CNN recomputed {metric} {query_id}",
            )
    if int(similarity_macro["query_count"]) != 153 or int(cnn_macro["query_count"]) != 153:
        raise ValueError("recomputed common query count mismatch")
    common_interactions = [
        row for row in expected_rows if row["pmhc_id"] in available_query_ids
    ]
    return common_predictions, common_per_query, {
        "manifest_test_inputs": input_audit,
        "query_ids_match": True,
        "fold_ids_match": True,
        "sample_labels_match": True,
        "interaction_keys_match": True,
        "common_sample_count": len(common_predictions),
        "common_query_count": len(available_query_ids),
        "common_interaction_key_sha256": _interaction_sha256(common_interactions),
    }


def _csv_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    bootstrap = result["bootstrap"]
    common = {
        "query_count": result["common_query_count"],
        "bootstrap_iterations": bootstrap["iterations"],
        "bootstrap_seed": bootstrap["seed"],
        "confidence_level": bootstrap["confidence_level"],
        "resampling_unit": bootstrap["resampling_unit"],
        "stratified_by": bootstrap["stratified_by"],
    }
    rows = []
    for metric in METRIC_NAMES:
        statistics = result["metrics"][metric]
        rows.append(
            {
                "record_type": "paired_difference",
                "metric": metric,
                "similarity_estimate": statistics["similarity"],
                "cnn_estimate": statistics["cnn"],
                "difference": statistics["difference"],
                "ci_lower": statistics["ci_lower"],
                "ci_upper": statistics["ci_upper"],
                **common,
            }
        )
    for metric in ("auroc", "auprc"):
        rows.append(
            {
                "record_type": "query_outcome",
                "metric": metric,
                **result["query_outcomes"][metric],
                **common,
            }
        )
    return rows


def _render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Paired OOF CNN vs sequence-similarity statistics",
        "",
        f"- Common evaluable queries: {result['common_query_count']}",
        f"- Common samples: {result['audit']['paired_inputs']['common_sample_count']}",
        f"- Bootstrap: {result['bootstrap']['iterations']:,} fixed-seed paired query-cluster resamples",
        "- Stratification: fold ID; each replicate preserves the available-query count of every fold",
        "- Pooled metrics keep every sampled query's rows together; no row-level bootstrap is used",
        "",
        "## Similarity minus CNN",
        "",
        "| Metric | Similarity | CNN | Difference | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRIC_NAMES:
        row = result["metrics"][metric]
        lines.append(
            f"| {metric} | {row['similarity']:.6f} | {row['cnn']:.6f} | "
            f"{row['difference']:+.6f} | [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Query outcomes",
            "",
            "| Metric | Wins | Ties | Losses |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in ("auroc", "auprc"):
        outcome = result["query_outcomes"][metric]
        lines.append(
            f"| {metric} | {outcome['wins']} | {outcome['ties']} | {outcome['losses']} |"
        )
    lines.extend(
        [
            "",
            "No rescoring, training, or modification of existing CNN or similarity results was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def build_query_balanced_paired_statistics(
    config_path: Path, project_dir: Path, *, force: bool = False
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    bootstrap_config = config["bootstrap"]
    if bootstrap_config["resampling_unit"] != "query_cluster":
        raise ValueError("pooled bootstrap must use query clusters")
    if bootstrap_config["stratified_by"] != "fold_id":
        raise ValueError("paired bootstrap must be stratified by fold ID")

    manifest_path = _resolve(project_dir, config["inputs"]["manifest"])
    manifest = _load_json(manifest_path)
    manifest_sha = file_sha256(manifest_path)
    cnn_summary_path = _resolve(project_dir, config["inputs"]["cnn_summary"])
    similarity_summary_path = _resolve(
        project_dir, config["inputs"]["similarity_summary"]
    )
    cnn_summary = _load_json(cnn_summary_path)
    similarity_summary = _load_json(similarity_summary_path)
    if cnn_summary["audit"]["manifest"]["sha256"] != manifest_sha:
        raise ValueError("CNN manifest SHA-256 mismatch")
    if similarity_summary["audit"]["manifest"]["sha256"] != manifest_sha:
        raise ValueError("similarity manifest SHA-256 mismatch")

    cnn_oof_path, cnn_rows, cnn_oof_audit = _verified_summary_output(
        project_dir, cnn_summary, "oof_predictions_csv", "CNN OOF predictions"
    )
    cnn_per_query_path, cnn_per_query_rows, cnn_per_query_audit = _verified_summary_output(
        project_dir, cnn_summary, "oof_per_query_csv", "CNN per-query metrics"
    )
    similarity_oof_path, similarity_rows, similarity_oof_audit = _verified_summary_output(
        project_dir,
        similarity_summary,
        "oof_predictions_csv",
        "similarity OOF predictions",
    )
    similarity_per_query_path, similarity_per_query_rows, similarity_per_query_audit = (
        _verified_summary_output(
            project_dir,
            similarity_summary,
            "per_query_csv",
            "similarity per-query metrics",
        )
    )
    common_predictions, common_per_query, paired_input_audit = _audit_and_join_inputs(
        project_dir=project_dir,
        manifest=manifest,
        cnn_rows=cnn_rows,
        cnn_per_query_rows=cnn_per_query_rows,
        similarity_rows=similarity_rows,
        similarity_per_query_rows=similarity_per_query_rows,
    )
    bootstrap = paired_stratified_query_bootstrap(
        common_predictions,
        common_per_query,
        seed=int(bootstrap_config["seed"]),
        iterations=int(bootstrap_config["iterations"]),
        confidence_level=float(bootstrap_config["confidence_level"]),
        batch_size=int(bootstrap_config["batch_size"]),
        tie_tolerance=float(bootstrap_config["tie_tolerance"]),
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "common_query_count": len(common_per_query),
        "metrics": bootstrap["metrics"],
        "query_outcomes": bootstrap["query_outcomes"],
        "bootstrap": {
            key: value for key, value in bootstrap.items() if key not in {"metrics", "query_outcomes"}
        },
        "audit": {
            "status": "passed",
            "manifest": {
                "path": _relative(project_dir, manifest_path),
                "sha256": manifest_sha,
            },
            "paired_inputs": paired_input_audit,
            "input_artifacts": {
                "cnn_summary": {
                    "path": _relative(project_dir, cnn_summary_path),
                    "sha256": file_sha256(cnn_summary_path),
                },
                "cnn_oof_predictions": cnn_oof_audit,
                "cnn_per_query_metrics": cnn_per_query_audit,
                "similarity_summary": {
                    "path": _relative(project_dir, similarity_summary_path),
                    "sha256": file_sha256(similarity_summary_path),
                },
                "similarity_oof_predictions": similarity_oof_audit,
                "similarity_per_query_metrics": similarity_per_query_audit,
            },
            "source_paths": {
                "cnn_oof": _relative(project_dir, cnn_oof_path),
                "cnn_per_query": _relative(project_dir, cnn_per_query_path),
                "similarity_oof": _relative(project_dir, similarity_oof_path),
                "similarity_per_query": _relative(project_dir, similarity_per_query_path),
            },
            "rescoring_performed": False,
            "training_performed": False,
        },
    }

    output_directory = _resolve(project_dir, config["outputs"]["directory"])
    output_paths = {
        name: output_directory / config["outputs"][name]
        for name in ("statistics_csv", "statistics_json", "report_md")
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(f"paired statistics outputs already exist: {existing}")
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_paths["statistics_csv"], _csv_rows(result))
    output_paths["report_md"].write_text(_render_report(result), encoding="utf-8")
    result["outputs"] = {
        name: {
            "path": _relative(project_dir, path),
            "sha256": file_sha256(path),
        }
        for name, path in output_paths.items()
        if name != "statistics_json"
    }
    output_paths["statistics_json"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
