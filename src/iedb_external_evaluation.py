from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .baseline_vs_null import holm_adjust
from .cross_resample_final_statistics import (
    generate_randomization_masks,
    paired_query_swap_replicates,
)
from .five_fold_summary import file_sha256
from .metrics import binary_classification_metrics
from .query_balanced_paired_statistics import (
    _quantile_interval,
    _weighted_pooled_replicates,
    stratified_query_bootstrap_counts,
)


FULL_METHODS = (
    "orthogonalized_interaction",
    "additive_only",
    "degree_matched_cnn",
    "degree_matched_tcr_only_logistic",
)
SIMILARITY_METHOD = "sequence_similarity"
ALL_METHODS = FULL_METHODS + (SIMILARITY_METHOD,)
PRIMARY_METHOD = "orthogonalized_interaction"
COMPARATOR_METHOD = "additive_only"
ENDPOINTS = (
    "query_level_macro_auprc",
    "query_level_macro_auroc",
    "pooled_auroc",
    "pooled_auprc",
)
EXPECTED_OUTPUT_FILES = {
    "external_metrics.csv",
    "external_per_query_metrics.csv",
    "primary_comparison.csv",
    "primary_comparison.json",
    "external_evaluation_report.md",
    "evaluation_manifest.json",
}

METRICS_COLUMNS = (
    "population",
    "method",
    "query_count",
    "total_query_count",
    "record_count",
    "coverage_fraction",
    "query_level_macro_auprc",
    "query_level_macro_auroc",
    "pooled_auprc",
    "pooled_auroc",
)
PER_QUERY_COLUMNS = (
    "method",
    "query_id",
    "status",
    "positive_count",
    "negative_count",
    "record_count",
    "auprc",
    "auroc",
)
COMPARISON_COLUMNS = (
    "endpoint",
    "endpoint_role",
    "orthogonalized",
    "additive_only",
    "difference",
    "ci_lower",
    "ci_upper",
    "p_value_two_sided",
    "p_value_holm",
    "bootstrap_iterations",
    "bootstrap_seed",
    "permutation_iterations",
    "permutation_seed",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def _csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    from io import StringIO

    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _resolve(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def _relative(project_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(project_dir.resolve()).as_posix()


def _format_float(value: float | None) -> str:
    return "" if value is None else format(float(value), ".12f")


def _verified_path(
    project_dir: Path, source: Mapping[str, Any], description: str
) -> Path:
    path = _resolve(project_dir, str(source["path"]))
    observed = file_sha256(path)
    if observed != source["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {description}: {path}")
    return path


def _verify_protocol_sources(
    project_dir: Path, protocol_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    verified = {}
    for name, source in sorted(protocol_manifest["sources"].items()):
        path = _verified_path(project_dir, source, f"protocol source {name}")
        verified[name] = {
            "path": _relative(project_dir, path),
            "sha256": source["sha256"],
        }
    for name, source in sorted(protocol_manifest["outputs"].items()):
        path = _verified_path(project_dir, source, f"protocol output {name}")
        verified[f"protocol_output:{name}"] = {
            "path": _relative(project_dir, path),
            "sha256": source["sha256"],
        }
    return verified


def _query_metric_rows(
    labels: np.ndarray,
    scores: np.ndarray,
    query_ids: Sequence[str],
    *,
    method: str,
    available: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    available_mask = (
        np.ones(labels.size, dtype=np.bool_)
        if available is None
        else np.asarray(available, dtype=np.bool_)
    )
    if not (labels.shape == scores.shape == available_mask.shape):
        raise ValueError("labels, scores, and availability must have identical shapes")
    query_array = np.asarray(query_ids, dtype=object)
    rows = []
    evaluable: dict[str, dict[str, float]] = {}
    for query_id in sorted(set(query_ids)):
        selected = query_array == query_id
        query_labels = labels[selected]
        query_available = available_mask[selected]
        positive_count = int(query_labels.sum())
        negative_count = int(query_labels.size - positive_count)
        if not np.all(query_available):
            status = "unavailable"
            auprc = None
            auroc = None
        elif positive_count == 0 or negative_count == 0:
            status = "one_class"
            auprc = None
            auroc = None
        else:
            status = "evaluable"
            metrics = binary_classification_metrics(query_labels, scores[selected])
            auprc = float(metrics["auprc"])
            auroc = float(metrics["auroc"])
            evaluable[query_id] = {"auprc": auprc, "auroc": auroc}
        rows.append(
            {
                "method": method,
                "query_id": query_id,
                "status": status,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "record_count": int(query_labels.size),
                "auprc": _format_float(auprc),
                "auroc": _format_float(auroc),
            }
        )
    return rows, evaluable


def _absolute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    query_ids: Sequence[str],
    *,
    method: str,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, dict[str, float]]]:
    per_query_rows, per_query = _query_metric_rows(
        labels, scores, query_ids, method=method
    )
    if not per_query:
        raise ValueError(f"No evaluable queries for {method}")
    pooled = binary_classification_metrics(labels, scores)
    return (
        {
            "query_level_macro_auprc": float(
                np.mean([row["auprc"] for row in per_query.values()])
            ),
            "query_level_macro_auroc": float(
                np.mean([row["auroc"] for row in per_query.values()])
            ),
            "pooled_auprc": float(pooled["auprc"]),
            "pooled_auroc": float(pooled["auroc"]),
        },
        per_query_rows,
        per_query,
    )


def paired_query_cluster_bootstrap(
    labels: np.ndarray,
    query_ids: Sequence[str],
    orthogonalized_scores: np.ndarray,
    additive_scores: np.ndarray,
    *,
    seed: int,
    iterations: int,
    confidence_level: float,
    batch_size: int = 128,
) -> dict[str, Any]:
    query_order = sorted(set(query_ids))
    if not query_order:
        raise ValueError("bootstrap requires at least one query")
    query_index = {query_id: index for index, query_id in enumerate(query_order)}
    row_query_indices = np.asarray(
        [query_index[query_id] for query_id in query_ids], dtype=np.int64
    )
    counts, _ = stratified_query_bootstrap_counts(
        ["external"] * len(query_order), seed=seed, iterations=iterations
    )
    labels = np.asarray(labels, dtype=np.float64)
    orthogonalized_scores = np.asarray(orthogonalized_scores, dtype=np.float64)
    additive_scores = np.asarray(additive_scores, dtype=np.float64)
    orthogonalized_pooled = _weighted_pooled_replicates(
        counts,
        row_query_indices,
        labels,
        orthogonalized_scores,
        batch_size=batch_size,
    )
    additive_pooled = _weighted_pooled_replicates(
        counts,
        row_query_indices,
        labels,
        additive_scores,
        batch_size=batch_size,
    )

    per_query_difference = {"auprc": [], "auroc": []}
    query_array = np.asarray(query_ids, dtype=object)
    for query_id in query_order:
        selected = query_array == query_id
        if len(np.unique(labels[selected])) != 2:
            raise ValueError(f"bootstrap query lacks both classes: {query_id}")
        orthogonalized = binary_classification_metrics(
            labels[selected], orthogonalized_scores[selected]
        )
        additive = binary_classification_metrics(labels[selected], additive_scores[selected])
        for metric in ("auprc", "auroc"):
            per_query_difference[metric].append(
                float(orthogonalized[metric] - additive[metric])
            )

    replicates = {
        "query_level_macro_auprc": counts
        @ np.asarray(per_query_difference["auprc"])
        / len(query_order),
        "query_level_macro_auroc": counts
        @ np.asarray(per_query_difference["auroc"])
        / len(query_order),
        "pooled_auroc": orthogonalized_pooled["auroc"]
        - additive_pooled["auroc"],
        "pooled_auprc": orthogonalized_pooled["auprc"]
        - additive_pooled["auprc"],
    }
    return {
        "replicates": replicates,
        "confidence_intervals": {
            endpoint: _quantile_interval(values, confidence_level)
            for endpoint, values in replicates.items()
        },
        "iterations": iterations,
        "seed": seed,
        "confidence_level": confidence_level,
    }


def paired_query_permutation(
    labels: np.ndarray,
    query_ids: Sequence[str],
    orthogonalized_scores: np.ndarray,
    additive_scores: np.ndarray,
    observed_differences: Mapping[str, float],
    *,
    seed: int,
    iterations: int,
    batch_size: int = 128,
) -> dict[str, Any]:
    query_order = sorted(set(query_ids))
    query_index = {query_id: index for index, query_id in enumerate(query_order)}
    row_query_indices = np.asarray(
        [query_index[query_id] for query_id in query_ids], dtype=np.int64
    )
    labels = np.asarray(labels, dtype=np.int8)
    orthogonalized_scores = np.asarray(orthogonalized_scores, dtype=np.float64)
    additive_scores = np.asarray(additive_scores, dtype=np.float64)
    masks = generate_randomization_masks(
        len(query_order), seed=seed, iterations=iterations
    )
    pooled = paired_query_swap_replicates(
        labels,
        row_query_indices,
        additive_scores,
        orthogonalized_scores,
        masks,
        batch_size=batch_size,
    )

    query_array = np.asarray(query_ids, dtype=object)
    per_query_difference = {"auprc": [], "auroc": []}
    for query_id in query_order:
        selected = query_array == query_id
        if len(np.unique(labels[selected])) != 2:
            raise ValueError(f"permutation query lacks both classes: {query_id}")
        orthogonalized = binary_classification_metrics(
            labels[selected], orthogonalized_scores[selected]
        )
        additive = binary_classification_metrics(labels[selected], additive_scores[selected])
        for metric in ("auprc", "auroc"):
            per_query_difference[metric].append(
                float(orthogonalized[metric] - additive[metric])
            )
    signs = 1.0 - 2.0 * masks.astype(np.float64)
    null = {
        "query_level_macro_auprc": signs
        @ np.asarray(per_query_difference["auprc"])
        / len(query_order),
        "query_level_macro_auroc": signs
        @ np.asarray(per_query_difference["auroc"])
        / len(query_order),
        "pooled_auroc": pooled["auroc"],
        "pooled_auprc": pooled["auprc"],
    }
    p_values = {}
    for endpoint, values in null.items():
        threshold = abs(float(observed_differences[endpoint])) - 1e-15
        p_values[endpoint] = float(
            (1 + np.sum(np.abs(values) >= threshold)) / (iterations + 1)
        )
    return {
        "null_replicates": null,
        "p_values_two_sided": p_values,
        "iterations": iterations,
        "seed": seed,
        "unit": "query_cluster_paired_score_swap",
    }


def _current_commit(project_dir: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _audit_and_load(
    project_dir: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_yaml(config_path)
    scoring_manifest_path = project_dir / "results/external/iedb/scoring/scoring_manifest.json"
    protocol_manifest_path = project_dir / "results/external/iedb/evaluation_protocol_manifest.json"
    scoring_manifest = _load_json(scoring_manifest_path)
    protocol_manifest = _load_json(protocol_manifest_path)
    if config.get("protocol_status") != "frozen":
        raise ValueError("external evaluation protocol is not frozen")
    if config["freeze_timing"] != "pre-scoring":
        raise ValueError("external evaluation protocol was not frozen pre-scoring")
    if config["endpoints"]["primary"]["metric"] != "query_level_macro_auprc":
        raise ValueError("unexpected primary endpoint")
    if tuple(config["endpoints"]["secondary"]) != (
        "query_level_macro_auroc",
        "pooled_auroc",
        "pooled_auprc",
    ):
        raise ValueError("unexpected secondary endpoints")

    verified_protocol = _verify_protocol_sources(project_dir, protocol_manifest)
    if file_sha256(config_path) != protocol_manifest["sources"]["configuration"]["sha256"]:
        raise ValueError("configuration does not match frozen protocol manifest")

    cohort_manifest_path = _verified_path(
        project_dir, config["cohort"]["manifest"], "cohort manifest"
    )
    cohort_manifest = _load_json(cohort_manifest_path)
    test_path = _verified_path(
        project_dir, cohort_manifest["outputs"]["test"], "cohort test"
    )
    for name in ("evaluation_config", "checkpoint_registry", "cohort_manifest"):
        _verified_path(
            project_dir,
            scoring_manifest["inputs"][name],
            f"scoring input {name}",
        )
    scoring_test = scoring_manifest["inputs"]["cohort_test"]
    if file_sha256(_resolve(project_dir, scoring_test["path"])) != scoring_test[
        "source_file_sha256"
    ]:
        raise ValueError("cohort test does not match scoring manifest")
    ensemble_path = _verified_path(
        project_dir,
        scoring_manifest["outputs"]["ensemble_predictions"],
        "ensemble predictions",
    )

    cohort_columns, cohort_rows = _read_csv(test_path)
    ensemble_columns, ensemble_rows = _read_csv(ensemble_path)
    required_cohort = {"interaction_key", "query_id", "label"}
    required_ensemble = {"interaction_key", "query_id", "method", "ensemble_score"}
    if not required_cohort.issubset(cohort_columns):
        raise ValueError("cohort test is missing required columns")
    if not required_ensemble.issubset(ensemble_columns):
        raise ValueError("ensemble predictions are missing required columns")
    if "label" in ensemble_columns:
        raise ValueError("blind ensemble prediction file unexpectedly contains labels")

    expected_summary = cohort_manifest["summary"]
    expected_test_count = int(expected_summary["test_count"])
    expected_query_count = int(expected_summary["query_count"])
    expected_positive = int(expected_summary["positive_count"])
    expected_negative = int(expected_summary["negative_count"])
    if (expected_query_count, expected_test_count, expected_positive, expected_negative) != (
        141,
        1612,
        806,
        806,
    ):
        raise ValueError("frozen cohort manifest has unexpected counts")
    if len(cohort_rows) != expected_test_count:
        raise ValueError("cohort row count differs from manifest")

    cohort_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    query_ids = set()
    labels = Counter()
    for row in cohort_rows:
        key = (row["interaction_key"], row["query_id"])
        if key in cohort_by_key:
            raise ValueError(f"duplicate cohort interaction/query key: {key}")
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError("cohort labels must be binary")
        cohort_by_key[key] = {**row, "label": label}
        query_ids.add(row["query_id"])
        labels[label] += 1
    if len(query_ids) != expected_query_count or labels != Counter({0: 806, 1: 806}):
        raise ValueError("cohort labels or queries differ from manifest")

    prediction_by_method: dict[str, dict[tuple[str, str], float | None]] = {
        method: {} for method in ALL_METHODS
    }
    for row in ensemble_rows:
        method = row["method"]
        if method not in prediction_by_method:
            raise ValueError(f"unexpected ensemble method: {method}")
        key = (row["interaction_key"], row["query_id"])
        if key in prediction_by_method[method]:
            raise ValueError(f"duplicate prediction key for {method}: {key}")
        if key not in cohort_by_key:
            raise ValueError(f"prediction key absent from cohort: {key}")
        raw_score = row["ensemble_score"].strip()
        score = None if raw_score == "" else float(raw_score)
        if score is not None and (not math.isfinite(score) or not 0.0 <= score <= 1.0):
            raise ValueError(f"invalid ensemble score for {method}: {score}")
        prediction_by_method[method][key] = score
    cohort_keys = set(cohort_by_key)
    for method, predictions in prediction_by_method.items():
        if set(predictions) != cohort_keys:
            raise ValueError(f"prediction keys do not exactly match cohort for {method}")
        if method in FULL_METHODS and any(value is None for value in predictions.values()):
            raise ValueError(f"full-coverage method has missing scores: {method}")

    similarity_available_by_query: dict[str, set[bool]] = defaultdict(set)
    for (interaction_key, query_id), value in prediction_by_method[
        SIMILARITY_METHOD
    ].items():
        del interaction_key
        similarity_available_by_query[query_id].add(value is not None)
    if any(len(statuses) != 1 for statuses in similarity_available_by_query.values()):
        raise ValueError("Similarity availability differs within a query")
    similarity_queries = {
        query_id
        for query_id, statuses in similarity_available_by_query.items()
        if statuses == {True}
    }
    if len(similarity_queries) != scoring_manifest["audit"][
        "similarity_available_query_count"
    ]:
        raise ValueError("Similarity query coverage differs from scoring manifest")
    if len(similarity_queries) != 112:
        raise ValueError("Similarity common-query count is not the frozen 112")

    ordered_keys = sorted(cohort_keys, key=lambda key: (key[1], key[0]))
    joined = []
    for key in ordered_keys:
        source = cohort_by_key[key]
        joined.append(
            {
                "interaction_key": key[0],
                "query_id": key[1],
                "label": source["label"],
                "scores": {
                    method: prediction_by_method[method][key] for method in ALL_METHODS
                },
            }
        )
    audit = {
        "cohort_query_count": len(query_ids),
        "cohort_record_count": len(joined),
        "positive_count": labels[1],
        "negative_count": labels[0],
        "full_coverage_methods": list(FULL_METHODS),
        "all_full_methods_cover_every_record": True,
        "similarity_available_query_count": len(similarity_queries),
        "similarity_total_query_count": len(query_ids),
        "similarity_coverage_fraction": len(similarity_queries) / len(query_ids),
        "label_and_score_keys_complete_and_unique": True,
        "all_input_sha256_match": True,
        "verified_protocol_artifacts": verified_protocol,
        "sources": {
            "configuration": {
                "path": _relative(project_dir, config_path),
                "sha256": file_sha256(config_path),
            },
            "cohort_test": {
                "path": _relative(project_dir, test_path),
                "sha256": file_sha256(test_path),
            },
            "ensemble_predictions": {
                "path": _relative(project_dir, ensemble_path),
                "sha256": file_sha256(ensemble_path),
            },
            "scoring_manifest": {
                "path": _relative(project_dir, scoring_manifest_path),
                "sha256": file_sha256(scoring_manifest_path),
            },
            "evaluation_protocol_manifest": {
                "path": _relative(project_dir, protocol_manifest_path),
                "sha256": file_sha256(protocol_manifest_path),
            },
            "cohort_manifest": {
                "path": _relative(project_dir, cohort_manifest_path),
                "sha256": file_sha256(cohort_manifest_path),
            },
        },
    }
    return {"config": config, "joined": joined, "similarity_queries": similarity_queries}, audit


def _evaluate_once(loaded: Mapping[str, Any]) -> dict[str, Any]:
    config = loaded["config"]
    joined = loaded["joined"]
    labels = np.asarray([row["label"] for row in joined], dtype=np.int8)
    query_ids = [str(row["query_id"]) for row in joined]
    total_query_count = len(set(query_ids))
    method_scores = {
        method: np.asarray([row["scores"][method] for row in joined], dtype=np.float64)
        for method in FULL_METHODS
    }

    metric_rows = []
    per_query_rows = []
    full_metrics = {}
    full_per_query = {}
    for method in FULL_METHODS:
        metrics, rows, query_values = _absolute_metrics(
            labels, method_scores[method], query_ids, method=method
        )
        if len(query_values) != total_query_count:
            raise ValueError(f"full-cohort macro metric excluded queries for {method}")
        full_metrics[method] = metrics
        full_per_query[method] = query_values
        per_query_rows.extend(rows)
        metric_rows.append(
            {
                "population": "full_cohort",
                "method": method,
                "query_count": total_query_count,
                "total_query_count": total_query_count,
                "record_count": len(joined),
                "coverage_fraction": _format_float(1.0),
                **{name: _format_float(metrics[name]) for name in ENDPOINTS},
            }
        )

    similarity_scores = np.asarray(
        [
            np.nan if row["scores"][SIMILARITY_METHOD] is None else row["scores"][SIMILARITY_METHOD]
            for row in joined
        ],
        dtype=np.float64,
    )
    similarity_available = np.isfinite(similarity_scores)
    similarity_rows, similarity_per_query = _query_metric_rows(
        labels,
        np.nan_to_num(similarity_scores),
        query_ids,
        method=SIMILARITY_METHOD,
        available=similarity_available,
    )
    per_query_rows.extend(similarity_rows)
    common_queries = set(loaded["similarity_queries"])
    if set(similarity_per_query) != common_queries:
        raise ValueError("Similarity evaluable queries do not match common-query audit")
    common_mask = np.asarray([query_id in common_queries for query_id in query_ids])
    common_metrics = {}
    for method in ALL_METHODS:
        scores = similarity_scores if method == SIMILARITY_METHOD else method_scores[method]
        metrics, _, query_values = _absolute_metrics(
            labels[common_mask],
            scores[common_mask],
            np.asarray(query_ids, dtype=object)[common_mask].tolist(),
            method=method,
        )
        if len(query_values) != len(common_queries):
            raise ValueError(f"common-query macro metric excluded queries for {method}")
        common_metrics[method] = metrics
        metric_rows.append(
            {
                "population": "similarity_common",
                "method": method,
                "query_count": len(common_queries),
                "total_query_count": total_query_count,
                "record_count": int(common_mask.sum()),
                "coverage_fraction": _format_float(len(common_queries) / total_query_count),
                **{name: _format_float(metrics[name]) for name in ENDPOINTS},
            }
        )

    observed_differences = {
        endpoint: full_metrics[PRIMARY_METHOD][endpoint]
        - full_metrics[COMPARATOR_METHOD][endpoint]
        for endpoint in ENDPOINTS
    }
    bootstrap_config = config["statistics"]["bootstrap"]
    bootstrap = paired_query_cluster_bootstrap(
        labels,
        query_ids,
        method_scores[PRIMARY_METHOD],
        method_scores[COMPARATOR_METHOD],
        seed=int(bootstrap_config["seed"]),
        iterations=int(bootstrap_config["iterations"]),
        confidence_level=float(bootstrap_config["confidence_level"]),
    )
    permutation_config = config["statistics"]["permutation"]
    permutation = paired_query_permutation(
        labels,
        query_ids,
        method_scores[PRIMARY_METHOD],
        method_scores[COMPARATOR_METHOD],
        observed_differences,
        seed=int(permutation_config["seed"]),
        iterations=int(permutation_config["iterations"]),
    )
    secondary = tuple(config["endpoints"]["secondary"])
    holm = holm_adjust(
        {
            endpoint: permutation["p_values_two_sided"][endpoint]
            for endpoint in secondary
        }
    )
    comparison_rows = []
    comparison_json = {}
    for endpoint in ENDPOINTS:
        lower, upper = bootstrap["confidence_intervals"][endpoint]
        role = "primary" if endpoint == "query_level_macro_auprc" else "secondary"
        row = {
            "endpoint": endpoint,
            "endpoint_role": role,
            "orthogonalized": _format_float(full_metrics[PRIMARY_METHOD][endpoint]),
            "additive_only": _format_float(full_metrics[COMPARATOR_METHOD][endpoint]),
            "difference": _format_float(observed_differences[endpoint]),
            "ci_lower": _format_float(lower),
            "ci_upper": _format_float(upper),
            "p_value_two_sided": _format_float(
                permutation["p_values_two_sided"][endpoint]
            ),
            "p_value_holm": "" if role == "primary" else _format_float(holm[endpoint]),
            "bootstrap_iterations": bootstrap["iterations"],
            "bootstrap_seed": bootstrap["seed"],
            "permutation_iterations": permutation["iterations"],
            "permutation_seed": permutation["seed"],
        }
        comparison_rows.append(row)
        comparison_json[endpoint] = {
            "role": role,
            "orthogonalized": full_metrics[PRIMARY_METHOD][endpoint],
            "additive_only": full_metrics[COMPARATOR_METHOD][endpoint],
            "difference": observed_differences[endpoint],
            "confidence_interval_95": [lower, upper],
            "p_value_two_sided": permutation["p_values_two_sided"][endpoint],
            "p_value_holm": None if role == "primary" else holm[endpoint],
        }

    per_query_rows.sort(key=lambda row: (row["method"], row["query_id"]))
    return {
        "metric_rows": metric_rows,
        "per_query_rows": per_query_rows,
        "comparison_rows": comparison_rows,
        "comparison_json": {
            "comparison": "orthogonalized_interaction_minus_additive_only",
            "population": "all_141_formal_cohort_queries",
            "primary_endpoint": "query_level_macro_auprc",
            "multiplicity": {
                "primary": "unadjusted_pre_specified_primary",
                "secondary": "holm_across_three_secondary_endpoints",
            },
            "statistics": {
                "bootstrap": {
                    "type": "paired_query_cluster",
                    "unit": "query",
                    "iterations": bootstrap["iterations"],
                    "seed": bootstrap["seed"],
                    "confidence_level": bootstrap["confidence_level"],
                },
                "permutation": {
                    "type": "query_level_paired_score_swap",
                    "unit": "query",
                    "iterations": permutation["iterations"],
                    "seed": permutation["seed"],
                    "alternative": "two_sided",
                },
            },
            "endpoints": comparison_json,
        },
        "full_metrics": full_metrics,
        "common_metrics": common_metrics,
        "common_query_count": len(common_queries),
        "common_record_count": int(common_mask.sum()),
        "total_query_count": total_query_count,
    }


def _report(result: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    lines = [
        "# IEDB external evaluation",
        "",
        "This report applies the pre-scoring frozen protocol without training, rescoring, calibration, threshold selection, or exploratory stratification.",
        "",
        "## Input audit",
        "",
        f"- Cohort: {audit['cohort_query_count']} queries, {audit['cohort_record_count']} records ({audit['positive_count']} positive / {audit['negative_count']} negative).",
        f"- Full score coverage: {', '.join(audit['full_coverage_methods'])}.",
        f"- Similarity coverage: {audit['similarity_available_query_count']}/{audit['similarity_total_query_count']} queries ({audit['similarity_coverage_fraction']:.6f}).",
        "- All declared source SHA-256 values, row keys, labels, and prediction keys passed validation.",
        "",
        "## Full-cohort metrics",
        "",
        "| Method | Macro AUPRC | Macro AUROC | Pooled AUPRC | Pooled AUROC |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in FULL_METHODS:
        metrics = result["full_metrics"][method]
        lines.append(
            f"| {method} | {metrics['query_level_macro_auprc']:.6f} | {metrics['query_level_macro_auroc']:.6f} | {metrics['pooled_auprc']:.6f} | {metrics['pooled_auroc']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen primary comparison",
            "",
            "Orthogonalized-interaction minus Additive-only; Macro AUPRC is the sole primary endpoint. Holm adjustment is applied only across the three secondary endpoints.",
            "",
            "| Endpoint | Difference | 95% CI | Two-sided p | Holm p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["comparison_rows"]:
        holm = row["p_value_holm"] or "not applied"
        lines.append(
            f"| {row['endpoint']} | {row['difference']} | [{row['ci_lower']}, {row['ci_upper']}] | {row['p_value_two_sided']} | {holm} |"
        )
    lines.extend(
        [
            "",
            "## Similarity common-query comparison",
            "",
            f"Similarity is evaluable for {result['common_query_count']}/{result['total_query_count']} queries and is reported only on that common subset.",
            "",
            "| Method | Macro AUPRC | Macro AUROC | Pooled AUPRC | Pooled AUROC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in ALL_METHODS:
        metrics = result["common_metrics"][method]
        lines.append(
            f"| {method} | {metrics['query_level_macro_auprc']:.6f} | {metrics['query_level_macro_auroc']:.6f} | {metrics['pooled_auprc']:.6f} | {metrics['pooled_auroc']:.6f} |"
        )
    lines.extend(
        [
            "",
            "No seen/unseen stratification or figure was generated.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_iedb_external(
    project_dir: Path,
    config_path: Path,
    *,
    output_dir: Path | None = None,
    analysis_source_commit: str | None = None,
) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    config_path = config_path.resolve()
    target = (
        output_dir.resolve()
        if output_dir is not None
        else project_dir / "results/external/iedb/evaluation"
    )
    source_commit = analysis_source_commit or _current_commit(project_dir)
    if target.exists():
        unexpected = {path.name for path in target.iterdir()} - EXPECTED_OUTPUT_FILES
        if unexpected:
            raise ValueError(
                "Evaluation output directory contains unexpected user-owned files: "
                + ", ".join(sorted(unexpected))
            )

    loaded, audit = _audit_and_load(project_dir, config_path)
    first = _evaluate_once(loaded)
    second = _evaluate_once(loaded)
    deterministic_payloads = {
        "external_metrics.csv": _csv_bytes(METRICS_COLUMNS, first["metric_rows"]),
        "external_per_query_metrics.csv": _csv_bytes(
            PER_QUERY_COLUMNS, first["per_query_rows"]
        ),
        "primary_comparison.csv": _csv_bytes(
            COMPARISON_COLUMNS, first["comparison_rows"]
        ),
        "primary_comparison.json": _json_bytes(first["comparison_json"]),
        "external_evaluation_report.md": _report(first, audit).encode("utf-8"),
    }
    repeated_payloads = {
        "external_metrics.csv": _csv_bytes(METRICS_COLUMNS, second["metric_rows"]),
        "external_per_query_metrics.csv": _csv_bytes(
            PER_QUERY_COLUMNS, second["per_query_rows"]
        ),
        "primary_comparison.csv": _csv_bytes(
            COMPARISON_COLUMNS, second["comparison_rows"]
        ),
        "primary_comparison.json": _json_bytes(second["comparison_json"]),
        "external_evaluation_report.md": _report(second, audit).encode("utf-8"),
    }
    if deterministic_payloads != repeated_payloads:
        raise ValueError("repeated external evaluation did not reproduce identical outputs")

    target.mkdir(parents=True, exist_ok=True)
    for filename, payload in deterministic_payloads.items():
        (target / filename).write_bytes(payload)
    output_records = {
        filename: {
            "path": _relative(project_dir, target / filename),
            "sha256": file_sha256(target / filename),
            "rows": (
                len(first["metric_rows"])
                if filename == "external_metrics.csv"
                else len(first["per_query_rows"])
                if filename == "external_per_query_metrics.csv"
                else len(first["comparison_rows"])
                if filename == "primary_comparison.csv"
                else None
            ),
        }
        for filename in deterministic_payloads
    }
    manifest = {
        "schema_version": 1,
        "analysis_source_commit": source_commit,
        "mode": "frozen_iedb_external_validation_statistics",
        "inputs": audit["sources"],
        "audit": {key: value for key, value in audit.items() if key != "sources"},
        "statistics": first["comparison_json"]["statistics"],
        "primary_endpoint": "query_level_macro_auprc",
        "secondary_endpoints": [
            "query_level_macro_auroc",
            "pooled_auroc",
            "pooled_auprc",
        ],
        "similarity_population": {
            "rule": "common_evaluable_queries_only",
            "query_count": first["common_query_count"],
            "record_count": first["common_record_count"],
        },
        "reproducibility": {
            "complete_analysis_runs_compared": 2,
            "all_deterministic_output_bytes_identical": True,
        },
        "operations": {
            "training": False,
            "rescoring": False,
            "hyperparameter_tuning": False,
            "calibration": False,
            "threshold_selection": False,
            "plotting": False,
            "seen_unseen_stratification": False,
        },
        "outputs": output_records,
    }
    manifest_path = target / "evaluation_manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    return {
        "output_dir": _relative(project_dir, target),
        "full_metrics": first["full_metrics"],
        "comparison": first["comparison_json"]["endpoints"],
        "similarity_common_query_count": first["common_query_count"],
        "similarity_common_metrics": first["common_metrics"],
        "outputs": {**output_records, "evaluation_manifest.json": {
            "path": _relative(project_dir, manifest_path),
            "sha256": file_sha256(manifest_path),
        }},
    }
