from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .five_fold_summary import file_sha256
from .metrics import binary_classification_metrics, query_level_classification_metrics
from .query_balanced_paired_statistics import (
    _quantile_interval,
    _weighted_pooled_replicates,
    stratified_query_bootstrap_counts,
)


METRIC_NAMES = ("pooled_auroc", "pooled_auprc", "macro_auroc", "macro_auprc")
METHODS = ("cnn", "logistic", "similarity")
COMPARISON_SPECS = (
    ("logistic_minus_cnn", "logistic", "cnn", "all_168"),
    ("logistic_minus_similarity", "logistic", "similarity", "similarity_common_153"),
    ("similarity_minus_cnn", "similarity", "cnn", "similarity_common_153"),
)
CSV_COLUMNS = (
    "record_type",
    "comparison",
    "query_scope",
    "metric",
    "left_method",
    "right_method",
    "left_estimate",
    "right_estimate",
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    rows = _read_csv(path)
    return path, rows, {
        "path": _relative(project_dir, path),
        "sha256": observed_sha,
        "rows": len(rows),
    }


def _verified_explicit_csv(
    project_dir: Path, value: str, description: str
) -> tuple[Path, list[dict[str, str]], dict[str, Any]]:
    path = _resolve(project_dir, value)
    rows = _read_csv(path)
    return path, rows, {
        "path": _relative(project_dir, path),
        "sha256": file_sha256(path),
        "rows": len(rows),
        "integrity_check": description,
    }


def stratified_multi_method_bootstrap(
    predictions: Sequence[Mapping[str, Any]],
    per_query: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    comparisons: Sequence[tuple[str, str, str]],
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
    query_rows = sorted(
        per_query, key=lambda row: (str(row["fold_id"]), str(row["query_id"]))
    )
    query_ids = [str(row["query_id"]) for row in query_rows]
    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError("bootstrap query IDs must be non-empty and unique")
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    query_folds = [str(row["fold_id"]) for row in query_rows]
    counts, stratum_sizes = stratified_query_bootstrap_counts(
        query_folds, seed=seed, iterations=iterations
    )
    prediction_query_ids = [str(row["query_id"]) for row in predictions]
    if set(prediction_query_ids) != set(query_ids):
        raise ValueError("prediction and per-query query sets differ")
    row_query_indices = np.asarray(
        [query_index[query_id] for query_id in prediction_query_ids], dtype=np.int64
    )
    labels = np.asarray([int(row["label"]) for row in predictions], dtype=np.int64)
    method_scores = {
        method: np.asarray(
            [float(row[f"{method}_score"]) for row in predictions], dtype=np.float64
        )
        for method in methods
    }
    if any(not np.isfinite(scores).all() for scores in method_scores.values()):
        raise ValueError("method scores must be finite")

    pooled_points = {
        method: binary_classification_metrics(labels, scores)
        for method, scores in method_scores.items()
    }
    pooled_replicates = {
        method: _weighted_pooled_replicates(
            counts,
            row_query_indices,
            labels.astype(np.float64),
            scores,
            batch_size=batch_size,
        )
        for method, scores in method_scores.items()
    }
    query_values = {
        method: {
            metric: np.asarray(
                [float(row[f"{method}_{metric}"]) for row in query_rows],
                dtype=np.float64,
            )
            for metric in ("auroc", "auprc")
        }
        for method in methods
    }
    query_count = len(query_rows)
    macro_replicates = {
        method: {
            metric: counts @ values / query_count
            for metric, values in method_values.items()
        }
        for method, method_values in query_values.items()
    }
    method_metrics = {}
    for method in methods:
        method_metrics[method] = {
            "pooled_auroc": float(pooled_points[method]["auroc"]),
            "pooled_auprc": float(pooled_points[method]["auprc"]),
            "macro_auroc": float(query_values[method]["auroc"].mean()),
            "macro_auprc": float(query_values[method]["auprc"].mean()),
        }

    comparison_results = {}
    for comparison_name, left, right in comparisons:
        if left not in methods or right not in methods:
            raise ValueError(f"comparison method missing: {comparison_name}")
        metrics = {}
        for metric_name, scope, metric in (
            ("pooled_auroc", "pooled", "auroc"),
            ("pooled_auprc", "pooled", "auprc"),
            ("macro_auroc", "macro", "auroc"),
            ("macro_auprc", "macro", "auprc"),
        ):
            if scope == "pooled":
                differences = (
                    pooled_replicates[left][metric] - pooled_replicates[right][metric]
                )
            else:
                differences = (
                    macro_replicates[left][metric] - macro_replicates[right][metric]
                )
            lower, upper = _quantile_interval(differences, confidence_level)
            left_estimate = method_metrics[left][metric_name]
            right_estimate = method_metrics[right][metric_name]
            metrics[metric_name] = {
                "left_estimate": left_estimate,
                "right_estimate": right_estimate,
                "difference": left_estimate - right_estimate,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        outcomes = {}
        for metric in ("auroc", "auprc"):
            differences = query_values[left][metric] - query_values[right][metric]
            outcomes[metric] = {
                "wins": int(np.sum(differences > tie_tolerance)),
                "ties": int(np.sum(np.abs(differences) <= tie_tolerance)),
                "losses": int(np.sum(differences < -tie_tolerance)),
            }
        comparison_results[comparison_name] = {
            "left_method": left,
            "right_method": right,
            "query_count": query_count,
            "metrics": metrics,
            "query_outcomes": outcomes,
        }
    return {
        "query_count": query_count,
        "method_metrics": method_metrics,
        "comparisons": comparison_results,
        "stratum_query_counts": stratum_sizes,
        "seed": seed,
        "iterations": iterations,
        "confidence_level": confidence_level,
        "tie_tolerance": tie_tolerance,
        "resampling_unit": "query_cluster",
        "stratified_by": "fold_id",
        "rows_within_query_resampled_together": True,
    }


def _manifest_expected_rows(
    project_dir: Path, manifest: Mapping[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    expected: list[dict[str, str]] = []
    input_audit: dict[str, Any] = {}
    for fold_id in sorted(manifest["folds"]):
        fold_audit = {}
        partition = manifest["folds"][fold_id]["partitions"]["test"]
        for kind, output_key, identifier, source_name in (
            ("positive", "positive_output", "record_id", "positive"),
            ("negative", "negative_output", "negative_id", "putative_negative"),
        ):
            output = partition[output_key]
            path = _resolve(project_dir, output["path"])
            observed_sha = file_sha256(path)
            if observed_sha != output["sha256"]:
                raise ValueError(f"manifest SHA-256 mismatch for {fold_id} test {kind}")
            rows = _read_csv(path)
            if len(rows) != int(output["rows"]):
                raise ValueError(f"manifest row count mismatch for {fold_id} test {kind}")
            for row in rows:
                expected.append(
                    {
                        "fold_id": fold_id,
                        "sample_id": row[identifier],
                        "source": source_name,
                        "pmhc_id": row["pmhc_id"],
                        "label": row["label"],
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
    return expected, input_audit


def _verify_summary_input_hashes(
    manifest: Mapping[str, Any],
    cnn_summary: Mapping[str, Any],
    logistic_summary: Mapping[str, Any],
    similarity_summary: Mapping[str, Any],
) -> None:
    for fold_id, fold in manifest["folds"].items():
        for partition, entry in fold["partitions"].items():
            for kind, output_key in (("positive", "positive_output"), ("negative", "negative_output")):
                expected = entry[output_key]
                cnn = cnn_summary["audit"]["fold_inputs"][fold_id]["partitions"][partition][kind]
                logistic = logistic_summary["audit"]["inputs"][fold_id][partition][kind]
                for method, observed in (("CNN", cnn), ("Logistic", logistic)):
                    if observed["path"] != expected["path"] or observed["sha256"] != expected["sha256"] or int(observed["rows"]) != int(expected["rows"]):
                        raise ValueError(f"{method} input manifest mismatch for {fold_id} {partition} {kind}")
        similarity_fold = similarity_summary["audit"]["fold_inputs"][fold_id]
        for name, partition, output_key in (
            ("train_positive", "train", "positive_output"),
            ("test_positive", "test", "positive_output"),
            ("test_negative", "test", "negative_output"),
        ):
            expected = fold["partitions"][partition][output_key]
            observed = similarity_fold[name]
            if observed["path"] != expected["path"] or observed["sha256"] != expected["sha256"] or int(observed["rows"]) != int(expected["rows"]):
                raise ValueError(f"Similarity input manifest mismatch for {fold_id} {name}")


def _per_query_index(rows: Sequence[Mapping[str, str]], method: str) -> dict[str, Mapping[str, str]]:
    index = {row["query_id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"duplicate {method} per-query row")
    return index


def _recompute_per_query(
    rows: Sequence[Mapping[str, Any]], methods: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    method_indices = {}
    method_metrics = {}
    for method in methods:
        macro, per_query = query_level_classification_metrics(
            [
                {
                    "pmhc_id": row["query_id"],
                    "label": row["label"],
                    "probability": row[f"{method}_score"],
                }
                for row in rows
            ]
        )
        method_indices[method] = {row["query_id"]: row for row in per_query}
        pooled = binary_classification_metrics(
            [int(row["label"]) for row in rows],
            [float(row[f"{method}_score"]) for row in rows],
        )
        method_metrics[method] = {
            "pooled_auroc": pooled["auroc"],
            "pooled_auprc": pooled["auprc"],
            "macro_auroc": float(macro["auroc"]),
            "macro_auprc": float(macro["auprc"]),
        }
    query_ids = set(method_indices[methods[0]])
    if any(set(method_indices[method]) != query_ids for method in methods[1:]):
        raise ValueError("recomputed method query IDs differ")
    combined = []
    for query_id in sorted(query_ids):
        first = method_indices[methods[0]][query_id]
        row = {"query_id": query_id, "fold_id": ""}
        for method in methods:
            item = method_indices[method][query_id]
            for field in ("positive_count", "negative_count", "sample_count"):
                if item[field] != first[field]:
                    raise ValueError(f"recomputed query count mismatch for {query_id}")
            row[f"{method}_auroc"] = float(item["auroc"])
            row[f"{method}_auprc"] = float(item["auprc"])
        combined.append(row)
    return combined, method_metrics


def _audit_and_join(
    *,
    project_dir: Path,
    manifest: Mapping[str, Any],
    cnn_rows: Sequence[Mapping[str, str]],
    cnn_per_query: Sequence[Mapping[str, str]],
    logistic_rows: Sequence[Mapping[str, str]],
    logistic_per_query: Sequence[Mapping[str, str]],
    similarity_rows: Sequence[Mapping[str, str]],
    similarity_per_query: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    expected_rows, manifest_audit = _manifest_expected_rows(project_dir, manifest)
    row_count = len(expected_rows)
    if {len(cnn_rows), len(logistic_rows), len(similarity_rows)} != {row_count}:
        raise ValueError("OOF row counts do not match the manifest")
    canonical = []
    identity_fields = ("fold_id", "sample_id", "source", "pmhc_id", "label")
    interaction_fields = ("tcr_cdr3_beta", "epitope", "hla")
    for expected, cnn, logistic, similarity in zip(
        expected_rows, cnn_rows, logistic_rows, similarity_rows, strict=True
    ):
        for field in identity_fields:
            for method, observed in (("CNN", cnn), ("Logistic", logistic), ("Similarity", similarity)):
                if str(observed[field]) != str(expected[field]):
                    raise ValueError(f"{method} OOF {field} mismatch for {expected['sample_id']}")
        for field in interaction_fields:
            if logistic[field] != expected[field] or similarity[field] != expected[field]:
                raise ValueError(f"OOF interaction field {field} mismatch for {expected['sample_id']}")
        _assert_close(
            float(similarity["cnn_probability"]),
            float(cnn["probability"]),
            f"Similarity embedded CNN score {expected['sample_id']}",
        )
        status = similarity["status"]
        if status not in {"available", "unavailable"}:
            raise ValueError(f"invalid similarity status: {status}")
        if status == "available" and not similarity["similarity_score"]:
            raise ValueError("available similarity query has a missing score")
        if status == "unavailable" and similarity["similarity_score"]:
            raise ValueError("unavailable similarity query has a score")
        canonical.append(
            {
                **expected,
                "query_id": expected["pmhc_id"],
                "cnn_score": float(cnn["probability"]),
                "logistic_score": float(logistic["probability"]),
                "similarity_status": status,
                "similarity_score": (
                    float(similarity["similarity_score"]) if status == "available" else None
                ),
            }
        )

    cnn_index = _per_query_index(cnn_per_query, "CNN")
    logistic_index = _per_query_index(logistic_per_query, "Logistic")
    similarity_index = _per_query_index(similarity_per_query, "Similarity")
    query_ids = set(cnn_index)
    if len(query_ids) != 168 or set(logistic_index) != query_ids or set(similarity_index) != query_ids:
        raise ValueError("expected 168 identical per-query IDs across methods")
    available_query_ids = {
        query_id for query_id, row in similarity_index.items() if row["status"] == "available"
    }
    if len(available_query_ids) != 153:
        raise ValueError("expected exactly 153 similarity-evaluable queries")

    all_per_query, _ = _recompute_per_query(canonical, ("cnn", "logistic"))
    fold_by_query = {}
    for query_id in sorted(query_ids):
        source_rows = (cnn_index[query_id], logistic_index[query_id], similarity_index[query_id])
        fold_ids = {row["fold_id"] for row in source_rows}
        if len(fold_ids) != 1:
            raise ValueError(f"per-query fold ID mismatch for {query_id}")
        fold_by_query[query_id] = next(iter(fold_ids))
        for field in ("epitope", "hla", "positive_count", "negative_count", "sample_count"):
            if len({row[field] for row in source_rows}) != 1:
                raise ValueError(f"per-query {field} mismatch for {query_id}")
    for row in all_per_query:
        query_id = row["query_id"]
        row["fold_id"] = fold_by_query[query_id]
        for method, source in (("cnn", cnn_index[query_id]), ("logistic", logistic_index[query_id])):
            for metric in ("auroc", "auprc"):
                _assert_close(
                    row[f"{method}_{metric}"],
                    float(source[metric]),
                    f"{method} per-query {metric} {query_id}",
                )

    common_rows = [row for row in canonical if row["query_id"] in available_query_ids]
    common_per_query, common_method_metrics = _recompute_per_query(
        common_rows, ("cnn", "logistic", "similarity")
    )
    for row in common_per_query:
        query_id = row["query_id"]
        row["fold_id"] = fold_by_query[query_id]
        similarity_source = similarity_index[query_id]
        for method, source, prefix in (
            ("cnn", cnn_index[query_id], ""),
            ("logistic", logistic_index[query_id], ""),
            ("similarity", similarity_source, "similarity_"),
        ):
            for metric in ("auroc", "auprc"):
                source_value = source[f"{prefix}{metric}"] if prefix else source[metric]
                _assert_close(
                    row[f"{method}_{metric}"],
                    float(source_value),
                    f"{method} common per-query {metric} {query_id}",
                )
        for metric in ("auroc", "auprc"):
            _assert_close(
                row[f"cnn_{metric}"],
                float(similarity_source[f"cnn_{metric}"]),
                f"Similarity embedded CNN per-query {metric} {query_id}",
            )

    interaction_payload = json.dumps(
        [
            [row[field] for field in ("fold_id", "sample_id", "tcr_cdr3_beta", "epitope", "hla")]
            for row in canonical
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return canonical, common_per_query, {
        "manifest_test_inputs": manifest_audit,
        "query_ids_match": True,
        "fold_ids_match": True,
        "sample_labels_match": True,
        "interaction_keys_match": True,
        "row_order_matches": True,
        "cnn_score_matches_similarity_copy": True,
        "all_query_count": len(query_ids),
        "similarity_common_query_count": len(available_query_ids),
        "all_sample_count": len(canonical),
        "similarity_common_sample_count": len(common_rows),
        "interaction_key_sha256": hashlib.sha256(interaction_payload).hexdigest(),
        "common_method_metrics": common_method_metrics,
    }


def _csv_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    bootstrap = result["bootstrap"]
    rows = []
    for comparison_name, comparison in result["comparisons"].items():
        common = {
            "comparison": comparison_name,
            "query_scope": comparison["query_scope"],
            "left_method": comparison["left_method"],
            "right_method": comparison["right_method"],
            "query_count": comparison["query_count"],
            "bootstrap_iterations": bootstrap["iterations"],
            "bootstrap_seed": bootstrap["seed"],
            "confidence_level": bootstrap["confidence_level"],
            "resampling_unit": bootstrap["resampling_unit"],
            "stratified_by": bootstrap["stratified_by"],
        }
        for metric in METRIC_NAMES:
            rows.append(
                {
                    "record_type": "paired_difference",
                    "metric": metric,
                    **comparison["metrics"][metric],
                    **common,
                }
            )
        for metric in ("auroc", "auprc"):
            rows.append(
                {
                    "record_type": "query_outcome",
                    "metric": metric,
                    **comparison["query_outcomes"][metric],
                    **common,
                }
            )
    return rows


def _render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Five-fold OOF baseline paired statistics",
        "",
        "- Logistic vs CNN scope: all 168 shared test queries.",
        "- Three-method scope: the 153 queries evaluable by Similarity.",
        f"- Bootstrap: {result['bootstrap']['iterations']:,} fixed-seed, fold-stratified query-cluster resamples.",
        "- Pooled metrics resample whole query clusters; rows are never sampled independently.",
    ]
    for comparison_name, comparison in result["comparisons"].items():
        lines.extend(
            [
                "",
                f"## {comparison_name}",
                "",
                f"Queries: {comparison['query_count']}",
                "",
                "| Metric | Left | Right | Difference | 95% CI |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric in METRIC_NAMES:
            row = comparison["metrics"][metric]
            lines.append(
                f"| {metric} | {row['left_estimate']:.6f} | {row['right_estimate']:.6f} | "
                f"{row['difference']:+.6f} | [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}] |"
            )
        lines.extend(["", "| Query metric | Wins | Ties | Losses |", "|---|---:|---:|---:|"])
        for metric in ("auroc", "auprc"):
            outcome = comparison["query_outcomes"][metric]
            lines.append(
                f"| {metric} | {outcome['wins']} | {outcome['ties']} | {outcome['losses']} |"
            )
    lines.extend(
        [
            "",
            "All point estimates were recomputed from existing OOF scores. No rescoring, training, or source-result modification was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def build_baseline_paired_statistics(
    config_path: Path, project_dir: Path
) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    bootstrap_config = config["bootstrap"]
    if bootstrap_config["resampling_unit"] != "query_cluster":
        raise ValueError("pooled bootstrap must use query clusters")
    if bootstrap_config["stratified_by"] != "fold_id":
        raise ValueError("paired bootstrap must be stratified by fold ID")
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
    for method, summary in (
        ("CNN", cnn_summary),
        ("Logistic", logistic_summary),
        ("Similarity", similarity_summary),
    ):
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
    all_per_query, _ = _recompute_per_query(canonical, ("cnn", "logistic"))
    fold_by_query = {row["query_id"]: row["fold_id"] for row in cnn_per_query}
    for row in all_per_query:
        row["fold_id"] = fold_by_query[row["query_id"]]

    kwargs = {
        "seed": int(bootstrap_config["seed"]),
        "iterations": int(bootstrap_config["iterations"]),
        "confidence_level": float(bootstrap_config["confidence_level"]),
        "batch_size": int(bootstrap_config["batch_size"]),
        "tie_tolerance": float(bootstrap_config["tie_tolerance"]),
    }
    all_bootstrap = stratified_multi_method_bootstrap(
        canonical,
        all_per_query,
        methods=("cnn", "logistic"),
        comparisons=(("logistic_minus_cnn", "logistic", "cnn"),),
        **kwargs,
    )
    common_bootstrap = stratified_multi_method_bootstrap(
        common_rows,
        common_per_query,
        methods=METHODS,
        comparisons=(
            ("logistic_minus_similarity", "logistic", "similarity"),
            ("similarity_minus_cnn", "similarity", "cnn"),
        ),
        **kwargs,
    )
    comparisons = {}
    for comparison_name, _, _, scope in COMPARISON_SPECS:
        source = all_bootstrap if scope == "all_168" else common_bootstrap
        comparisons[comparison_name] = {
            **source["comparisons"][comparison_name],
            "query_scope": scope,
        }
    result: dict[str, Any] = {
        "schema_version": 1,
        "comparisons": comparisons,
        "common_153_method_metrics": common_bootstrap["method_metrics"],
        "bootstrap": {
            "seed": kwargs["seed"],
            "iterations": kwargs["iterations"],
            "confidence_level": kwargs["confidence_level"],
            "tie_tolerance": kwargs["tie_tolerance"],
            "resampling_unit": "query_cluster",
            "stratified_by": "fold_id",
            "rows_within_query_resampled_together": True,
            "all_168_stratum_query_counts": all_bootstrap["stratum_query_counts"],
            "common_153_stratum_query_counts": common_bootstrap["stratum_query_counts"],
        },
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
            "input_sha256_consistent_with_manifest": True,
            "rescoring_performed": False,
            "training_performed": False,
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
