from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .five_fold_summary import (
    file_sha256,
    validate_query_test_rotation_from_inputs,
)
from .query_balanced_five_fold_similarity import (
    METRIC_NAMES,
    _classification_metrics,
    _delta,
    _flatten_metrics,
    _verified_rows,
    score_fixed_test_rows,
)


PREDICTION_COLUMNS = (
    "fold_id",
    "sample_id",
    "source",
    "tier",
    "strategy",
    "negative_strategy",
    "split",
    "pmhc_id",
    "tcr_cdr3_beta",
    "epitope",
    "hla",
    "label",
    "status",
    "similarity_score",
    "cnn_probability",
    "full_logistic_probability",
)
PER_QUERY_COLUMNS = (
    "fold_id",
    "query_id",
    "epitope",
    "hla",
    "status",
    "positive_count",
    "negative_count",
    "sample_count",
    "selected_training_epitopes",
    "training_reference_tcr_count",
    "similarity_auroc",
    "similarity_auprc",
    "cnn_auroc",
    "cnn_auprc",
    "full_logistic_auroc",
    "full_logistic_auprc",
    "similarity_minus_cnn_auroc",
    "similarity_minus_cnn_auprc",
    "similarity_minus_full_logistic_auroc",
    "similarity_minus_full_logistic_auprc",
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


def _write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _resolve(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def _relative(project_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(project_dir.resolve()).as_posix()


def _fixed_test_rows(
    positive_rows: Sequence[Mapping[str, str]],
    negative_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for source, source_rows, identifier in (
        ("positive", positive_rows, "record_id"),
        ("putative_negative", negative_rows, "negative_id"),
    ):
        for row in source_rows:
            rows.append(
                {
                    "sample_id": row[identifier],
                    "source": source,
                    "tier": "development",
                    "strategy": "distance_aware",
                    "negative_strategy": "DEGREE_MATCHED",
                    "split": "test",
                    "pmhc_id": row["pmhc_id"],
                    "tcr_cdr3_beta": row["tcr_cdr3_beta"],
                    "epitope": row["epitope"],
                    "hla": row["hla"],
                    "label": int(row["label"]),
                }
            )
    return rows


def _verified_summary_predictions(
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


def _assert_input_audit_matches(
    expected: Mapping[str, Any], observed: Mapping[str, Any], description: str
) -> None:
    if (
        observed["path"] != expected["path"]
        or observed["sha256"] != expected["sha256"]
        or int(observed["rows"]) != int(expected["rows"])
    ):
        raise ValueError(f"input manifest mismatch for {description}")


def _fold_statistics(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statistics = {}
    for metric in METRIC_NAMES:
        values = np.asarray(
            [float(row["similarity"][metric]) for row in fold_results],
            dtype=np.float64,
        )
        statistics[metric] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return statistics


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Degree-matched five-fold sequence-similarity baseline",
        "",
        "The existing normalized-Levenshtein algorithm and frozen top-5 parameters were reused without modification.",
        "Each fold uses only same-HLA train-positive references and preserves the existing test rows, labels, and order.",
        "",
        "## Coverage",
        "",
        f"- Evaluable queries: {summary['coverage']['available_query_count']} / "
        f"{summary['coverage']['query_count']} "
        f"({summary['coverage']['query_coverage']:.2%})",
        f"- Unavailable queries: {summary['coverage']['unavailable_query_count']}",
        "- No cross-HLA fallback was used.",
        "",
        "## Fold results",
        "",
        "| Fold | Available queries | Pooled AUROC | Pooled AUPRC | Macro AUROC | Macro AUPRC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in summary["fold_results"]:
        metrics = result["similarity"]
        lines.append(
            f"| {result['fold_id']} | {result['available_query_count']}/"
            f"{result['query_count']} | {metrics['pooled_auroc']:.6f} | "
            f"{metrics['pooled_auprc']:.6f} | {metrics['macro_auroc']:.6f} | "
            f"{metrics['macro_auprc']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Five-fold standard deviations use ddof=1.",
            "",
            "## OOF common-query comparisons",
            "",
            "| Metric | Similarity | CNN | Full Logistic | Similarity - CNN | Similarity - Full Logistic |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in METRIC_NAMES:
        lines.append(
            f"| {metric} | {summary['oof']['similarity'][metric]:.6f} | "
            f"{summary['oof']['cnn_common'][metric]:.6f} | "
            f"{summary['oof']['full_logistic_common'][metric]:.6f} | "
            f"{summary['oof']['similarity_minus_cnn'][metric]:+.6f} | "
            f"{summary['oof']['similarity_minus_full_logistic'][metric]:+.6f} |"
        )
    lines.extend(
        [
            "",
            "No model training, source-data resampling, label changes, or source-result modification was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_degree_matched_five_fold_similarity(
    config_path: Path, project_dir: Path
) -> dict[str, Any]:
    config = _load_yaml(config_path.resolve())
    if config.get("schema_version") != 1:
        raise ValueError("similarity config schema version mismatch")
    similarity_config = config["similarity"]
    expected_algorithm = {
        "top_k_epitopes": 5,
        "definition": "1 - levenshtein_distance / max_sequence_length",
        "candidate_chunk_size": 512,
        "workers": -1,
        "same_hla_only": True,
        "cross_hla_fallback": False,
    }
    if similarity_config != expected_algorithm:
        raise ValueError("similarity algorithm parameters differ from frozen baseline")

    fold_ids = tuple(config["fold_ids"])
    manifest_path = _resolve(project_dir, config["inputs"]["manifest"])
    manifest = _load_json(manifest_path)
    manifest_sha = file_sha256(manifest_path)
    rotation = validate_query_test_rotation_from_inputs(
        project_dir,
        manifest,
        fold_ids,
        positive_manifest_key="positive_input",
    )

    cnn_summary_path = _resolve(project_dir, config["inputs"]["cnn_summary"])
    logistic_summary_path = _resolve(
        project_dir, config["inputs"]["full_logistic_summary"]
    )
    cnn_summary = _load_json(cnn_summary_path)
    logistic_summary = _load_json(logistic_summary_path)
    for method, summary in (("CNN", cnn_summary), ("full Logistic", logistic_summary)):
        if summary["audit"]["manifest"]["sha256"] != manifest_sha:
            raise ValueError(f"{method} summary uses a different data manifest")
    cnn_oof_path, cnn_rows, cnn_oof_audit = _verified_summary_predictions(
        project_dir,
        cnn_summary,
        "oof_predictions_csv",
        "CNN OOF predictions",
    )
    logistic_oof_path, logistic_rows, logistic_oof_audit = (
        _verified_summary_predictions(
            project_dir,
            logistic_summary,
            "oof_predictions",
            "full Logistic OOF predictions",
        )
    )
    if len(cnn_rows) != len(logistic_rows):
        raise ValueError("CNN and full Logistic OOF row counts differ")
    cnn_by_fold = {
        fold_id: [row for row in cnn_rows if row["fold_id"] == fold_id]
        for fold_id in fold_ids
    }
    logistic_by_fold = {
        fold_id: [row for row in logistic_rows if row["fold_id"] == fold_id]
        for fold_id in fold_ids
    }

    original_config_path = _resolve(
        project_dir, config["inputs"]["original_similarity_config"]
    )
    original_config = _load_yaml(original_config_path)
    for key in ("top_k_epitopes", "definition", "candidate_chunk_size", "workers"):
        if similarity_config[key] != original_config["similarity"][key]:
            raise ValueError(f"similarity parameter {key} differs from original baseline")

    fold_results = []
    fold_audits = {}
    all_predictions = []
    all_per_query = []
    for fold_id in fold_ids:
        partitions = manifest["folds"][fold_id]["partitions"]
        train_path, train_rows, train_audit = _verified_rows(
            project_dir,
            partitions["train"]["positive_input"],
            f"{fold_id} train positives",
        )
        test_positive_path, test_positive_rows, test_positive_audit = _verified_rows(
            project_dir,
            partitions["test"]["positive_input"],
            f"{fold_id} test positives",
        )
        test_negative_path, test_negative_rows, test_negative_audit = _verified_rows(
            project_dir,
            partitions["test"]["negative_output"],
            f"{fold_id} test negatives",
        )
        if any(row.get("label") != "1" for row in train_rows + test_positive_rows):
            raise ValueError(f"positive label mismatch for {fold_id}")
        if any(row.get("label") != "0" for row in test_negative_rows):
            raise ValueError(f"negative label mismatch for {fold_id}")

        for kind, expected, cnn_observed, logistic_observed in (
            (
                "train positive",
                partitions["train"]["positive_input"],
                cnn_summary["audit"]["fold_inputs"][fold_id]["partitions"][
                    "train"
                ]["positive"],
                logistic_summary["audit"]["inputs"][fold_id]["train"]["positive"],
            ),
            (
                "test positive",
                partitions["test"]["positive_input"],
                cnn_summary["audit"]["fold_inputs"][fold_id]["partitions"]["test"][
                    "positive"
                ],
                logistic_summary["audit"]["inputs"][fold_id]["test"]["positive"],
            ),
            (
                "test negative",
                partitions["test"]["negative_output"],
                cnn_summary["audit"]["fold_inputs"][fold_id]["partitions"]["test"][
                    "negative"
                ],
                logistic_summary["audit"]["inputs"][fold_id]["test"]["negative"],
            ),
        ):
            _assert_input_audit_matches(
                expected, cnn_observed, f"{fold_id} CNN {kind}"
            )
            _assert_input_audit_matches(
                expected, logistic_observed, f"{fold_id} full Logistic {kind}"
            )

        test_rows = _fixed_test_rows(test_positive_rows, test_negative_rows)
        cnn_fold_rows = cnn_by_fold[fold_id]
        logistic_fold_rows = logistic_by_fold[fold_id]
        for expected, cnn_row, logistic_row in zip(
            test_rows, cnn_fold_rows, logistic_fold_rows, strict=True
        ):
            for field in ("sample_id", "source", "pmhc_id"):
                if str(expected[field]) != str(cnn_row[field]) or str(
                    expected[field]
                ) != str(logistic_row[field]):
                    raise ValueError(f"frozen test {field} mismatch for {fold_id}")
            if str(expected["label"]) != cnn_row["label"] or str(
                expected["label"]
            ) != logistic_row["label"]:
                raise ValueError(f"frozen test label mismatch for {fold_id}")
            for field in ("tcr_cdr3_beta", "epitope", "hla"):
                if expected[field] != logistic_row[field]:
                    raise ValueError(f"frozen test interaction mismatch for {fold_id}")

        scored_rows, query_metadata = score_fixed_test_rows(
            train_rows,
            test_rows,
            top_k_epitopes=int(similarity_config["top_k_epitopes"]),
            candidate_chunk_size=int(similarity_config["candidate_chunk_size"]),
            workers=int(similarity_config["workers"]),
        )
        for row, cnn_row, logistic_row in zip(
            scored_rows, cnn_fold_rows, logistic_fold_rows, strict=True
        ):
            row["fold_id"] = fold_id
            row["cnn_probability"] = float(cnn_row["probability"])
            row["full_logistic_probability"] = float(logistic_row["probability"])

        available_rows = [row for row in scored_rows if row["status"] == "available"]
        similarity_metrics, similarity_queries = _classification_metrics(
            available_rows, "similarity_score"
        )
        cnn_metrics, cnn_queries = _classification_metrics(
            available_rows, "cnn_probability"
        )
        logistic_metrics, logistic_queries = _classification_metrics(
            available_rows, "full_logistic_probability"
        )
        similarity_flat = _flatten_metrics(similarity_metrics)
        cnn_flat = _flatten_metrics(cnn_metrics)
        logistic_flat = _flatten_metrics(logistic_metrics)

        metadata_index = {row["query_id"]: row for row in query_metadata}
        expected_queries = set(rotation["fold_test_queries"][fold_id])
        if set(metadata_index) != expected_queries:
            raise ValueError(f"test query set mismatch for {fold_id}")
        for query_id in rotation["fold_test_queries"][fold_id]:
            metadata = metadata_index[query_id]
            query_rows = [row for row in scored_rows if row["pmhc_id"] == query_id]
            labels = [int(row["label"]) for row in query_rows]
            per_query = {
                "fold_id": fold_id,
                **metadata,
                "positive_count": sum(labels),
                "negative_count": len(labels) - sum(labels),
                "sample_count": len(labels),
            }
            if metadata["status"] == "available":
                for metric in ("auroc", "auprc"):
                    similarity_value = float(similarity_queries[query_id][metric])
                    cnn_value = float(cnn_queries[query_id][metric])
                    logistic_value = float(logistic_queries[query_id][metric])
                    per_query[f"similarity_{metric}"] = similarity_value
                    per_query[f"cnn_{metric}"] = cnn_value
                    per_query[f"full_logistic_{metric}"] = logistic_value
                    per_query[f"similarity_minus_cnn_{metric}"] = (
                        similarity_value - cnn_value
                    )
                    per_query[f"similarity_minus_full_logistic_{metric}"] = (
                        similarity_value - logistic_value
                    )
            else:
                for prefix in (
                    "similarity",
                    "cnn",
                    "full_logistic",
                    "similarity_minus_cnn",
                    "similarity_minus_full_logistic",
                ):
                    per_query.update(
                        {f"{prefix}_{metric}": "" for metric in ("auroc", "auprc")}
                    )
            all_per_query.append(per_query)

        available_query_count = len(similarity_queries)
        query_count = len(query_metadata)
        fold_results.append(
            {
                "fold_id": fold_id,
                "query_count": query_count,
                "available_query_count": available_query_count,
                "unavailable_query_count": query_count - available_query_count,
                "query_coverage": available_query_count / query_count,
                "available_sample_count": len(available_rows),
                "similarity": similarity_flat,
                "cnn_common": cnn_flat,
                "full_logistic_common": logistic_flat,
                "similarity_minus_cnn": _delta(similarity_flat, cnn_flat),
                "similarity_minus_full_logistic": _delta(
                    similarity_flat, logistic_flat
                ),
            }
        )
        all_predictions.extend(scored_rows)
        fold_audits[fold_id] = {
            "reference_partitions_loaded": ["train_positive"],
            "forbidden_reference_partitions_loaded": [],
            "train_positive": train_audit,
            "test_positive": test_positive_audit,
            "test_negative": test_negative_audit,
            "test_order_labels_and_interactions_match": True,
            "paths": {
                "train_positive": _relative(project_dir, train_path),
                "test_positive": _relative(project_dir, test_positive_path),
                "test_negative": _relative(project_dir, test_negative_path),
            },
        }

    if [row["sample_id"] for row in all_predictions] != [
        row["sample_id"] for row in cnn_rows
    ] or [row["sample_id"] for row in all_predictions] != [
        row["sample_id"] for row in logistic_rows
    ]:
        raise ValueError("OOF prediction order differs from frozen source results")
    if len({row["query_id"] for row in all_per_query}) != rotation["query_count"]:
        raise ValueError("OOF per-query rows are not unique and complete")

    available_oof = [row for row in all_predictions if row["status"] == "available"]
    similarity_oof, _ = _classification_metrics(available_oof, "similarity_score")
    cnn_oof, _ = _classification_metrics(available_oof, "cnn_probability")
    logistic_oof, _ = _classification_metrics(
        available_oof, "full_logistic_probability"
    )
    similarity_oof_flat = _flatten_metrics(similarity_oof)
    cnn_oof_flat = _flatten_metrics(cnn_oof)
    logistic_oof_flat = _flatten_metrics(logistic_oof)
    available_query_ids = sorted(
        row["query_id"] for row in all_per_query if row["status"] == "available"
    )
    coverage = {
        "query_count": rotation["query_count"],
        "available_query_count": len(available_query_ids),
        "unavailable_query_count": rotation["query_count"] - len(available_query_ids),
        "query_coverage": len(available_query_ids) / rotation["query_count"],
        "available_query_ids": available_query_ids,
        "unavailable_query_ids": sorted(
            row["query_id"] for row in all_per_query if row["status"] == "unavailable"
        ),
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "seed": int(config["seed"]),
        "experiment": config["experiment"],
        "algorithm": similarity_config,
        "coverage": coverage,
        "fold_results": fold_results,
        "five_fold_statistics": _fold_statistics(fold_results),
        "five_fold_statistics_definition": {
            "standard_deviation": "sample",
            "degrees_of_freedom": 1,
        },
        "oof": {
            "available_sample_count": len(available_oof),
            "similarity": similarity_oof_flat,
            "cnn_common": cnn_oof_flat,
            "full_logistic_common": logistic_oof_flat,
            "similarity_minus_cnn": _delta(similarity_oof_flat, cnn_oof_flat),
            "similarity_minus_full_logistic": _delta(
                similarity_oof_flat, logistic_oof_flat
            ),
        },
        "audit": {
            "status": "passed",
            "manifest": {
                "path": _relative(project_dir, manifest_path),
                "sha256": manifest_sha,
            },
            "query_rotation": rotation,
            "fold_inputs": fold_audits,
            "frozen_test": {
                "sample_count": len(all_predictions),
                "labels_order_and_interactions_unchanged": True,
                "resampling_performed": False,
            },
            "reference_policy": {
                "allowed_partition": "train_positive",
                "validation_or_test_reference_rows_used": False,
                "same_hla_only": True,
                "cross_hla_fallback": False,
            },
            "comparisons": {
                "common_evaluable_queries_only": True,
                "cnn_summary_path": _relative(project_dir, cnn_summary_path),
                "cnn_summary_sha256": file_sha256(cnn_summary_path),
                "cnn_oof": cnn_oof_audit,
                "full_logistic_summary_path": _relative(
                    project_dir, logistic_summary_path
                ),
                "full_logistic_summary_sha256": file_sha256(logistic_summary_path),
                "full_logistic_oof": logistic_oof_audit,
            },
            "algorithm_reuse": {
                "original_config_path": _relative(project_dir, original_config_path),
                "original_config_sha256": file_sha256(original_config_path),
                "source_path": config["inputs"]["algorithm_source"],
                "source_sha256": file_sha256(
                    _resolve(project_dir, config["inputs"]["algorithm_source"])
                ),
            },
            "training_performed": False,
            "source_results_modified": False,
        },
    }

    output_directory = _resolve(project_dir, config["outputs"]["directory"])
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    prediction_path = output_directory / config["outputs"]["predictions_csv"]
    per_query_path = output_directory / config["outputs"]["per_query_csv"]
    summary_path = output_directory / config["outputs"]["summary_json"]
    report_path = output_directory / config["outputs"]["report_md"]
    _write_csv(prediction_path, PREDICTION_COLUMNS, all_predictions)
    _write_csv(per_query_path, PER_QUERY_COLUMNS, all_per_query)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    summary["outputs"] = {
        "predictions_csv": {
            "path": _relative(project_dir, prediction_path),
            "sha256": file_sha256(prediction_path),
        },
        "per_query_csv": {
            "path": _relative(project_dir, per_query_path),
            "sha256": file_sha256(per_query_path),
        },
        "report_md": {
            "path": _relative(project_dir, report_path),
            "sha256": file_sha256(report_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
