from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .five_fold_summary import file_sha256, validate_query_test_rotation
from .metrics import binary_classification_metrics, query_level_classification_metrics
from .retrieval_similarity import (
    build_training_references,
    max_tcr_similarities,
    score_similarity_query,
    select_training_epitopes,
)


PARTITIONS_USED = ("train_positive", "test_positive", "test_negative")
METRIC_NAMES = ("pooled_auroc", "pooled_auprc", "macro_auroc", "macro_auprc")
OOF_COLUMNS = (
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
    "delta_auroc",
    "delta_auprc",
)
SUMMARY_COLUMNS = (
    "record_type",
    "fold_id",
    "query_count",
    "available_query_count",
    "query_coverage",
    "available_sample_count",
    "similarity_pooled_auroc",
    "similarity_pooled_auprc",
    "similarity_macro_auroc",
    "similarity_macro_auprc",
    "cnn_pooled_auroc",
    "cnn_pooled_auprc",
    "cnn_macro_auroc",
    "cnn_macro_auprc",
    "delta_pooled_auroc",
    "delta_pooled_auprc",
    "delta_macro_auroc",
    "delta_macro_auprc",
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


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
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


def _verified_rows(
    project_dir: Path, source: Mapping[str, Any], description: str
) -> tuple[Path, list[dict[str, str]], dict[str, Any]]:
    path = _resolve(project_dir, str(source["path"]))
    observed_sha = file_sha256(path)
    if observed_sha != source["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {description}")
    _, rows = _read_csv(path)
    if len(rows) != int(source["rows"]):
        raise ValueError(f"row count mismatch for {description}")
    return path, rows, {
        "path": _relative(project_dir, path),
        "sha256": observed_sha,
        "rows": len(rows),
    }


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
                    "negative_strategy": "N1",
                    "split": "test",
                    "pmhc_id": row["pmhc_id"],
                    "tcr_cdr3_beta": row["tcr_cdr3_beta"],
                    "epitope": row["epitope"],
                    "hla": row["hla"],
                    "label": int(row["label"]),
                }
            )
    return rows


def score_fixed_test_rows(
    train_positive_rows: Sequence[Mapping[str, str]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    top_k_epitopes: int,
    candidate_chunk_size: int,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    references = build_training_references(train_positive_rows)
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    query_order = []
    for index, row in enumerate(test_rows):
        query_id = str(row["pmhc_id"]).strip().upper()
        if query_id not in grouped_indices:
            query_order.append(query_id)
        grouped_indices[query_id].append(index)

    scored_rows = [dict(row) for row in test_rows]
    query_metadata = []
    for query_id in query_order:
        indices = grouped_indices[query_id]
        query_rows = [test_rows[index] for index in indices]
        epitope = str(query_rows[0]["epitope"]).strip().upper()
        hla = str(query_rows[0]["hla"]).strip().upper()
        if any(
            str(row["epitope"]).strip().upper() != epitope
            or str(row["hla"]).strip().upper() != hla
            for row in query_rows
        ):
            raise ValueError(f"inconsistent pMHC fields for {query_id}")
        hla_references = references.get(hla)
        selected = (
            select_training_epitopes(epitope, hla_references, top_k_epitopes)
            if hla_references
            else []
        )
        metadata = {
            "query_id": query_id,
            "epitope": epitope,
            "hla": hla,
            "status": "available" if selected else "unavailable",
            "selected_training_epitopes": json.dumps(
                [
                    {"epitope": selected_epitope, "similarity": similarity}
                    for selected_epitope, similarity in selected
                ],
                separators=(",", ":"),
                sort_keys=True,
            ),
            "training_reference_tcr_count": (
                sum(len(hla_references[selected_epitope]) for selected_epitope, _ in selected)
                if hla_references
                else 0
            ),
        }
        query_metadata.append(metadata)
        if not selected:
            for index in indices:
                scored_rows[index]["status"] = "unavailable"
                scored_rows[index]["similarity_score"] = ""
            continue

        candidate_tcrs = [str(row["tcr_cdr3_beta"]).strip().upper() for row in query_rows]
        cache = {
            selected_epitope: max_tcr_similarities(
                candidate_tcrs,
                hla_references[selected_epitope],
                candidate_chunk_size=candidate_chunk_size,
                workers=workers,
            )
            for selected_epitope, _ in selected
        }
        scores = score_similarity_query(
            candidate_tcrs, selected, hla_references, cache
        )
        for index, score in zip(indices, scores, strict=True):
            scored_rows[index]["status"] = "available"
            scored_rows[index]["similarity_score"] = score
    return scored_rows, query_metadata


def _classification_metrics(
    rows: Sequence[Mapping[str, Any]], score_field: str
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    if not rows:
        raise ValueError("classification metrics require evaluable rows")
    predictions = [
        {
            "pmhc_id": row["pmhc_id"],
            "label": int(row["label"]),
            "probability": float(row[score_field]),
        }
        for row in rows
    ]
    pooled = binary_classification_metrics(
        [row["label"] for row in predictions],
        [row["probability"] for row in predictions],
    )
    query_macro, per_query = query_level_classification_metrics(predictions)
    return (
        {
            "pooled": pooled,
            "query_macro": {
                "auroc": float(query_macro["auroc"]),
                "auprc": float(query_macro["auprc"]),
                "query_count": int(query_macro["query_count"]),
            },
        },
        {str(row["query_id"]): row for row in per_query},
    )


def _flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        "pooled_auroc": float(metrics["pooled"]["auroc"]),
        "pooled_auprc": float(metrics["pooled"]["auprc"]),
        "macro_auroc": float(metrics["query_macro"]["auroc"]),
        "macro_auprc": float(metrics["query_macro"]["auprc"]),
    }


def _delta(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    return {metric: float(left[metric]) - float(right[metric]) for metric in METRIC_NAMES}


def _fold_statistics(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statistics = {}
    for metric in METRIC_NAMES:
        values = np.asarray(
            [float(row["similarity"][metric]) for row in fold_results], dtype=np.float64
        )
        statistics[metric] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return statistics


def _summary_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in summary["fold_results"]:
        row = {
            "record_type": "fold",
            "fold_id": result["fold_id"],
            "query_count": result["query_count"],
            "available_query_count": result["available_query_count"],
            "query_coverage": result["query_coverage"],
            "available_sample_count": result["available_sample_count"],
        }
        for prefix in ("similarity", "cnn_common", "delta"):
            csv_prefix = "cnn" if prefix == "cnn_common" else prefix
            row.update(
                {f"{csv_prefix}_{metric}": value for metric, value in result[prefix].items()}
            )
        rows.append(row)
    for record_type, key in (
        ("five_fold_mean", "mean"),
        ("five_fold_standard_deviation", "standard_deviation"),
        ("five_fold_minimum", "minimum"),
        ("five_fold_maximum", "maximum"),
    ):
        rows.append(
            {
                "record_type": record_type,
                **{
                    f"similarity_{metric}": summary["five_fold_statistics"][metric][key]
                    for metric in METRIC_NAMES
                },
            }
        )
    oof = summary["oof"]
    row = {
        "record_type": "oof",
        "query_count": summary["coverage"]["query_count"],
        "available_query_count": summary["coverage"]["available_query_count"],
        "query_coverage": summary["coverage"]["query_coverage"],
        "available_sample_count": oof["available_sample_count"],
    }
    for prefix in ("similarity", "cnn_common", "delta"):
        csv_prefix = "cnn" if prefix == "cnn_common" else prefix
        row.update({f"{csv_prefix}_{metric}": value for metric, value in oof[prefix].items()})
    rows.append(row)
    return rows


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Distance-aware 5-fold query-balanced sequence-similarity baseline",
        "",
        "The existing normalized-Levenshtein sequence-similarity algorithm was reused without modification.",
        "Each fold uses only that fold's train positives as same-HLA references and scores the frozen cap=100 test rows in their original order.",
        "",
        "## Coverage",
        "",
        f"- Evaluable queries: {summary['coverage']['available_query_count']} / {summary['coverage']['query_count']} "
        f"({summary['coverage']['query_coverage']:.2%})",
        f"- Unavailable queries: {summary['coverage']['unavailable_query_count']}",
        "- Queries without same-HLA train references are unavailable; no cross-HLA fallback is used.",
        "",
        "## Fold results",
        "",
        "| Fold | Available queries | Pooled AUROC | Pooled AUPRC | Macro AUROC | Macro AUPRC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in summary["fold_results"]:
        metric = result["similarity"]
        lines.append(
            f"| {result['fold_id']} | {result['available_query_count']}/{result['query_count']} | "
            f"{metric['pooled_auroc']:.6f} | {metric['pooled_auprc']:.6f} | "
            f"{metric['macro_auroc']:.6f} | {metric['macro_auprc']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Five-fold standard deviations are sample standard deviations (ddof=1).",
            "",
            "## OOF comparison on common evaluable queries",
            "",
            "| Metric | Similarity | CNN | Similarity - CNN |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in METRIC_NAMES:
        lines.append(
            f"| {metric} | {summary['oof']['similarity'][metric]:.6f} | "
            f"{summary['oof']['cnn_common'][metric]:.6f} | "
            f"{summary['oof']['delta'][metric]:+.6f} |"
        )
    lines.extend(
        [
            "",
            "No training, resampling, label changes, or CNN result modifications were performed.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_query_balanced_five_fold_similarity(
    config_path: Path, project_dir: Path, *, force: bool = False
) -> dict[str, Any]:
    config = _load_yaml(config_path)
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
        raise ValueError("similarity algorithm parameters differ from the frozen baseline")

    fold_ids = tuple(config["fold_ids"])
    manifest_path = _resolve(project_dir, config["inputs"]["manifest"])
    manifest = _load_json(manifest_path)
    manifest_sha = file_sha256(manifest_path)
    rotation = validate_query_test_rotation(manifest, fold_ids)

    cnn_summary_path = _resolve(project_dir, config["inputs"]["cnn_summary"])
    cnn_summary = _load_json(cnn_summary_path)
    if cnn_summary["audit"]["manifest"]["sha256"] != manifest_sha:
        raise ValueError("CNN summary uses a different data manifest")
    cnn_oof_source = cnn_summary["outputs"]["oof_predictions_csv"]
    cnn_oof_path = _resolve(project_dir, cnn_oof_source["path"])
    if file_sha256(cnn_oof_path) != cnn_oof_source["sha256"]:
        raise ValueError("CNN OOF prediction SHA-256 mismatch")
    _, cnn_oof_rows = _read_csv(cnn_oof_path)
    cnn_by_fold = {
        fold_id: [row for row in cnn_oof_rows if row["fold_id"] == fold_id]
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
        fold_manifest = manifest["folds"][fold_id]["partitions"]
        train_path, train_rows, train_audit = _verified_rows(
            project_dir, fold_manifest["train"]["positive_output"], f"{fold_id} train positives"
        )
        test_positive_path, test_positive_rows, test_positive_audit = _verified_rows(
            project_dir,
            fold_manifest["test"]["positive_output"],
            f"{fold_id} test positives",
        )
        test_negative_path, test_negative_rows, test_negative_audit = _verified_rows(
            project_dir,
            fold_manifest["test"]["negative_output"],
            f"{fold_id} test negatives",
        )
        if any(row.get("label") != "1" for row in train_rows + test_positive_rows):
            raise ValueError(f"positive label mismatch for {fold_id}")
        if any(row.get("label") != "0" for row in test_negative_rows):
            raise ValueError(f"negative label mismatch for {fold_id}")
        test_rows = _fixed_test_rows(test_positive_rows, test_negative_rows)
        cnn_rows = cnn_by_fold[fold_id]
        if [row["sample_id"] for row in test_rows] != [row["sample_id"] for row in cnn_rows]:
            raise ValueError(f"frozen test sample order mismatch for {fold_id}")
        if [str(row["label"]) for row in test_rows] != [row["label"] for row in cnn_rows]:
            raise ValueError(f"frozen test labels mismatch for {fold_id}")
        if [row["pmhc_id"] for row in test_rows] != [row["pmhc_id"] for row in cnn_rows]:
            raise ValueError(f"frozen test query order mismatch for {fold_id}")

        scored_rows, query_metadata = score_fixed_test_rows(
            train_rows,
            test_rows,
            top_k_epitopes=int(similarity_config["top_k_epitopes"]),
            candidate_chunk_size=int(similarity_config["candidate_chunk_size"]),
            workers=int(similarity_config["workers"]),
        )
        for row, cnn_row in zip(scored_rows, cnn_rows, strict=True):
            row["fold_id"] = fold_id
            row["cnn_probability"] = float(cnn_row["probability"])
        available_rows = [row for row in scored_rows if row["status"] == "available"]
        similarity_metrics, similarity_per_query = _classification_metrics(
            available_rows, "similarity_score"
        )
        cnn_metrics, cnn_per_query = _classification_metrics(
            available_rows, "cnn_probability"
        )
        similarity_flat = _flatten_metrics(similarity_metrics)
        cnn_flat = _flatten_metrics(cnn_metrics)
        metadata_index = {row["query_id"]: row for row in query_metadata}
        expected_test_queries = set(rotation["fold_test_queries"][fold_id])
        if set(metadata_index) != expected_test_queries:
            raise ValueError(f"test query set mismatch for {fold_id}")
        for query_id in rotation["fold_test_queries"][fold_id]:
            metadata = metadata_index[query_id]
            query_rows = [row for row in scored_rows if row["pmhc_id"] == query_id]
            labels = [int(row["label"]) for row in query_rows]
            per_query_row = {
                "fold_id": fold_id,
                **metadata,
                "positive_count": sum(labels),
                "negative_count": len(labels) - sum(labels),
                "sample_count": len(labels),
            }
            if metadata["status"] == "available":
                similarity_row = similarity_per_query[query_id]
                cnn_row = cnn_per_query[query_id]
                for metric in ("auroc", "auprc"):
                    similarity_value = float(similarity_row[metric])
                    cnn_value = float(cnn_row[metric])
                    per_query_row[f"similarity_{metric}"] = similarity_value
                    per_query_row[f"cnn_{metric}"] = cnn_value
                    per_query_row[f"delta_{metric}"] = similarity_value - cnn_value
            else:
                for prefix in ("similarity", "cnn", "delta"):
                    per_query_row.update(
                        {f"{prefix}_{metric}": "" for metric in ("auroc", "auprc")}
                    )
            all_per_query.append(per_query_row)

        available_query_count = len(similarity_per_query)
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
                "delta": _delta(similarity_flat, cnn_flat),
            }
        )
        all_predictions.extend(scored_rows)
        fold_audits[fold_id] = {
            "reference_partitions_loaded": ["train_positive"],
            "forbidden_reference_partitions_loaded": [],
            "train_positive": train_audit,
            "test_positive": test_positive_audit,
            "test_negative": test_negative_audit,
            "test_order_and_labels_match_cnn": True,
            "paths": {
                "train_positive": _relative(project_dir, train_path),
                "test_positive": _relative(project_dir, test_positive_path),
                "test_negative": _relative(project_dir, test_negative_path),
            },
        }

    if [row["sample_id"] for row in all_predictions] != [
        row["sample_id"] for row in cnn_oof_rows
    ]:
        raise ValueError("OOF prediction order differs from frozen CNN OOF order")
    if len({row["query_id"] for row in all_per_query}) != rotation["query_count"]:
        raise ValueError("OOF per-query rows are not unique and complete")
    available_oof = [row for row in all_predictions if row["status"] == "available"]
    similarity_oof, _ = _classification_metrics(available_oof, "similarity_score")
    cnn_oof, _ = _classification_metrics(available_oof, "cnn_probability")
    similarity_oof_flat = _flatten_metrics(similarity_oof)
    cnn_oof_flat = _flatten_metrics(cnn_oof)
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
            "delta": _delta(similarity_oof_flat, cnn_oof_flat),
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
                "labels_and_order_unchanged": True,
                "resampling_performed": False,
            },
            "reference_policy": {
                "allowed_partition": "train_positive",
                "validation_or_test_reference_rows_used": False,
                "same_hla_only": True,
                "cross_hla_fallback": False,
            },
            "cnn_comparison": {
                "summary_path": _relative(project_dir, cnn_summary_path),
                "summary_sha256": file_sha256(cnn_summary_path),
                "oof_predictions_path": _relative(project_dir, cnn_oof_path),
                "oof_predictions_sha256": file_sha256(cnn_oof_path),
                "common_evaluable_queries_only": True,
            },
            "algorithm_reuse": {
                "original_config_path": _relative(project_dir, original_config_path),
                "original_config_sha256": file_sha256(original_config_path),
                "source_path": config["inputs"]["algorithm_source"],
                "source_sha256": file_sha256(
                    _resolve(project_dir, config["inputs"]["algorithm_source"])
                ),
            },
        },
    }

    output_directory = _resolve(project_dir, config["outputs"]["directory"])
    output_paths = {
        name: output_directory / config["outputs"][name]
        for name in (
            "summary_csv",
            "summary_json",
            "oof_predictions_csv",
            "per_query_csv",
            "report_md",
        )
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(f"similarity outputs already exist: {existing}")
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_paths["summary_csv"], SUMMARY_COLUMNS, _summary_rows(summary))
    _write_csv(output_paths["oof_predictions_csv"], OOF_COLUMNS, all_predictions)
    _write_csv(output_paths["per_query_csv"], PER_QUERY_COLUMNS, all_per_query)
    output_paths["report_md"].write_text(_render_report(summary), encoding="utf-8")
    summary["outputs"] = {
        name: {
            "path": _relative(project_dir, path),
            "sha256": file_sha256(path),
        }
        for name, path in output_paths.items()
        if name != "summary_json"
    }
    output_paths["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
