from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .five_fold_summary import file_sha256
from .kmer_logistic import (
    METRIC_NAMES,
    PARTITIONS,
    PER_QUERY_COLUMNS,
    PREDICTION_COLUMNS,
    _load_json,
    _load_yaml,
    _metrics,
    _read_csv,
    _relative,
    _resolve,
    _write_csv,
)
from .tcr_only_logistic import (
    TCR_ONLY_FEATURE_COUNT,
    build_tcr_only_feature_matrix,
)
from .tcrue.degree_matched_negatives import audit_degree_matched_partition
from .tcrue.negative_sampling import load_positive_registry


PairKey = tuple[str, str, str]
QueryKey = tuple[str, str]


def _normalize(value: Any) -> str:
    return str(value).strip().upper()


def _query_key(row: Mapping[str, Any]) -> QueryKey:
    return _normalize(row["epitope"]), _normalize(row["hla"])


def _fixed_rows(
    positive_rows: Sequence[Mapping[str, str]],
    negative_rows: Sequence[Mapping[str, str]],
    *,
    fold_id: str,
    partition: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, source_rows, identifier, expected_label in (
        ("positive", positive_rows, "record_id", 1),
        ("putative_negative", negative_rows, "negative_id", 0),
    ):
        for row in source_rows:
            if int(row["label"]) != expected_label:
                raise ValueError(f"invalid {source} label in {fold_id}/{partition}")
            if source == "putative_negative":
                if row.get("negative_strategy") != "degree_matched":
                    raise ValueError(
                        f"unexpected negative strategy in {fold_id}/{partition}"
                    )
                if row.get("fold_id") != fold_id or row.get("partition") != partition:
                    raise ValueError(
                        f"negative fold/partition metadata mismatch: {fold_id}/{partition}"
                    )
            rows.append(
                {
                    "sample_id": row[identifier],
                    "source": source,
                    "tier": "development",
                    "strategy": "distance_aware",
                    "negative_strategy": "degree_matched",
                    "split": partition,
                    "pmhc_id": row["pmhc_id"],
                    "tcr_cdr3_beta": row["tcr_cdr3_beta"],
                    "epitope": row["epitope"],
                    "hla": row["hla"],
                    "label": expected_label,
                }
            )
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate sample ID in {fold_id}/{partition}")
    return rows


def audit_partition_degrees(
    positive_rows: Sequence[Mapping[str, Any]],
    negative_rows: Sequence[Mapping[str, Any]],
    positive_registry: set[PairKey],
) -> dict[str, Any]:
    base = audit_degree_matched_partition(
        positive_rows, negative_rows, positive_registry
    )
    positive_tcr = Counter(_normalize(row["tcr_cdr3_beta"]) for row in positive_rows)
    negative_tcr = Counter(_normalize(row["tcr_cdr3_beta"]) for row in negative_rows)
    positive_query = Counter(_query_key(row) for row in positive_rows)
    negative_query = Counter(_query_key(row) for row in negative_rows)
    return {
        **base,
        "unique_tcr_count": len(positive_tcr),
        "unique_query_count": len(positive_query),
        "tcr_degree_exact": positive_tcr == negative_tcr,
        "query_degree_exact": positive_query == negative_query,
        "maximum_tcr_degree": max(positive_tcr.values()),
        "maximum_query_degree": max(positive_query.values()),
    }


def _validate_query_rotation(
    rows_by_fold: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    fold_ids: Sequence[str],
) -> dict[str, Any]:
    reference_universe: set[QueryKey] | None = None
    test_counts: Counter[QueryKey] = Counter()
    fold_test_counts: dict[str, int] = {}
    for fold_id in fold_ids:
        partition_queries = {
            partition: {
                _query_key(row)
                for row in rows_by_fold[fold_id][partition]
                if row["source"] == "positive"
            }
            for partition in PARTITIONS
        }
        for index, left in enumerate(PARTITIONS):
            for right in PARTITIONS[index + 1 :]:
                if partition_queries[left] & partition_queries[right]:
                    raise ValueError(f"query leakage in {fold_id}: {left} vs {right}")
        universe = set().union(*partition_queries.values())
        if reference_universe is None:
            reference_universe = universe
        elif universe != reference_universe:
            raise ValueError(f"query universe mismatch in {fold_id}")
        test_counts.update(partition_queries["test"])
        fold_test_counts[fold_id] = len(partition_queries["test"])
    if reference_universe is None or not reference_universe:
        raise ValueError("query universe is empty")
    if set(test_counts) != reference_universe or set(test_counts.values()) != {1}:
        raise ValueError("each query must appear as test exactly once")
    return {
        "query_count": len(reference_universe),
        "all_queries_tested_exactly_once": True,
        "test_occurrence_minimum": min(test_counts.values()),
        "test_occurrence_maximum": max(test_counts.values()),
        "fold_test_query_counts": fold_test_counts,
    }


def _load_and_audit_inputs(
    project_dir: Path,
    manifest: Mapping[str, Any],
    fold_ids: Sequence[str],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, list[dict[str, Any]]]],
    dict[str, Any],
]:
    if manifest.get("invariants", {}).get(
        "all_fold_partitions_have_exact_solution"
    ) is not True:
        raise ValueError("degree-matched manifest does not certify exact solutions")
    registry_source = manifest["inputs"]["positive_registry"]
    registry_path = _resolve(project_dir, registry_source["path"])
    registry_sha = file_sha256(registry_path)
    if registry_sha != registry_source["sha256"]:
        raise ValueError("positive registry SHA-256 mismatch")
    positive_registry = {
        (_normalize(tcr), _normalize(epitope), _normalize(hla))
        for tcr, epitope, hla in load_positive_registry(registry_path)
    }

    input_audit: dict[str, Any] = {}
    rows_by_fold: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for fold_id in fold_ids:
        input_audit[fold_id] = {}
        rows_by_fold[fold_id] = {}
        for partition in PARTITIONS:
            entry = manifest["folds"][fold_id]["partitions"][partition]
            if entry.get("exact_solution") is not True:
                raise ValueError(f"non-exact manifest entry: {fold_id}/{partition}")
            kind_rows: dict[str, list[dict[str, str]]] = {}
            partition_audit: dict[str, Any] = {}
            for kind, source_key in (
                ("positive", "positive_input"),
                ("negative", "negative_output"),
            ):
                source = entry[source_key]
                path = _resolve(project_dir, source["path"])
                observed_sha = file_sha256(path)
                if observed_sha != source["sha256"]:
                    raise ValueError(
                        f"SHA-256 mismatch for {fold_id}/{partition}/{kind}"
                    )
                rows = _read_csv(path)
                if len(rows) != int(source["rows"]):
                    raise ValueError(
                        f"row count mismatch for {fold_id}/{partition}/{kind}"
                    )
                kind_rows[kind] = rows
                partition_audit[kind] = {
                    "path": _relative(project_dir, path),
                    "sha256": observed_sha,
                    "rows": len(rows),
                }
            degree_audit = audit_partition_degrees(
                kind_rows["positive"], kind_rows["negative"], positive_registry
            )
            if not degree_audit["tcr_degree_exact"] or not degree_audit[
                "query_degree_exact"
            ]:
                raise ValueError(f"degree mismatch in {fold_id}/{partition}")
            rows_by_fold[fold_id][partition] = _fixed_rows(
                kind_rows["positive"],
                kind_rows["negative"],
                fold_id=fold_id,
                partition=partition,
            )
            input_audit[fold_id][partition] = {
                **partition_audit,
                "degree_audit": degree_audit,
            }
    return (
        input_audit,
        rows_by_fold,
        {
            "path": _relative(project_dir, registry_path),
            "sha256": registry_sha,
            "interaction_count": len(positive_registry),
        },
    )


def _assert_close(observed: float, expected: float, description: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{description} mismatch: {observed} != {expected}")


def _verify_fixed_specification(
    project_dir: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_source = config["references"]["original_tcr_only_config"]
    reference_config_path = _resolve(project_dir, config_source["path"])
    if file_sha256(reference_config_path) != config_source["sha256"]:
        raise ValueError("original TCR-only config SHA-256 mismatch")
    reference_config = _load_yaml(reference_config_path)
    for field in ("seed", "features", "model"):
        if config[field] != reference_config[field]:
            raise ValueError(f"degree-matched TCR-only {field} differs from original")

    summary_source = config["references"]["original_tcr_only_summary"]
    reference_summary_path = _resolve(project_dir, summary_source["path"])
    if file_sha256(reference_summary_path) != summary_source["sha256"]:
        raise ValueError("original TCR-only summary SHA-256 mismatch")
    reference_summary = _load_json(reference_summary_path)
    if reference_summary["seed"] != config["seed"]:
        raise ValueError("original TCR-only summary seed mismatch")
    return reference_config, reference_summary


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Degree-matched five-fold TCR-only logistic regression",
        "",
        "This control uses the frozen exact degree-matched negatives without resampling.",
        "",
        "## Fixed specification",
        "",
        "- Features: normalized TCR 1-mer and 2-mer frequencies plus raw TCR length (421 dimensions).",
        "- Epitope, HLA, and TCR-epitope interaction features are forbidden.",
        "- Model and seed exactly match the original TCR-only Logistic control.",
        "- Each fold is fit on train only; validation and test do not participate in fitting.",
        "",
        "## Fold test results",
        "",
        "| Fold | Samples | Queries | Pooled AUROC | Pooled AUPRC | Macro AUROC | Macro AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fold_id, fold in summary["folds"].items():
        metrics = fold["test_metrics"]
        lines.append(
            f"| {fold_id} | {fold['test_sample_count']} | {fold['test_query_count']} | "
            f"{metrics['pooled_auroc']:.6f} | {metrics['pooled_auprc']:.6f} | "
            f"{metrics['macro_auroc']:.6f} | {metrics['macro_auprc']:.6f} |"
        )
    lines.extend(["", "## Five-fold and OOF summary", ""])
    for metric in METRIC_NAMES:
        item = summary["five_fold_test_statistics"][metric]
        lines.append(
            f"- {metric}: {item['mean']:.6f} +/- "
            f"{item['standard_deviation']:.6f}"
        )
    for metric, value in summary["oof_metrics"].items():
        lines.append(f"- OOF {metric}: {value:.6f}")
    lines.extend(["", "## Difference from original TCR-only N1", ""])
    differences = summary["oof_differences"][
        "degree_matched_minus_original_tcr_only_n1"
    ]
    for metric in METRIC_NAMES:
        lines.append(f"- {metric}: {differences[metric]:+.6f}")
    lines.extend(
        [
            "",
            "## Audit",
            "",
            "All 15 input pairs matched their manifest SHA-256 values. TCR and query positive/negative degrees were independently verified as exact before fitting.",
            "",
        ]
    )
    return "\n".join(lines)


def train_degree_matched_tcr_only_five_fold(
    project_dir: Path, config_path: Path
) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    fold_ids = [str(value) for value in config["fold_ids"]]
    reference_config, reference_summary = _verify_fixed_specification(
        project_dir, config
    )

    manifest_source = config["data"]["manifest"]
    manifest_path = _resolve(project_dir, manifest_source["path"])
    manifest_sha = file_sha256(manifest_path)
    if manifest_sha != manifest_source["sha256"]:
        raise ValueError("degree-matched manifest SHA-256 mismatch")
    manifest = _load_json(manifest_path)
    input_audit, rows_by_fold, registry_audit = _load_and_audit_inputs(
        project_dir, manifest, fold_ids
    )
    rotation_audit = _validate_query_rotation(rows_by_fold, fold_ids)

    output_directory = _resolve(project_dir, config["output"]["directory"])
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    fold_summaries: dict[str, Any] = {}
    oof_rows: list[dict[str, Any]] = []
    oof_query_rows: list[dict[str, Any]] = []
    for fold_id in fold_ids:
        fold_directory = output_directory / fold_id
        fold_directory.mkdir()
        train_rows = rows_by_fold[fold_id]["train"]
        test_rows = rows_by_fold[fold_id]["test"]
        train_features = build_tcr_only_feature_matrix(train_rows)
        test_features = build_tcr_only_feature_matrix(test_rows)
        train_labels = np.asarray(
            [row["label"] for row in train_rows], dtype=np.int64
        )
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight=None,
            max_iter=5000,
            random_state=seed,
            solver="liblinear",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train_features, train_labels)
        logits = np.asarray(model.decision_function(test_features), dtype=np.float64)
        probabilities = np.asarray(
            model.predict_proba(test_features)[:, 1], dtype=np.float64
        )
        scored_rows = [
            {
                **row,
                "fold_id": fold_id,
                "logit": float(logit),
                "probability": float(probability),
            }
            for row, logit, probability in zip(
                test_rows, logits, probabilities, strict=True
            )
        ]
        fold_metrics, per_query = _metrics(scored_rows)
        per_query = [{"fold_id": fold_id, **row} for row in per_query]

        prediction_path = fold_directory / "test_predictions.csv"
        per_query_path = fold_directory / "per_query_metrics.csv"
        model_path = fold_directory / "model.joblib"
        _write_csv(prediction_path, PREDICTION_COLUMNS, scored_rows)
        _write_csv(per_query_path, PER_QUERY_COLUMNS, per_query)
        joblib.dump(
            {
                "estimator": model,
                "feature_config": config["features"],
                "model_config": config["model"],
                "seed": seed,
                "fold_id": fold_id,
                "feature_count": TCR_ONLY_FEATURE_COUNT,
                "feature_fields_used": ["tcr_cdr3_beta"],
                "forbidden_feature_fields_used": [],
                "input_audit": input_audit[fold_id],
                "model_fit_partitions": ["train"],
            },
            model_path,
            compress=3,
        )
        fold_summaries[fold_id] = {
            "train_sample_count": len(train_rows),
            "validation_sample_count": len(rows_by_fold[fold_id]["validation"]),
            "test_sample_count": len(test_rows),
            "test_query_count": len(per_query),
            "feature_count": train_features.shape[1],
            "iterations": int(model.n_iter_[0]),
            "test_metrics": fold_metrics,
            "artifacts": {
                "model": {
                    "path": _relative(project_dir, model_path),
                    "sha256": file_sha256(model_path),
                },
                "test_predictions": {
                    "path": _relative(project_dir, prediction_path),
                    "sha256": file_sha256(prediction_path),
                },
                "per_query_metrics": {
                    "path": _relative(project_dir, per_query_path),
                    "sha256": file_sha256(per_query_path),
                },
            },
        }
        oof_rows.extend(scored_rows)
        oof_query_rows.extend(per_query)

    oof_metrics, recomputed_queries = _metrics(oof_rows)
    if len(recomputed_queries) != len(oof_query_rows):
        raise ValueError("OOF query IDs are not unique across folds")
    statistics: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        values = np.asarray(
            [
                fold_summaries[fold_id]["test_metrics"][metric]
                for fold_id in fold_ids
            ],
            dtype=np.float64,
        )
        statistics[metric] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }

    original_metrics = {
        metric: float(reference_summary["oof_metrics"][metric])
        for metric in METRIC_NAMES
    }
    differences = {
        metric: float(oof_metrics[metric]) - original_metrics[metric]
        for metric in METRIC_NAMES
    }
    oof_path = output_directory / "oof_test_predictions.csv"
    oof_query_path = output_directory / "oof_per_query_metrics.csv"
    _write_csv(oof_path, PREDICTION_COLUMNS, oof_rows)
    _write_csv(oof_query_path, PER_QUERY_COLUMNS, oof_query_rows)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "seed": seed,
        "feature_config": config["features"],
        "model_config": config["model"],
        "folds": fold_summaries,
        "five_fold_test_statistics": statistics,
        "oof_sample_count": len(oof_rows),
        "oof_query_count": len(oof_query_rows),
        "oof_metrics": oof_metrics,
        "oof_differences": {
            "degree_matched_minus_original_tcr_only_n1": differences
        },
        "outputs": {
            "oof_predictions": {
                "path": _relative(project_dir, oof_path),
                "sha256": file_sha256(oof_path),
            },
            "oof_per_query_metrics": {
                "path": _relative(project_dir, oof_query_path),
                "sha256": file_sha256(oof_query_path),
            },
        },
        "audit": {
            "status": "passed",
            "manifest": {
                "path": _relative(project_dir, manifest_path),
                "sha256": manifest_sha,
            },
            "config": {
                "path": _relative(project_dir, config_path),
                "sha256": file_sha256(config_path),
            },
            "positive_registry": registry_audit,
            "inputs": input_audit,
            "query_test_rotation": rotation_audit,
            "reference_original_tcr_only": {
                "config": config["references"]["original_tcr_only_config"],
                "summary": config["references"]["original_tcr_only_summary"],
                "oof_metrics": original_metrics,
            },
            "fixed_specification_matches_original": {
                "seed": seed == int(reference_config["seed"]),
                "features": config["features"] == reference_config["features"],
                "model": config["model"] == reference_config["model"],
            },
            "feature_fields_used": ["tcr_cdr3_beta"],
            "forbidden_feature_fields_used": [],
            "feature_schema_is_fixed_without_data_fitting": True,
            "model_fit_partitions": ["train"],
            "validation_rows_used_for_fitting": False,
            "test_rows_used_for_fitting": False,
            "resampling_performed": False,
            "hyperparameter_tuning_performed": False,
            "full_logistic_trained": False,
            "cnn_trained": False,
        },
    }
    summary_path = output_directory / "summary.json"
    report_path = output_directory / "report.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(_report(summary), encoding="utf-8")
    return summary
