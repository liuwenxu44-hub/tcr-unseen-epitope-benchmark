from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .degree_matched_tcr_only_logistic import (
    _load_and_audit_inputs,
    _validate_query_rotation,
)
from .five_fold_summary import file_sha256
from .kmer_logistic import (
    METRIC_NAMES,
    PER_QUERY_COLUMNS,
    PREDICTION_COLUMNS,
    UNK_HLA,
    _load_json,
    _load_yaml,
    _metrics,
    _relative,
    _resolve,
    _write_csv,
    build_feature_matrix,
    feature_dimensions,
    fit_hla_mapping,
)


def _load_locked_reference(
    project_dir: Path, source: Mapping[str, str], description: str
) -> tuple[Path, dict[str, Any]]:
    path = _resolve(project_dir, source["path"])
    if file_sha256(path) != source["sha256"]:
        raise ValueError(f"{description} SHA-256 mismatch")
    loader = _load_yaml if path.suffix in {".yaml", ".yml"} else _load_json
    return path, loader(path)


def _verify_fixed_specification(
    project_dir: Path,
    config: Mapping[str, Any],
    manifest_sha: str,
) -> dict[str, Any]:
    references = config["references"]
    original_config_path, original_config = _load_locked_reference(
        project_dir, references["original_full_config"], "original full config"
    )
    for field in ("seed", "features", "model"):
        if config[field] != original_config[field]:
            raise ValueError(f"degree-matched full Logistic {field} differs from original")

    original_summary_path, original_summary = _load_locked_reference(
        project_dir, references["original_full_summary"], "original full summary"
    )
    tcr_only_summary_path, tcr_only_summary = _load_locked_reference(
        project_dir,
        references["degree_matched_tcr_only_summary"],
        "degree-matched TCR-only summary",
    )
    if original_summary["seed"] != config["seed"]:
        raise ValueError("original full Logistic summary seed mismatch")
    if tcr_only_summary["seed"] != config["seed"]:
        raise ValueError("degree-matched TCR-only summary seed mismatch")
    if tcr_only_summary["audit"]["manifest"]["sha256"] != manifest_sha:
        raise ValueError("degree-matched TCR-only manifest SHA-256 mismatch")
    if tcr_only_summary["feature_config"]["forbidden"] != [
        "epitope",
        "hla",
        "tcr_epitope_interaction",
    ]:
        raise ValueError("degree-matched TCR-only reference is not TCR-only")
    return {
        "original_config_path": original_config_path,
        "original_config": original_config,
        "original_summary_path": original_summary_path,
        "original_summary": original_summary,
        "tcr_only_summary_path": tcr_only_summary_path,
        "tcr_only_summary": tcr_only_summary,
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Degree-matched five-fold full Logistic Regression",
        "",
        "This evaluation uses the frozen exact degree-matched negatives without resampling or tuning.",
        "",
        "## Fixed specification",
        "",
        "- Features: normalized TCR and epitope 1-mer/2-mer frequencies, 20x20 amino-acid composition interactions, sequence lengths, and train-only HLA one-hot with UNK.",
        "- Model and seed exactly match the original full Logistic baseline.",
        "- Each fold fits its HLA mapping and model from train only.",
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
    comparison_titles = {
        "full_logistic_minus_degree_matched_tcr_only": (
            "Full Logistic minus degree-matched TCR-only"
        ),
        "degree_matched_full_logistic_minus_original_full_logistic_n1": (
            "Degree-matched full Logistic minus original full Logistic N1"
        ),
    }
    for key, title in comparison_titles.items():
        lines.extend(["", f"## {title}", ""])
        for metric in METRIC_NAMES:
            lines.append(
                f"- {metric}: {summary['oof_differences'][key][metric]:+.6f}"
            )
    lines.extend(
        [
            "",
            "## Audit",
            "",
            "All 15 positive/negative input pairs matched the manifest SHA-256 values and independently passed exact TCR/query degree checks. No CNN was trained.",
            "",
        ]
    )
    return "\n".join(lines)


def train_degree_matched_full_logistic_five_fold(
    project_dir: Path, config_path: Path
) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    fold_ids = [str(value) for value in config["fold_ids"]]

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
    references = _verify_fixed_specification(project_dir, config, manifest_sha)

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
        hla_mapping = fit_hla_mapping(train_rows)
        train_features = build_feature_matrix(train_rows, hla_mapping)
        test_features = build_feature_matrix(test_rows, hla_mapping)
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
                "hla_mapping": hla_mapping,
                "feature_dimensions": feature_dimensions(hla_mapping),
                "input_audit": input_audit[fold_id],
                "feature_fit_partitions": ["train"],
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
            "known_train_hla_count": len(hla_mapping) - 1,
            "unknown_test_hla_count": sum(
                str(row["hla"]).strip().upper() not in hla_mapping
                for row in test_rows
            ),
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

    tcr_only_metrics = {
        metric: float(references["tcr_only_summary"]["oof_metrics"][metric])
        for metric in METRIC_NAMES
    }
    original_full_metrics = {
        metric: float(references["original_summary"]["oof_metrics"][metric])
        for metric in METRIC_NAMES
    }
    differences = {
        "full_logistic_minus_degree_matched_tcr_only": {
            metric: float(oof_metrics[metric]) - tcr_only_metrics[metric]
            for metric in METRIC_NAMES
        },
        "degree_matched_full_logistic_minus_original_full_logistic_n1": {
            metric: float(oof_metrics[metric]) - original_full_metrics[metric]
            for metric in METRIC_NAMES
        },
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
        "oof_differences": differences,
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
            "references": {
                "original_full_config": {
                    "path": _relative(
                        project_dir, references["original_config_path"]
                    ),
                    "sha256": config["references"]["original_full_config"][
                        "sha256"
                    ],
                },
                "original_full_summary": {
                    "path": _relative(
                        project_dir, references["original_summary_path"]
                    ),
                    "sha256": config["references"]["original_full_summary"][
                        "sha256"
                    ],
                    "oof_metrics": original_full_metrics,
                },
                "degree_matched_tcr_only_summary": {
                    "path": _relative(
                        project_dir, references["tcr_only_summary_path"]
                    ),
                    "sha256": config["references"][
                        "degree_matched_tcr_only_summary"
                    ]["sha256"],
                    "oof_metrics": tcr_only_metrics,
                },
            },
            "fixed_specification_matches_original": {
                "seed": seed == int(references["original_config"]["seed"]),
                "features": config["features"]
                == references["original_config"]["features"],
                "model": config["model"]
                == references["original_config"]["model"],
            },
            "feature_fields_used": ["tcr_cdr3_beta", "epitope", "hla"],
            "feature_fit_partitions": ["train"],
            "model_fit_partitions": ["train"],
            "validation_rows_used_for_fitting": False,
            "test_rows_used_for_fitting": False,
            "resampling_performed": False,
            "hyperparameter_tuning_performed": False,
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
