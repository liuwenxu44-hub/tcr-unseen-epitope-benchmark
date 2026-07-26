from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .five_fold_summary import file_sha256, validate_query_test_rotation
from .kmer_logistic import (
    METRIC_NAMES,
    PER_QUERY_COLUMNS,
    PREDICTION_COLUMNS,
    _audit_manifest_inputs,
    _load_json,
    _load_yaml,
    _metrics,
    _read_csv,
    _relative,
    _resolve,
    _write_csv,
    normalized_kmer_frequencies,
)
from .metrics import binary_classification_metrics, query_level_classification_metrics
from .vocab import STANDARD_AMINO_ACIDS


TCR_ONLY_FEATURE_COUNT = 421


def build_tcr_only_feature_matrix(rows: Sequence[Mapping[str, Any]]) -> csr_matrix:
    alphabet_size = len(STANDARD_AMINO_ACIDS)
    tcr_1_offset = 0
    tcr_2_offset = alphabet_size
    length_offset = alphabet_size + alphabet_size**2
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []

    def append(row_index: int, offset: int, values: Mapping[int, float]) -> None:
        for column, value in values.items():
            if value:
                matrix_rows.append(row_index)
                matrix_columns.append(offset + column)
                matrix_values.append(float(value))

    for row_index, row in enumerate(rows):
        tcr = str(row["tcr_cdr3_beta"]).strip().upper()
        append(row_index, tcr_1_offset, normalized_kmer_frequencies(tcr, 1))
        append(row_index, tcr_2_offset, normalized_kmer_frequencies(tcr, 2))
        append(row_index, length_offset, {0: len(tcr)})
    return csr_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(rows), TCR_ONLY_FEATURE_COUNT),
        dtype=np.float64,
    )


def _assert_close(observed: float, expected: float, description: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{description} mismatch: {observed} != {expected}")


def _verify_reference_inputs(
    manifest: Mapping[str, Any], full_summary: Mapping[str, Any]
) -> None:
    for fold_id, fold in manifest["folds"].items():
        for partition, entry in fold["partitions"].items():
            for kind, output_key in (
                ("positive", "positive_output"),
                ("negative", "negative_output"),
            ):
                expected = entry[output_key]
                observed = full_summary["audit"]["inputs"][fold_id][partition][kind]
                if (
                    observed["path"] != expected["path"]
                    or observed["sha256"] != expected["sha256"]
                    or int(observed["rows"]) != int(expected["rows"])
                ):
                    raise ValueError(
                        f"full Logistic input mismatch for {fold_id} {partition} {kind}"
                    )


def _verify_and_compare_oof(
    *,
    project_dir: Path,
    scored_rows: Sequence[Mapping[str, Any]],
    tcr_only_metrics: Mapping[str, float],
    full_summary: Mapping[str, Any],
    full_oof_path: Path,
    cnn_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    full_rows = _read_csv(full_oof_path)
    cnn_source = cnn_summary["outputs"]["oof_predictions_csv"]
    cnn_oof_path = _resolve(project_dir, cnn_source["path"])
    cnn_sha = file_sha256(cnn_oof_path)
    if cnn_sha != cnn_source["sha256"]:
        raise ValueError("CNN OOF SHA-256 mismatch")
    cnn_rows = _read_csv(cnn_oof_path)
    if len(scored_rows) != len(full_rows) or len(scored_rows) != len(cnn_rows):
        raise ValueError("OOF row counts differ across TCR-only, full Logistic, and CNN")
    identity_fields = ("fold_id", "sample_id", "source", "pmhc_id", "label")
    for control, full, cnn in zip(scored_rows, full_rows, cnn_rows, strict=True):
        for field in identity_fields:
            expected = str(control[field])
            if str(full[field]) != expected or str(cnn[field]) != expected:
                raise ValueError(f"OOF identity mismatch for {control['sample_id']}: {field}")
        for field in ("tcr_cdr3_beta", "epitope", "hla"):
            if str(full[field]) != str(control[field]):
                raise ValueError(
                    f"full Logistic interaction mismatch for {control['sample_id']}: {field}"
                )

    full_pooled = binary_classification_metrics(
        [int(row["label"]) for row in full_rows],
        [float(row["probability"]) for row in full_rows],
    )
    full_macro, _ = query_level_classification_metrics(full_rows)
    full_metrics = {
        "pooled_auroc": full_pooled["auroc"],
        "pooled_auprc": full_pooled["auprc"],
        "macro_auroc": float(full_macro["auroc"]),
        "macro_auprc": float(full_macro["auprc"]),
    }
    for metric in METRIC_NAMES:
        _assert_close(
            full_metrics[metric],
            float(full_summary["oof_metrics"][metric]),
            f"full Logistic OOF {metric}",
        )

    cnn_pooled = binary_classification_metrics(
        [int(row["label"]) for row in cnn_rows],
        [float(row["probability"]) for row in cnn_rows],
    )
    cnn_macro, _ = query_level_classification_metrics(cnn_rows)
    cnn_metrics = {
        "pooled_auroc": cnn_pooled["auroc"],
        "pooled_auprc": cnn_pooled["auprc"],
        "macro_auroc": float(cnn_macro["auroc"]),
        "macro_auprc": float(cnn_macro["auprc"]),
    }
    cnn_stored = {
        "pooled_auroc": cnn_summary["oof"]["pooled"]["auroc"],
        "pooled_auprc": cnn_summary["oof"]["pooled"]["auprc"],
        "macro_auroc": cnn_summary["oof"]["query_macro"]["auroc"],
        "macro_auprc": cnn_summary["oof"]["query_macro"]["auprc"],
    }
    for metric in METRIC_NAMES:
        _assert_close(cnn_metrics[metric], float(cnn_stored[metric]), f"CNN OOF {metric}")

    comparisons = {
        "tcr_only_minus_full_logistic": {
            metric: float(tcr_only_metrics[metric]) - full_metrics[metric]
            for metric in METRIC_NAMES
        },
        "tcr_only_minus_cnn": {
            metric: float(tcr_only_metrics[metric]) - cnn_metrics[metric]
            for metric in METRIC_NAMES
        },
    }
    audit = {
        "identity_fields_match": True,
        "interaction_keys_match": True,
        "sample_count": len(scored_rows),
        "full_logistic": {
            "path": _relative(project_dir, full_oof_path),
            "sha256": file_sha256(full_oof_path),
            "metrics": full_metrics,
        },
        "cnn": {
            "path": _relative(project_dir, cnn_oof_path),
            "sha256": cnn_sha,
            "metrics": cnn_metrics,
        },
    }
    return comparisons, audit


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Five-fold TCR-only logistic regression control",
        "",
        "This control uses the frozen cap=100 folds without resampling.",
        "",
        "## Fixed specification",
        "",
        "- Features: normalized TCR 1-mer and 2-mer frequencies plus raw TCR length (421 dimensions).",
        "- Forbidden inputs: epitope, HLA, and all TCR-epitope interaction features.",
        "- Model: L2 LogisticRegression, C=1.0, class_weight=None, max_iter=5000, seed 13.",
        "- Fitting: each fold's model uses train only; validation and test never participate in fitting.",
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
            f"- {metric}: {item['mean']:.6f} +/- {item['standard_deviation']:.6f}"
        )
    for metric, value in summary["oof_metrics"].items():
        lines.append(f"- OOF {metric}: {value:.6f}")
    for comparison_name, metrics in summary["oof_differences"].items():
        lines.extend(["", f"## {comparison_name}", ""])
        for metric in METRIC_NAMES:
            lines.append(f"- {metric}: {metrics[metric]:+.6f}")
    lines.extend(
        [
            "",
            "## Audit",
            "",
            "All manifest hashes and OOF identities passed. Existing models were not retrained or modified.",
            "",
        ]
    )
    return "\n".join(lines)


def train_tcr_only_logistic_five_fold(
    project_dir: Path, config_path: Path
) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    reference_config_path = _resolve(
        project_dir, config["references"]["full_logistic_config"]
    )
    reference_config = _load_yaml(reference_config_path)
    if int(config["seed"]) != int(reference_config["seed"]):
        raise ValueError("TCR-only seed differs from the full Logistic baseline")
    if config["model"] != reference_config["model"]:
        raise ValueError("TCR-only model configuration differs from the full baseline")
    expected_features = {
        "amino_acids": "ACDEFGHIKLMNPQRSTVWY",
        "tcr_kmers": [1, 2],
        "normalized_frequencies": True,
        "include_tcr_length": True,
        "feature_count": TCR_ONLY_FEATURE_COUNT,
        "forbidden": ["epitope", "hla", "tcr_epitope_interaction"],
    }
    if config["features"] != expected_features:
        raise ValueError("TCR-only feature configuration is not the fixed 421-feature control")
    output_directory = _resolve(project_dir, config["output"]["directory"])
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output_directory}"
        )

    fold_ids = [str(value) for value in config["fold_ids"]]
    manifest_path = _resolve(project_dir, config["data"]["manifest"])
    manifest = _load_json(manifest_path)
    manifest_sha = file_sha256(manifest_path)
    rotation_audit = validate_query_test_rotation(manifest, fold_ids)
    input_audit, rows_by_fold = _audit_manifest_inputs(project_dir, manifest, fold_ids)
    full_summary_path = _resolve(
        project_dir, config["references"]["full_logistic_summary"]
    )
    cnn_summary_path = _resolve(project_dir, config["references"]["cnn_summary"])
    full_summary = _load_json(full_summary_path)
    cnn_summary = _load_json(cnn_summary_path)
    for method, summary in (("full Logistic", full_summary), ("CNN", cnn_summary)):
        if summary["audit"]["manifest"]["sha256"] != manifest_sha:
            raise ValueError(f"{method} manifest SHA-256 mismatch")
    _verify_reference_inputs(manifest, full_summary)
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
        train_labels = np.asarray([row["label"] for row in train_rows], dtype=np.int64)
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
        probabilities = np.asarray(model.predict_proba(test_features)[:, 1], dtype=np.float64)
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
    statistics = {}
    for metric in METRIC_NAMES:
        values = np.asarray(
            [fold_summaries[fold_id]["test_metrics"][metric] for fold_id in fold_ids],
            dtype=np.float64,
        )
        statistics[metric] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    oof_path = output_directory / "oof_test_predictions.csv"
    oof_query_path = output_directory / "oof_per_query_metrics.csv"
    _write_csv(oof_path, PREDICTION_COLUMNS, oof_rows)
    _write_csv(oof_query_path, PER_QUERY_COLUMNS, oof_query_rows)
    full_oof_path = _resolve(project_dir, config["references"]["full_logistic_oof"])
    differences, comparison_audit = _verify_and_compare_oof(
        project_dir=project_dir,
        scored_rows=oof_rows,
        tcr_only_metrics=oof_metrics,
        full_summary=full_summary,
        full_oof_path=full_oof_path,
        cnn_summary=cnn_summary,
    )
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
            "reference_full_logistic_config": {
                "path": _relative(project_dir, reference_config_path),
                "sha256": file_sha256(reference_config_path),
            },
            "reference_full_logistic_summary": {
                "path": _relative(project_dir, full_summary_path),
                "sha256": file_sha256(full_summary_path),
            },
            "reference_cnn_summary": {
                "path": _relative(project_dir, cnn_summary_path),
                "sha256": file_sha256(cnn_summary_path),
            },
            "inputs": input_audit,
            "query_test_rotation": rotation_audit,
            "oof_comparison": comparison_audit,
            "feature_fields_used": ["tcr_cdr3_beta"],
            "forbidden_feature_fields_used": [],
            "feature_schema_is_fixed_without_data_fitting": True,
            "model_fit_partitions": ["train"],
            "validation_rows_used_for_fitting": False,
            "test_rows_used_for_fitting": False,
            "resampling_performed": False,
            "hyperparameter_tuning_performed": False,
            "existing_models_retrained": False,
        },
    }
    summary_path = output_directory / "summary.json"
    report_path = output_directory / "report.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(_report(summary), encoding="utf-8")
    return summary
