from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .degree_matched_full_logistic import _load_locked_reference
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
    fit_hla_mapping,
    normalized_kmer_frequencies,
)
from .vocab import STANDARD_AMINO_ACIDS


ADDITIVE_SEQUENCE_FEATURE_COUNT = 2 * (
    len(STANDARD_AMINO_ACIDS) + len(STANDARD_AMINO_ACIDS) ** 2
) + 2
FORBIDDEN_INTERACTIONS = (
    "tcr_epitope_interaction",
    "amino_acid_composition_interaction",
)


def additive_feature_dimensions(hla_mapping: Mapping[str, int]) -> dict[str, int]:
    if set(hla_mapping.values()) != set(range(len(hla_mapping))):
        raise ValueError("HLA IDs must be contiguous and zero-based")
    sequence_size = len(STANDARD_AMINO_ACIDS) + len(STANDARD_AMINO_ACIDS) ** 2
    return {
        "tcr_sequence": sequence_size,
        "epitope_sequence": sequence_size,
        "length": 2,
        "interaction": 0,
        "sequence_and_length": ADDITIVE_SEQUENCE_FEATURE_COUNT,
        "hla": len(hla_mapping),
        "total": ADDITIVE_SEQUENCE_FEATURE_COUNT + len(hla_mapping),
    }


def build_additive_feature_matrix(
    rows: Sequence[Mapping[str, Any]], hla_mapping: Mapping[str, int]
) -> csr_matrix:
    dimensions = additive_feature_dimensions(hla_mapping)
    alphabet_size = len(STANDARD_AMINO_ACIDS)
    tcr_1_offset = 0
    tcr_2_offset = tcr_1_offset + alphabet_size
    epitope_1_offset = tcr_2_offset + alphabet_size**2
    epitope_2_offset = epitope_1_offset + alphabet_size
    length_offset = epitope_2_offset + alphabet_size**2
    hla_offset = dimensions["sequence_and_length"]
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
        epitope = str(row["epitope"]).strip().upper()
        append(row_index, tcr_1_offset, normalized_kmer_frequencies(tcr, 1))
        append(row_index, tcr_2_offset, normalized_kmer_frequencies(tcr, 2))
        append(
            row_index,
            epitope_1_offset,
            normalized_kmer_frequencies(epitope, 1),
        )
        append(
            row_index,
            epitope_2_offset,
            normalized_kmer_frequencies(epitope, 2),
        )
        append(row_index, length_offset, {0: len(tcr), 1: len(epitope)})
        hla = str(row["hla"]).strip().upper()
        append(
            row_index,
            hla_offset,
            {hla_mapping.get(hla, hla_mapping[UNK_HLA]): 1.0},
        )
    return csr_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(rows), dimensions["total"]),
        dtype=np.float64,
    )


def _verify_fixed_specification(
    project_dir: Path,
    config: Mapping[str, Any],
    manifest_sha: str,
) -> dict[str, Any]:
    references = config["references"]
    full_config_path, full_config = _load_locked_reference(
        project_dir, references["degree_matched_full_config"], "full config"
    )
    full_summary_path, full_summary = _load_locked_reference(
        project_dir, references["degree_matched_full_summary"], "full summary"
    )
    tcr_only_summary_path, tcr_only_summary = _load_locked_reference(
        project_dir, references["degree_matched_tcr_only_summary"], "TCR-only summary"
    )
    if config["seed"] != full_config["seed"] or config["model"] != full_config["model"]:
        raise ValueError("additive-only model parameters or seed differ from full Logistic")
    shared_feature_fields = (
        "amino_acids",
        "tcr_kmers",
        "epitope_kmers",
        "normalized_frequencies",
        "include_tcr_length",
        "include_epitope_length",
        "hla_encoding",
        "unknown_hla_token",
    )
    for field in shared_feature_fields:
        if config["features"].get(field) != full_config["features"].get(field):
            raise ValueError(f"additive-only feature setting differs for {field}")
    if config["features"].get("interaction_features") != []:
        raise ValueError("additive-only interaction_features must be empty")
    if config["features"].get("forbidden") != list(FORBIDDEN_INTERACTIONS):
        raise ValueError("additive-only forbidden interaction list is incomplete")
    if "amino_acid_composition_interaction" in config["features"]:
        raise ValueError("additive-only config must remove the 20x20 interaction")
    for description, summary in (
        ("full Logistic", full_summary),
        ("TCR-only", tcr_only_summary),
    ):
        if int(summary["seed"]) != int(config["seed"]):
            raise ValueError(f"{description} seed mismatch")
        if summary["audit"]["manifest"]["sha256"] != manifest_sha:
            raise ValueError(f"{description} manifest SHA-256 mismatch")
    return {
        "full_config_path": full_config_path,
        "full_config": full_config,
        "full_summary_path": full_summary_path,
        "full_summary": full_summary,
        "tcr_only_summary_path": tcr_only_summary_path,
        "tcr_only_summary": tcr_only_summary,
        "shared_feature_fields": list(shared_feature_fields),
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Degree-matched five-fold additive-only Logistic Regression",
        "",
        "This evaluation uses the frozen degree-matched folds without resampling or tuning.",
        "",
        "## Fixed specification",
        "",
        "- Features: additive normalized TCR and epitope 1-mer/2-mer frequencies, both sequence lengths, and train-only HLA one-hot with UNK.",
        "- All TCR-epitope interaction features, including the 20x20 amino-acid composition interaction, are absent.",
        "- Model parameters and seed exactly match the degree-matched full Logistic baseline.",
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
        "additive_only_minus_degree_matched_tcr_only": (
            "Additive-only minus degree-matched TCR-only"
        ),
        "degree_matched_full_logistic_minus_additive_only": (
            "Degree-matched full Logistic minus additive-only"
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
            "All 15 input pairs matched the manifest hashes and passed exact TCR/query degree checks. Interaction feature count is zero; no CNN was trained.",
            "",
        ]
    )
    return "\n".join(lines)


def train_degree_matched_additive_logistic_five_fold(
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
        train_features = build_additive_feature_matrix(train_rows, hla_mapping)
        test_features = build_additive_feature_matrix(test_rows, hla_mapping)
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
        dimensions = additive_feature_dimensions(hla_mapping)
        joblib.dump(
            {
                "estimator": model,
                "feature_config": config["features"],
                "model_config": config["model"],
                "seed": seed,
                "fold_id": fold_id,
                "hla_mapping": hla_mapping,
                "feature_dimensions": dimensions,
                "feature_fields_used": ["tcr_cdr3_beta", "epitope", "hla"],
                "interaction_feature_count": 0,
                "forbidden_interaction_fields_used": [],
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
            "interaction_feature_count": 0,
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

    full_metrics = {
        metric: float(references["full_summary"]["oof_metrics"][metric])
        for metric in METRIC_NAMES
    }
    tcr_only_metrics = {
        metric: float(references["tcr_only_summary"]["oof_metrics"][metric])
        for metric in METRIC_NAMES
    }
    differences = {
        "additive_only_minus_degree_matched_tcr_only": {
            metric: float(oof_metrics[metric]) - tcr_only_metrics[metric]
            for metric in METRIC_NAMES
        },
        "degree_matched_full_logistic_minus_additive_only": {
            metric: full_metrics[metric] - float(oof_metrics[metric])
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
                "degree_matched_full_config": {
                    "path": _relative(project_dir, references["full_config_path"]),
                    "sha256": config["references"]["degree_matched_full_config"][
                        "sha256"
                    ],
                },
                "degree_matched_full_summary": {
                    "path": _relative(project_dir, references["full_summary_path"]),
                    "sha256": config["references"]["degree_matched_full_summary"][
                        "sha256"
                    ],
                    "oof_metrics": full_metrics,
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
            "fixed_specification_matches_full": {
                "seed": seed == int(references["full_config"]["seed"]),
                "model": config["model"] == references["full_config"]["model"],
                "shared_features": True,
            },
            "feature_fields_used": ["tcr_cdr3_beta", "epitope", "hla"],
            "interaction_feature_count": 0,
            "forbidden_interaction_fields_used": [],
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
