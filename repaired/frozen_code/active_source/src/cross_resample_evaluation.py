from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from .degree_matched_additive_logistic import build_additive_feature_matrix
from .degree_matched_orthogonalized_logistic import (
    _build_feature_matrix_and_audit,
)
from .five_fold_summary import file_sha256
from .interaction_resample_robustness import (
    COMPARISON,
    METHODS,
    RESAMPLE_IDS,
    load_and_audit_resample_inputs,
)
from .kmer_logistic import (
    METRIC_NAMES,
    _load_json,
    _load_yaml,
    _metrics,
    _relative,
    _resolve,
    _write_csv,
    fit_hla_mapping,
)


MATRIX_COLUMNS = (
    "train_resample",
    "test_resample",
    "diagonal",
    *(f"additive_only_{metric}" for metric in METRIC_NAMES),
    *(f"orthogonalized_interaction_{metric}" for metric in METRIC_NAMES),
    *(f"orthogonalized_minus_additive_only_{metric}" for metric in METRIC_NAMES),
)


def _locked_json(
    project_dir: Path, source: Mapping[str, str], description: str
) -> tuple[Path, dict[str, Any]]:
    path = _resolve(project_dir, source["path"])
    if file_sha256(path) != source["sha256"]:
        raise ValueError(f"{description} SHA-256 mismatch")
    return path, _load_json(path)


def _locked_yaml(
    project_dir: Path, source: Mapping[str, str], description: str
) -> tuple[Path, dict[str, Any]]:
    path = _resolve(project_dir, source["path"])
    if file_sha256(path) != source["sha256"]:
        raise ValueError(f"{description} SHA-256 mismatch")
    return path, _load_yaml(path)


def _assert_close(observed: float, expected: float, description: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{description} mismatch: {observed} != {expected}")


def load_and_audit_models(
    project_dir: Path,
    training_summary: Mapping[str, Any],
    training_config: Mapping[str, Any],
    rows_by_resample: Mapping[
        str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]
    ],
    input_audits: Mapping[str, Mapping[str, Any]],
    fold_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_count = len(RESAMPLE_IDS) * len(METHODS) * len(fold_ids)
    if training_summary.get("trained_model_count") != expected_count:
        raise ValueError("training summary does not contain 50 models")
    if training_summary.get("seed") != training_config.get("seed"):
        raise ValueError("training seed mismatch")
    if training_summary.get("model_config") != training_config.get("model"):
        raise ValueError("training model configuration mismatch")
    if training_summary.get("feature_configs") != training_config.get("features"):
        raise ValueError("training feature configuration mismatch")

    models: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    for resample_id in RESAMPLE_IDS:
        models[resample_id] = {}
        audit[resample_id] = {}
        for method in METHODS:
            models[resample_id][method] = {}
            audit[resample_id][method] = {}
            method_summary = training_summary["resamples"][resample_id]["methods"][
                method
            ]
            for fold_id in fold_ids:
                source = method_summary["folds"][fold_id]["model"]
                path = _resolve(project_dir, source["path"])
                observed_sha = file_sha256(path)
                if observed_sha != source["sha256"]:
                    raise ValueError(
                        f"checkpoint SHA-256 mismatch: {resample_id}/{method}/{fold_id}"
                    )
                checkpoint = joblib.load(path)
                expected_metadata = {
                    "resample_id": resample_id,
                    "method": method,
                    "fold_id": fold_id,
                    "seed": int(training_config["seed"]),
                    "model_config": training_config["model"],
                    "feature_config": training_config["features"][method],
                    "feature_fit_partitions": ["train"],
                    "model_fit_partitions": ["train"],
                    "test_evaluation_count": 1,
                }
                for field, expected in expected_metadata.items():
                    if checkpoint.get(field) != expected:
                        raise ValueError(
                            f"checkpoint metadata mismatch: "
                            f"{resample_id}/{method}/{fold_id}/{field}"
                        )
                if not isinstance(checkpoint.get("estimator"), LogisticRegression):
                    raise TypeError("checkpoint estimator is not LogisticRegression")
                expected_hla = fit_hla_mapping(
                    rows_by_resample[resample_id][fold_id]["train"]
                )
                if checkpoint.get("hla_mapping") != expected_hla:
                    raise ValueError(
                        f"train-only HLA mapping mismatch: {resample_id}/{method}/{fold_id}"
                    )
                if checkpoint.get("input_audit") != input_audits[resample_id][fold_id]:
                    raise ValueError(
                        f"checkpoint input audit mismatch: {resample_id}/{method}/{fold_id}"
                    )
                models[resample_id][method][fold_id] = checkpoint
                audit[resample_id][method][fold_id] = {
                    "path": _relative(project_dir, path),
                    "sha256": observed_sha,
                    "metadata_match": True,
                    "train_only_hla_mapping_match": True,
                }
    return models, audit


def summarize_cross_resample_cells(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(cells) != 25:
        raise ValueError("cross-resample evaluation requires 25 cells")
    identities = {
        (str(cell["train_resample"]), str(cell["test_resample"])) for cell in cells
    }
    expected = {
        (train_resample, test_resample)
        for train_resample in RESAMPLE_IDS
        for test_resample in RESAMPLE_IDS
    }
    if identities != expected:
        raise ValueError("cross-resample matrix is incomplete or duplicated")

    diagonal = [
        cell for cell in cells if cell["train_resample"] == cell["test_resample"]
    ]
    off_diagonal = [
        cell for cell in cells if cell["train_resample"] != cell["test_resample"]
    ]
    if len(diagonal) != 5 or len(off_diagonal) != 20:
        raise ValueError("diagonal/off-diagonal cell counts are incorrect")

    def statistics(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result = {}
        for section in (*METHODS, COMPARISON):
            result[section] = {}
            for metric in METRIC_NAMES:
                values = np.asarray(
                    [float(cell[section][metric]) for cell in selected],
                    dtype=np.float64,
                )
                result[section][metric] = {
                    "mean": float(np.mean(values)),
                    "standard_deviation": float(np.std(values, ddof=1)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
        return result

    worst_case = {}
    for train_resample in RESAMPLE_IDS:
        selected = [
            cell for cell in cells if cell["train_resample"] == train_resample
        ]
        worst_case[train_resample] = {
            section: {
                metric: min(float(cell[section][metric]) for cell in selected)
                for metric in METRIC_NAMES
            }
            for section in (*METHODS, COMPARISON)
        }
    positive_counts = {
        metric: sum(float(cell[COMPARISON][metric]) > 0.0 for cell in cells)
        for metric in METRIC_NAMES
    }
    return {
        "diagonal_cell_count": len(diagonal),
        "off_diagonal_cell_count": len(off_diagonal),
        "diagonal_statistics": statistics(diagonal),
        "off_diagonal_statistics": statistics(off_diagonal),
        "worst_case_by_train_resample": worst_case,
        "positive_interaction_gain_counts": positive_counts,
        "combination_count": len(cells),
    }


def _matrix_rows(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_key = {
        (cell["train_resample"], cell["test_resample"]): cell for cell in cells
    }
    for train_resample in RESAMPLE_IDS:
        for test_resample in RESAMPLE_IDS:
            cell = by_key[(train_resample, test_resample)]
            row: dict[str, Any] = {
                "train_resample": train_resample,
                "test_resample": test_resample,
                "diagonal": train_resample == test_resample,
            }
            for method in METHODS:
                for metric in METRIC_NAMES:
                    row[f"{method}_{metric}"] = cell[method][metric]
            for metric in METRIC_NAMES:
                row[f"{COMPARISON}_{metric}"] = cell[COMPARISON][metric]
            rows.append(row)
    return rows


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Cross-negative-resample Logistic evaluation",
        "",
        "All 50 fixed checkpoints were evaluated across all five corresponding test resamples without retraining or tuning.",
        "",
    ]
    cells = {
        (cell["train_resample"], cell["test_resample"]): cell
        for cell in summary["cells"]
    }
    lines.extend(
        [
            "## Diagonal results",
            "",
            "Each cell is Orthogonalized / Additive-only / difference.",
            "",
            "| Resample | Pooled AUROC | Pooled AUPRC | Macro AUROC | Macro AUPRC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for resample_id in RESAMPLE_IDS:
        cell = cells[(resample_id, resample_id)]
        values = []
        for metric in METRIC_NAMES:
            values.append(
                f"{cell['orthogonalized_interaction'][metric]:.6f} / "
                f"{cell['additive_only'][metric]:.6f} / "
                f"{cell[COMPARISON][metric]:+.6f}"
            )
        lines.append(f"| {resample_id} | {' | '.join(values)} |")
    lines.extend(["", "## Orthogonalized OOF metric matrices", ""])
    for metric in METRIC_NAMES:
        lines.extend(
            [
                f"### {metric}",
                "",
                "| Train \\ Test | resample_1 | resample_2 | resample_3 | resample_4 | resample_5 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for train_resample in RESAMPLE_IDS:
            values = " | ".join(
                f"{cells[(train_resample, test_resample)]['orthogonalized_interaction'][metric]:.6f}"
                for test_resample in RESAMPLE_IDS
            )
            lines.append(f"| {train_resample} | {values} |")
        lines.append("")

    off_diagonal = summary["off_diagonal_statistics"]
    lines.extend(["## Twenty off-diagonal combinations", ""])
    for method in METHODS:
        lines.append(f"### {method}")
        lines.append("")
        for metric in METRIC_NAMES:
            item = off_diagonal[method][metric]
            lines.append(
                f"- {metric}: {item['mean']:.6f} +/- {item['standard_deviation']:.6f} "
                f"(range {item['minimum']:.6f} to {item['maximum']:.6f})"
            )
        lines.append("")
    lines.extend(
        [
            "## Worst case by training resample",
            "",
            "Each Orthogonalized, Additive-only, and difference value is independently minimized over the five test resamples.",
            "",
            "| Train resample | Pooled AUROC | Pooled AUPRC | Macro AUROC | Macro AUPRC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for train_resample in RESAMPLE_IDS:
        worst = summary["worst_case_by_train_resample"][train_resample]
        values = []
        for metric in METRIC_NAMES:
            values.append(
                f"{worst['orthogonalized_interaction'][metric]:.6f} / "
                f"{worst['additive_only'][metric]:.6f} / "
                f"{worst[COMPARISON][metric]:+.6f}"
            )
        lines.append(f"| {train_resample} | {' | '.join(values)} |")
    lines.extend(["", "## Positive Orthogonalized interaction gains", ""])
    for metric, count in summary["positive_interaction_gain_counts"].items():
        lines.append(f"- {metric}: {count}/25")
    lines.extend(
        [
            "",
            "## Audit",
            "",
            "All checkpoint and test CSV SHA-256 values matched. Every test partition passed exact TCR/query degree, label, known-positive conflict, duplicate, and query-leakage audits. The 5 diagonal cells reproduce the prior OOF results. No checkpoint or data file was modified; no model was trained and no bootstrap was run.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_cross_resamples(
    project_dir: Path, config_path: Path
) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    fold_ids = [str(value) for value in config["fold_ids"]]
    if tuple(fold_ids) != tuple(f"fold_{index}" for index in range(1, 6)):
        raise ValueError("five fixed folds are required")
    if tuple(config["resample_ids"]) != RESAMPLE_IDS:
        raise ValueError("resample_1 through resample_5 are required")
    if tuple(config["methods"]) != METHODS or tuple(config["metrics"]) != METRIC_NAMES:
        raise ValueError("cross-resample methods or metrics changed")

    output_directory = _resolve(project_dir, config["output"]["directory"])
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output_directory}"
        )
    training_config_path, training_config = _locked_yaml(
        project_dir, config["inputs"]["training_config"], "training config"
    )
    training_summary_path, training_summary = _locked_json(
        project_dir, config["inputs"]["training_summary"], "training summary"
    )
    manifest_path, manifest = _locked_json(
        project_dir, config["inputs"]["resample_manifest"], "resample manifest"
    )
    if training_config["data"]["resample_manifest"]["sha256"] != config["inputs"][
        "resample_manifest"
    ]["sha256"]:
        raise ValueError("training and evaluation manifests differ")

    input_audits = {}
    rows_by_resample = {}
    leakage_audits = {}
    registry_audit = None
    for resample_id in RESAMPLE_IDS:
        inputs, rows, registry, leakage = load_and_audit_resample_inputs(
            project_dir, manifest, resample_id, fold_ids
        )
        input_audits[resample_id] = inputs
        rows_by_resample[resample_id] = rows
        leakage_audits[resample_id] = leakage
        if registry_audit is None:
            registry_audit = registry
        elif registry_audit != registry:
            raise ValueError("positive registry audit differs across resamples")

    models, model_audit = load_and_audit_models(
        project_dir,
        training_summary,
        training_config,
        rows_by_resample,
        input_audits,
        fold_ids,
    )
    checkpoint_hashes_before = {
        (resample_id, method, fold_id): model_audit[resample_id][method][fold_id][
            "sha256"
        ]
        for resample_id in RESAMPLE_IDS
        for method in METHODS
        for fold_id in fold_ids
    }

    cells = []
    orthogonality_audit = {}
    for train_resample in RESAMPLE_IDS:
        orthogonality_audit[train_resample] = {}
        for test_resample in RESAMPLE_IDS:
            method_results = {}
            fold_results = {fold_id: {} for fold_id in fold_ids}
            orthogonality_audit[train_resample][test_resample] = {}
            for method in METHODS:
                scored_rows = []
                orthogonality_audit[train_resample][test_resample][method] = {}
                for fold_id in fold_ids:
                    checkpoint = models[train_resample][method][fold_id]
                    test_rows = rows_by_resample[test_resample][fold_id]["test"]
                    hla_mapping = checkpoint["hla_mapping"]
                    if method == "additive_only":
                        features = build_additive_feature_matrix(
                            test_rows, hla_mapping
                        )
                        feature_audit = None
                    else:
                        features, feature_audit = _build_feature_matrix_and_audit(
                            test_rows, hla_mapping
                        )
                    estimator = checkpoint["estimator"]
                    logits = np.asarray(
                        estimator.decision_function(features), dtype=np.float64
                    )
                    probabilities = 1.0 / (1.0 + np.exp(-logits))
                    fold_rows = [
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
                    fold_metrics, _ = _metrics(fold_rows)
                    fold_results[fold_id][method] = fold_metrics
                    scored_rows.extend(fold_rows)
                    orthogonality_audit[train_resample][test_resample][method][
                        fold_id
                    ] = feature_audit
                oof_metrics, per_query = _metrics(scored_rows)
                if len(per_query) != 168:
                    raise ValueError(
                        f"OOF query count mismatch: {train_resample}/{test_resample}/{method}"
                    )
                method_results[method] = oof_metrics
            difference = {
                metric: float(method_results["orthogonalized_interaction"][metric])
                - float(method_results["additive_only"][metric])
                for metric in METRIC_NAMES
            }
            cell = {
                "train_resample": train_resample,
                "test_resample": test_resample,
                "diagonal": train_resample == test_resample,
                **method_results,
                COMPARISON: difference,
                "fold_metrics": fold_results,
            }
            if cell["diagonal"]:
                prior = training_summary["resamples"][train_resample]["methods"]
                for method in METHODS:
                    for metric in METRIC_NAMES:
                        _assert_close(
                            float(cell[method][metric]),
                            float(prior[method]["oof_metrics"][metric]),
                            f"diagonal {train_resample}/{method}/{metric}",
                        )
            cells.append(cell)

    aggregate = summarize_cross_resample_cells(cells)
    checkpoint_hashes_after = {
        key: file_sha256(
            _resolve(
                project_dir,
                model_audit[key[0]][key[1]][key[2]]["path"],
            )
        )
        for key in checkpoint_hashes_before
    }
    if checkpoint_hashes_after != checkpoint_hashes_before:
        raise ValueError("checkpoint files changed during evaluation")

    output_directory.mkdir(parents=True, exist_ok=True)
    matrix_path = output_directory / config["output"]["matrix_csv"]
    report_path = output_directory / config["output"]["report_md"]
    summary_path = output_directory / config["output"]["summary_json"]
    _write_csv(matrix_path, MATRIX_COLUMNS, _matrix_rows(cells))
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "cells": cells,
        **aggregate,
        "audit": {
            "status": "passed",
            "config": {
                "path": _relative(project_dir, config_path),
                "sha256": file_sha256(config_path),
            },
            "training_config": {
                "path": _relative(project_dir, training_config_path),
                "sha256": config["inputs"]["training_config"]["sha256"],
            },
            "training_summary": {
                "path": _relative(project_dir, training_summary_path),
                "sha256": config["inputs"]["training_summary"]["sha256"],
            },
            "resample_manifest": {
                "path": _relative(project_dir, manifest_path),
                "sha256": config["inputs"]["resample_manifest"]["sha256"],
            },
            "positive_registry": registry_audit,
            "inputs": input_audits,
            "leakage": leakage_audits,
            "models": model_audit,
            "loaded_checkpoint_count": len(checkpoint_hashes_before),
            "checkpoint_sha256_unchanged": True,
            "test_fold_scoring_count": 250,
            "all_input_sha256_match": True,
            "all_tcr_degrees_exact": True,
            "all_query_degrees_exact": True,
            "labels_valid": True,
            "queries_valid": True,
            "known_positive_registry_intersections": 0,
            "partition_leakage": 0,
            "diagonal_reproduces_prior_oof": True,
            "models_trained": False,
            "hyperparameter_tuning_performed": False,
            "bootstrap_performed": False,
            "data_or_checkpoints_modified": False,
            "orthogonality": orthogonality_audit,
        },
    }
    report_path.write_text(_render_report(summary), encoding="utf-8")
    summary["outputs"] = {
        "matrix_csv": {
            "path": _relative(project_dir, matrix_path),
            "sha256": file_sha256(matrix_path),
        },
        "report": {
            "path": _relative(project_dir, report_path),
            "sha256": file_sha256(report_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
