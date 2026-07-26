from __future__ import annotations

import json
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .degree_matched_additive_logistic import (
    additive_feature_dimensions,
    build_additive_feature_matrix,
)
from .degree_matched_orthogonalized_logistic import (
    _build_feature_matrix_and_audit,
    orthogonalized_feature_dimensions,
)
from .degree_matched_tcr_only_logistic import (
    _normalize,
    _query_key,
    _validate_query_rotation,
    audit_partition_degrees,
)
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
    fit_hla_mapping,
)
from .tcrue.negative_sampling import load_positive_registry


RESAMPLE_IDS = tuple(f"resample_{index}" for index in range(1, 6))
METHODS = ("additive_only", "orthogonalized_interaction")
COMPARISON = "orthogonalized_minus_additive_only"
OOF_PREDICTION_COLUMNS = ("resample_id", "method", *PREDICTION_COLUMNS)
OOF_QUERY_COLUMNS = ("resample_id", "method", *PER_QUERY_COLUMNS)


def _interaction_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _normalize(row["tcr_cdr3_beta"]),
        _normalize(row["epitope"]),
        _normalize(row["hla"]),
    )


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
                if row.get("negative_strategy") != "degree_matched_resample":
                    raise ValueError(
                        f"unexpected negative strategy in {fold_id}/{partition}"
                    )
                if row.get("fold_id") != fold_id or row.get("partition") != partition:
                    raise ValueError(
                        f"negative fold/partition mismatch: {fold_id}/{partition}"
                    )
            rows.append(
                {
                    "sample_id": row[identifier],
                    "source": source,
                    "tier": "development",
                    "strategy": "distance_aware",
                    "negative_strategy": "degree_matched_resample",
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


def load_and_audit_resample_inputs(
    project_dir: Path,
    manifest: Mapping[str, Any],
    resample_id: str,
    fold_ids: Sequence[str],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, list[dict[str, Any]]]],
    dict[str, Any],
    dict[str, Any],
]:
    if resample_id not in RESAMPLE_IDS:
        raise ValueError(f"unsupported resample: {resample_id}")
    invariants = manifest.get("invariants", {})
    required_invariants = {
        "all_outputs_exact": True,
        "all_tcr_degrees_exact": True,
        "all_query_degrees_exact": True,
        "known_positive_registry_intersections": 0,
        "duplicate_negative_interactions": 0,
        "partition_leakage": 0,
    }
    for field, expected in required_invariants.items():
        if invariants.get(field) != expected:
            raise ValueError(f"resample manifest invariant failed: {field}")

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
    leakage_audit: dict[str, Any] = {}
    for fold_id in fold_ids:
        input_audit[fold_id] = {}
        rows_by_fold[fold_id] = {}
        partition_keys: dict[str, set[tuple[str, str, str]]] = {}
        partition_queries: dict[str, set[tuple[str, str]]] = {}
        for partition in PARTITIONS:
            entry = manifest["folds"][fold_id]["partitions"][partition]
            resample_entry = entry["resamples"][resample_id]
            if resample_entry.get("exact_solution") is not True:
                raise ValueError(
                    f"non-exact resample entry: {resample_id}/{fold_id}/{partition}"
                )
            sources = {
                "positive": entry["positive_input"],
                "negative": resample_entry["negative_output"],
            }
            kind_rows: dict[str, list[dict[str, str]]] = {}
            partition_audit: dict[str, Any] = {}
            for kind, source in sources.items():
                path = _resolve(project_dir, source["path"])
                observed_sha = file_sha256(path)
                if observed_sha != source["sha256"]:
                    raise ValueError(
                        f"SHA-256 mismatch: {resample_id}/{fold_id}/{partition}/{kind}"
                    )
                rows = _read_csv(path)
                if len(rows) != int(source["rows"]):
                    raise ValueError(
                        f"row count mismatch: {resample_id}/{fold_id}/{partition}/{kind}"
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
            required_degree_fields = {
                "tcr_degree_exact": True,
                "query_degree_exact": True,
                "known_positive_registry_intersections": 0,
                "duplicate_negative_interactions": 0,
                "negative_tcrs_outside_partition": 0,
                "negative_queries_outside_partition": 0,
            }
            for field, expected in required_degree_fields.items():
                if degree_audit.get(field) != expected:
                    raise ValueError(
                        f"degree audit failed: {resample_id}/{fold_id}/{partition}/{field}"
                    )
            fixed = _fixed_rows(
                kind_rows["positive"],
                kind_rows["negative"],
                fold_id=fold_id,
                partition=partition,
            )
            rows_by_fold[fold_id][partition] = fixed
            input_audit[fold_id][partition] = {
                **partition_audit,
                "degree_audit": degree_audit,
                "manifest_audit": resample_entry["audit"],
                "derived_seed": resample_entry["derived_seed"],
            }
            partition_keys[partition] = {_interaction_key(row) for row in fixed}
            partition_queries[partition] = {_query_key(row) for row in fixed}

        key_intersections = {}
        query_intersections = {}
        for left, right in combinations(PARTITIONS, 2):
            key_intersections[f"{left}_vs_{right}"] = len(
                partition_keys[left] & partition_keys[right]
            )
            query_intersections[f"{left}_vs_{right}"] = len(
                partition_queries[left] & partition_queries[right]
            )
        if any(key_intersections.values()) or any(query_intersections.values()):
            raise ValueError(f"partition leakage: {resample_id}/{fold_id}")
        leakage_audit[fold_id] = {
            "interaction_intersections": key_intersections,
            "query_intersections": query_intersections,
        }

    rotation_audit = _validate_query_rotation(rows_by_fold, fold_ids)
    registry_audit = {
        "path": _relative(project_dir, registry_path),
        "sha256": registry_sha,
        "interaction_count": len(positive_registry),
    }
    return input_audit, rows_by_fold, registry_audit, {
        "query_test_rotation": rotation_audit,
        "cross_partition": leakage_audit,
    }


def _load_locked_reference(
    project_dir: Path, source: Mapping[str, str], description: str
) -> tuple[Path, dict[str, Any]]:
    path = _resolve(project_dir, source["path"])
    if file_sha256(path) != source["sha256"]:
        raise ValueError(f"{description} SHA-256 mismatch")
    return path, _load_json(path) if path.suffix == ".json" else _load_yaml(path)


def _validate_fixed_specification(
    project_dir: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    references = config["references"]
    additive_config_path, additive_config = _load_locked_reference(
        project_dir, references["additive_config"], "additive config"
    )
    orthogonalized_config_path, orthogonalized_config = _load_locked_reference(
        project_dir,
        references["orthogonalized_config"],
        "orthogonalized config",
    )
    additive_summary_path, additive_summary = _load_locked_reference(
        project_dir,
        references["resample_0_additive_summary"],
        "resample_0 additive summary",
    )
    orthogonalized_summary_path, orthogonalized_summary = _load_locked_reference(
        project_dir,
        references["resample_0_orthogonalized_summary"],
        "resample_0 orthogonalized summary",
    )
    if config["seed"] != additive_config["seed"] or config["seed"] != orthogonalized_config["seed"]:
        raise ValueError("model seed differs from locked references")
    if config["model"] != additive_config["model"] or config["model"] != orthogonalized_config["model"]:
        raise ValueError("model parameters differ from locked references")
    if config["features"]["additive_only"] != additive_config["features"]:
        raise ValueError("additive-only features differ from locked reference")
    if (
        config["features"]["orthogonalized_interaction"]
        != orthogonalized_config["features"]
    ):
        raise ValueError("orthogonalized features differ from locked reference")
    for description, summary in (
        ("additive", additive_summary),
        ("orthogonalized", orthogonalized_summary),
    ):
        if int(summary["seed"]) != int(config["seed"]):
            raise ValueError(f"resample_0 {description} summary seed mismatch")
    return {
        "additive_config": (additive_config_path, additive_config),
        "orthogonalized_config": (
            orthogonalized_config_path,
            orthogonalized_config,
        ),
        "additive_summary": (additive_summary_path, additive_summary),
        "orthogonalized_summary": (
            orthogonalized_summary_path,
            orthogonalized_summary,
        ),
    }


def _model_from_config(config: Mapping[str, Any], seed: int) -> LogisticRegression:
    expected = {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "penalty": "l2",
        "C": 1.0,
        "class_weight": None,
        "max_iter": 5000,
        "solver": "liblinear",
        "random_state_source": "seed",
    }
    if config != expected:
        raise ValueError("Logistic Regression model configuration changed")
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight=None,
        max_iter=5000,
        solver="liblinear",
        random_state=seed,
    )


def _statistics(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (5,):
        raise ValueError("five-resample statistics require exactly five values")
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def summarize_five_resamples(
    resamples: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    if set(resamples) != set(RESAMPLE_IDS):
        raise ValueError("five resample summaries are required")
    result: dict[str, dict[str, dict[str, float]]] = {}
    for section in (*METHODS, COMPARISON):
        result[section] = {}
        for metric in METRIC_NAMES:
            if section == COMPARISON:
                values = [
                    float(resamples[resample_id]["oof_difference"][metric])
                    for resample_id in RESAMPLE_IDS
                ]
            else:
                values = [
                    float(resamples[resample_id]["methods"][section]["oof_metrics"][metric])
                    for resample_id in RESAMPLE_IDS
                ]
            result[section][metric] = _statistics(values)
    return result


def _train_method(
    *,
    project_dir: Path,
    output_directory: Path,
    resample_id: str,
    method: str,
    rows_by_fold: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    input_audit: Mapping[str, Any],
    fold_ids: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    method_directory = output_directory / resample_id / method
    method_directory.mkdir(parents=True)
    seed = int(config["seed"])
    fold_summaries: dict[str, Any] = {}
    oof_rows: list[dict[str, Any]] = []
    oof_query_rows: list[dict[str, Any]] = []
    for fold_id in fold_ids:
        train_rows = rows_by_fold[fold_id]["train"]
        test_rows = rows_by_fold[fold_id]["test"]
        hla_mapping = fit_hla_mapping(train_rows)
        orthogonality = None
        if method == "additive_only":
            train_features = build_additive_feature_matrix(train_rows, hla_mapping)
            test_features = build_additive_feature_matrix(test_rows, hla_mapping)
            dimensions = additive_feature_dimensions(hla_mapping)
        elif method == "orthogonalized_interaction":
            train_features, train_orthogonality = _build_feature_matrix_and_audit(
                train_rows, hla_mapping
            )
            test_features, test_orthogonality = _build_feature_matrix_and_audit(
                test_rows, hla_mapping
            )
            dimensions = orthogonalized_feature_dimensions(hla_mapping)
            orthogonality = {
                "train": train_orthogonality,
                "test": test_orthogonality,
            }
        else:
            raise ValueError(f"unsupported method: {method}")
        if not isinstance(train_features, csr_matrix) or not isinstance(
            test_features, csr_matrix
        ):
            raise TypeError("feature builders must return CSR matrices")
        train_labels = np.asarray(
            [int(row["label"]) for row in train_rows], dtype=np.int64
        )
        model = _model_from_config(config["model"], seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train_features, train_labels)
        logits = np.asarray(model.decision_function(test_features), dtype=np.float64)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        scored_rows = [
            {
                "resample_id": resample_id,
                "method": method,
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
        query_rows = [
            {"resample_id": resample_id, "method": method, "fold_id": fold_id, **row}
            for row in per_query
        ]
        model_path = method_directory / f"{fold_id}_model.joblib"
        joblib.dump(
            {
                "estimator": model,
                "method": method,
                "resample_id": resample_id,
                "fold_id": fold_id,
                "seed": seed,
                "model_config": config["model"],
                "feature_config": config["features"][method],
                "feature_dimensions": dimensions,
                "hla_mapping": hla_mapping,
                "feature_fit_partitions": ["train"],
                "model_fit_partitions": ["train"],
                "test_evaluation_count": 1,
                "input_audit": input_audit[fold_id],
                "interaction_orthogonality": orthogonality,
            },
            model_path,
            compress=3,
        )
        fold_summaries[fold_id] = {
            "train_sample_count": len(train_rows),
            "validation_sample_count": len(rows_by_fold[fold_id]["validation"]),
            "test_sample_count": len(test_rows),
            "test_query_count": len(per_query),
            "feature_count": int(train_features.shape[1]),
            "known_train_hla_count": len(hla_mapping) - 1,
            "unknown_test_hla_count": sum(
                _normalize(row["hla"]) not in hla_mapping for row in test_rows
            ),
            "iterations": int(model.n_iter_[0]),
            "test_metrics": fold_metrics,
            "test_evaluation_count": 1,
            "interaction_orthogonality": orthogonality,
            "model": {
                "path": _relative(project_dir, model_path),
                "sha256": file_sha256(model_path),
            },
        }
        oof_rows.extend(scored_rows)
        oof_query_rows.extend(query_rows)

    oof_metrics, recomputed_queries = _metrics(oof_rows)
    if len(recomputed_queries) != len(oof_query_rows):
        raise ValueError(f"OOF query IDs are not unique: {resample_id}/{method}")
    oof_path = method_directory / "oof_test_predictions.csv"
    query_path = method_directory / "oof_per_query_metrics.csv"
    _write_csv(oof_path, OOF_PREDICTION_COLUMNS, oof_rows)
    _write_csv(query_path, OOF_QUERY_COLUMNS, oof_query_rows)
    return {
        "folds": fold_summaries,
        "oof_sample_count": len(oof_rows),
        "oof_query_count": len(oof_query_rows),
        "oof_metrics": oof_metrics,
        "artifacts": {
            "oof_predictions": {
                "path": _relative(project_dir, oof_path),
                "sha256": file_sha256(oof_path),
            },
            "oof_per_query_metrics": {
                "path": _relative(project_dir, query_path),
                "sha256": file_sha256(query_path),
            },
        },
    }


def _wide_values(
    additive: Mapping[str, float],
    orthogonalized: Mapping[str, float],
    difference: Mapping[str, float],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in METRIC_NAMES:
        values[f"additive_only_{metric}"] = float(additive[metric])
        values[f"orthogonalized_interaction_{metric}"] = float(
            orthogonalized[metric]
        )
        values[f"orthogonalized_minus_additive_{metric}"] = float(
            difference[metric]
        )
    return values


def _summary_rows(
    resample_0: Mapping[str, Any],
    resamples: Mapping[str, Mapping[str, Any]],
    statistics: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    metric_columns = tuple(
        column
        for metric in METRIC_NAMES
        for column in (
            f"additive_only_{metric}",
            f"orthogonalized_interaction_{metric}",
            f"orthogonalized_minus_additive_{metric}",
        )
    )
    columns = ("row_type", "resample_id", "included_in_five_resample_summary", *metric_columns)
    rows = [
        {
            "row_type": "resample",
            "resample_id": "resample_0",
            "included_in_five_resample_summary": False,
            **_wide_values(
                resample_0["methods"]["additive_only"]["oof_metrics"],
                resample_0["methods"]["orthogonalized_interaction"]["oof_metrics"],
                resample_0["oof_difference"],
            ),
        }
    ]
    for resample_id in RESAMPLE_IDS:
        item = resamples[resample_id]
        rows.append(
            {
                "row_type": "resample",
                "resample_id": resample_id,
                "included_in_five_resample_summary": True,
                **_wide_values(
                    item["methods"]["additive_only"]["oof_metrics"],
                    item["methods"]["orthogonalized_interaction"]["oof_metrics"],
                    item["oof_difference"],
                ),
            }
        )
    statistic_names = ("mean", "standard_deviation", "minimum", "maximum")
    for statistic_name in statistic_names:
        row: dict[str, Any] = {
            "row_type": "five_resample_statistic",
            "resample_id": statistic_name,
            "included_in_five_resample_summary": True,
        }
        for metric in METRIC_NAMES:
            row[f"additive_only_{metric}"] = statistics["additive_only"][metric][statistic_name]
            row[f"orthogonalized_interaction_{metric}"] = statistics["orthogonalized_interaction"][metric][statistic_name]
            row[f"orthogonalized_minus_additive_{metric}"] = statistics[COMPARISON][metric][statistic_name]
        rows.append(row)
    return columns, rows


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Degree-matched negative-resample interaction robustness",
        "",
        "Five independent degree-matched negative resamples were evaluated with fixed Additive-only and Orthogonalized-interaction Logistic Regression specifications.",
        "",
        "## OOF results",
        "",
        "| Resample | Orth pooled AUROC | Orth pooled AUPRC | Orth macro AUROC | Orth macro AUPRC | Delta pooled AUROC | Delta pooled AUPRC | Delta macro AUROC | Delta macro AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for resample_id in ("resample_0", *RESAMPLE_IDS):
        item = summary["resample_0"] if resample_id == "resample_0" else summary["resamples"][resample_id]
        orthogonalized = item["methods"]["orthogonalized_interaction"]["oof_metrics"]
        difference = item["oof_difference"]
        lines.append(
            f"| {resample_id} | {orthogonalized['pooled_auroc']:.6f} | "
            f"{orthogonalized['pooled_auprc']:.6f} | {orthogonalized['macro_auroc']:.6f} | "
            f"{orthogonalized['macro_auprc']:.6f} | {difference['pooled_auroc']:+.6f} | "
            f"{difference['pooled_auprc']:+.6f} | {difference['macro_auroc']:+.6f} | "
            f"{difference['macro_auprc']:+.6f} |"
        )
    lines.extend(["", "## Five-resample statistics", ""])
    for section in ("orthogonalized_interaction", COMPARISON):
        lines.append(f"### {section}")
        lines.append("")
        for metric in METRIC_NAMES:
            item = summary["five_resample_statistics"][section][metric]
            lines.append(
                f"- {metric}: {item['mean']:.6f} +/- {item['standard_deviation']:.6f} "
                f"(range {item['minimum']:.6f} to {item['maximum']:.6f})"
            )
        lines.append("")
    lines.extend(
        [
            "## Audit",
            "",
            "All 75 positive/negative input pairs matched the resample manifest and passed exact TCR/query degree, positive-registry conflict, duplicate, and partition-leakage audits. All 50 models used train only; each test set was evaluated once. Resample_0 is shown separately and excluded from the five-resample statistics. No CNN, bootstrap, or permutation test was run.",
            "",
        ]
    )
    return "\n".join(lines)


def train_interaction_resample_robustness(
    project_dir: Path, config_path: Path
) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    fold_ids = [str(value) for value in config["fold_ids"]]
    resample_ids = [str(value) for value in config["resample_ids"]]
    if tuple(fold_ids) != tuple(f"fold_{index}" for index in range(1, 6)):
        raise ValueError("five fixed folds are required")
    if tuple(resample_ids) != RESAMPLE_IDS:
        raise ValueError("resample_1 through resample_5 are required")
    references = _validate_fixed_specification(project_dir, config)

    manifest_source = config["data"]["resample_manifest"]
    manifest_path = _resolve(project_dir, manifest_source["path"])
    manifest_sha = file_sha256(manifest_path)
    if manifest_sha != manifest_source["sha256"]:
        raise ValueError("negative resample manifest SHA-256 mismatch")
    manifest = _load_json(manifest_path)

    audited_inputs = {}
    rows_by_resample = {}
    registry_audit = None
    leakage_audits = {}
    for resample_id in RESAMPLE_IDS:
        input_audit, rows_by_fold, observed_registry, leakage_audit = (
            load_and_audit_resample_inputs(
                project_dir, manifest, resample_id, fold_ids
            )
        )
        if registry_audit is None:
            registry_audit = observed_registry
        elif registry_audit != observed_registry:
            raise ValueError("positive registry audit differs across resamples")
        audited_inputs[resample_id] = input_audit
        rows_by_resample[resample_id] = rows_by_fold
        leakage_audits[resample_id] = leakage_audit

    output_directory = _resolve(project_dir, config["output"]["directory"])
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    resample_results = {}
    for resample_id in RESAMPLE_IDS:
        methods = {}
        for method in METHODS:
            methods[method] = _train_method(
                project_dir=project_dir,
                output_directory=output_directory,
                resample_id=resample_id,
                method=method,
                rows_by_fold=rows_by_resample[resample_id],
                input_audit=audited_inputs[resample_id],
                fold_ids=fold_ids,
                config=config,
            )
        difference = {
            metric: float(methods["orthogonalized_interaction"]["oof_metrics"][metric])
            - float(methods["additive_only"]["oof_metrics"][metric])
            for metric in METRIC_NAMES
        }
        resample_results[resample_id] = {
            "methods": methods,
            "oof_difference": difference,
            "input_audit": audited_inputs[resample_id],
            "leakage_audit": leakage_audits[resample_id],
        }

    additive_reference_path, additive_reference = references["additive_summary"]
    orthogonalized_reference_path, orthogonalized_reference = references[
        "orthogonalized_summary"
    ]
    resample_0 = {
        "included_in_five_resample_statistics": False,
        "methods": {
            "additive_only": {
                "oof_metrics": {
                    metric: float(additive_reference["oof_metrics"][metric])
                    for metric in METRIC_NAMES
                },
                "source": {
                    "path": _relative(project_dir, additive_reference_path),
                    "sha256": config["references"]["resample_0_additive_summary"]["sha256"],
                },
            },
            "orthogonalized_interaction": {
                "oof_metrics": {
                    metric: float(orthogonalized_reference["oof_metrics"][metric])
                    for metric in METRIC_NAMES
                },
                "source": {
                    "path": _relative(project_dir, orthogonalized_reference_path),
                    "sha256": config["references"]["resample_0_orthogonalized_summary"]["sha256"],
                },
            },
        },
    }
    resample_0["oof_difference"] = {
        metric: resample_0["methods"]["orthogonalized_interaction"]["oof_metrics"][metric]
        - resample_0["methods"]["additive_only"]["oof_metrics"][metric]
        for metric in METRIC_NAMES
    }
    statistics = summarize_five_resamples(resample_results)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "seed": int(config["seed"]),
        "model_config": config["model"],
        "feature_configs": config["features"],
        "trained_model_count": len(RESAMPLE_IDS) * len(fold_ids) * len(METHODS),
        "resample_0": resample_0,
        "resamples": resample_results,
        "five_resample_statistics": statistics,
        "audit": {
            "status": "passed",
            "config": {
                "path": _relative(project_dir, config_path),
                "sha256": file_sha256(config_path),
            },
            "resample_manifest": {
                "path": _relative(project_dir, manifest_path),
                "sha256": manifest_sha,
            },
            "positive_registry": registry_audit,
            "all_input_sha256_match": True,
            "all_tcr_degrees_exact": True,
            "all_query_degrees_exact": True,
            "known_positive_registry_intersections": 0,
            "duplicate_interactions": 0,
            "partition_leakage": 0,
            "model_fit_partitions": ["train"],
            "feature_fit_partitions": ["train"],
            "test_evaluations_per_model": 1,
            "hyperparameter_tuning_performed": False,
            "cnn_trained": False,
            "bootstrap_performed": False,
            "permutation_test_performed": False,
        },
    }

    csv_path = output_directory / config["output"]["summary_csv"]
    report_path = output_directory / config["output"]["report_md"]
    json_path = output_directory / config["output"]["summary_json"]
    columns, rows = _summary_rows(resample_0, resample_results, statistics)
    _write_csv(csv_path, columns, rows)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    summary["outputs"] = {
        "summary_csv": {
            "path": _relative(project_dir, csv_path),
            "sha256": file_sha256(csv_path),
        },
        "report": {
            "path": _relative(project_dir, report_path),
            "sha256": file_sha256(report_path),
        },
    }
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
