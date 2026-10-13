from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from .data import DatasetSpec, build_training_data_bundle
from .metrics import binary_classification_metrics, query_level_classification_metrics
from .train import evaluate_model, load_model_from_checkpoint


PARTITIONS = ("train", "validation", "test")
METRIC_NAMES = ("pooled_auroc", "pooled_auprc", "macro_auroc", "macro_auprc")
PREDICTION_COLUMNS = (
    "sample_id",
    "source",
    "tier",
    "strategy",
    "negative_strategy",
    "split",
    "pmhc_id",
    "label",
    "logit",
    "probability",
)
OOF_PREDICTION_COLUMNS = ("fold_id",) + PREDICTION_COLUMNS
OOF_QUERY_COLUMNS = (
    "fold_id",
    "query_id",
    "epitope",
    "hla",
    "positive_count",
    "negative_count",
    "sample_count",
    "auroc",
    "auprc",
)
SUMMARY_COLUMNS = (
    "record_type",
    "fold_id",
    "best_epoch",
    "sample_count",
    "query_count",
    *METRIC_NAMES,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or ()), rows


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


def validate_query_test_rotation(
    manifest: Mapping[str, Any], fold_ids: Sequence[str]
) -> dict[str, Any]:
    if int(manifest.get("fold_count", -1)) != len(fold_ids):
        raise ValueError("manifest fold count mismatch")
    manifest_folds = manifest.get("folds", {})
    if set(manifest_folds) != set(fold_ids):
        raise ValueError("configured folds do not match manifest folds")

    reference_universe: set[str] | None = None
    test_counts: Counter[str] = Counter()
    fold_test_queries: dict[str, list[str]] = {}
    for fold_id in fold_ids:
        partition_sets: dict[str, set[str]] = {}
        for partition in PARTITIONS:
            entry = manifest_folds[fold_id]["partitions"][partition]
            query_ids = [row["query_id"] for row in entry["query_statistics"]]
            if len(query_ids) != len(set(query_ids)):
                raise ValueError(f"duplicate {partition} query in {fold_id}")
            if len(query_ids) != int(entry["audit"]["query_count"]):
                raise ValueError(f"query count mismatch for {fold_id} {partition}")
            partition_sets[partition] = set(query_ids)
        for left_index, left in enumerate(PARTITIONS):
            for right in PARTITIONS[left_index + 1 :]:
                if partition_sets[left] & partition_sets[right]:
                    raise ValueError(f"query leakage in {fold_id}: {left} vs {right}")
        universe = set().union(*partition_sets.values())
        if reference_universe is None:
            reference_universe = universe
        elif universe != reference_universe:
            raise ValueError(f"query universe mismatch in {fold_id}")
        fold_test_queries[fold_id] = sorted(partition_sets["test"])
        test_counts.update(partition_sets["test"])

    assert reference_universe is not None
    missing = sorted(reference_universe - set(test_counts))
    repeated = sorted(query_id for query_id, count in test_counts.items() if count != 1)
    if missing or repeated or set(test_counts) != reference_universe:
        raise ValueError("each query must appear as test exactly once")
    return {
        "query_count": len(reference_universe),
        "all_queries_tested_exactly_once": True,
        "test_occurrence_minimum": min(test_counts.values()),
        "test_occurrence_maximum": max(test_counts.values()),
        "fold_test_query_counts": {
            fold_id: len(query_ids) for fold_id, query_ids in fold_test_queries.items()
        },
        "fold_test_queries": fold_test_queries,
    }


def validate_query_test_rotation_from_inputs(
    project_dir: Path,
    manifest: Mapping[str, Any],
    fold_ids: Sequence[str],
    *,
    positive_manifest_key: str,
) -> dict[str, Any]:
    if int(manifest.get("fold_count", -1)) != len(fold_ids):
        raise ValueError("manifest fold count mismatch")
    manifest_folds = manifest.get("folds", {})
    if set(manifest_folds) != set(fold_ids):
        raise ValueError("configured folds do not match manifest folds")

    reference_universe: set[str] | None = None
    test_counts: Counter[str] = Counter()
    fold_test_queries: dict[str, list[str]] = {}
    for fold_id in fold_ids:
        partition_sets: dict[str, set[str]] = {}
        for partition in PARTITIONS:
            entry = manifest_folds[fold_id]["partitions"][partition]
            output = entry[positive_manifest_key]
            path = _resolve(project_dir, output["path"])
            if file_sha256(path) != output["sha256"]:
                raise ValueError(
                    f"positive input SHA-256 mismatch for {fold_id} {partition}"
                )
            _, rows = _read_csv(path)
            if len(rows) != int(output["rows"]):
                raise ValueError(f"positive input row mismatch for {fold_id} {partition}")
            query_ids = {
                str(row.get("pmhc_id") or f"{row['epitope']}|{row['hla']}").upper()
                for row in rows
            }
            expected_query_count = entry.get("audit", {}).get(
                "query_count", entry.get("audit", {}).get("unique_query_count")
            )
            if expected_query_count is not None and len(query_ids) != int(
                expected_query_count
            ):
                raise ValueError(f"query count mismatch for {fold_id} {partition}")
            partition_sets[partition] = query_ids
        for left_index, left in enumerate(PARTITIONS):
            for right in PARTITIONS[left_index + 1 :]:
                if partition_sets[left] & partition_sets[right]:
                    raise ValueError(f"query leakage in {fold_id}: {left} vs {right}")
        universe = set().union(*partition_sets.values())
        if reference_universe is None:
            reference_universe = universe
        elif universe != reference_universe:
            raise ValueError(f"query universe mismatch in {fold_id}")
        fold_test_queries[fold_id] = sorted(partition_sets["test"])
        test_counts.update(partition_sets["test"])

    assert reference_universe is not None
    if set(test_counts) != reference_universe or any(
        count != 1 for count in test_counts.values()
    ):
        raise ValueError("each query must appear as test exactly once")
    return {
        "query_count": len(reference_universe),
        "all_queries_tested_exactly_once": True,
        "test_occurrence_minimum": min(test_counts.values()),
        "test_occurrence_maximum": max(test_counts.values()),
        "fold_test_query_counts": {
            fold_id: len(query_ids) for fold_id, query_ids in fold_test_queries.items()
        },
        "fold_test_queries": fold_test_queries,
        "source": "positive_input_files",
    }


def _verify_partition_inputs(
    *,
    project_dir: Path,
    fold_id: str,
    manifest_fold: Mapping[str, Any],
    recorded: Mapping[str, Any],
    positive_manifest_key: str,
) -> tuple[dict[str, Any], dict[str, tuple[list[dict[str, str]], list[dict[str, str]]]]]:
    audit: dict[str, Any] = {}
    rows_by_partition = {}
    for partition in PARTITIONS:
        manifest_partition = manifest_fold["partitions"][partition]
        recorded_partition = recorded["partitions"][partition]
        kind_rows = {}
        partition_audit = {}
        for kind, output_key, hash_key, count_key in (
            ("positive", positive_manifest_key, "positive_sha256", "positive_count"),
            ("negative", "negative_output", "negative_sha256", "negative_count"),
        ):
            output = manifest_partition[output_key]
            path = _resolve(project_dir, output["path"])
            observed_hash = file_sha256(path)
            if observed_hash != output["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {fold_id} {partition} {kind}")
            if observed_hash != recorded_partition[hash_key]:
                raise ValueError(f"metrics input hash mismatch for {fold_id} {partition} {kind}")
            _, rows = _read_csv(path)
            if len(rows) != int(output["rows"]) or len(rows) != int(
                recorded_partition[count_key]
            ):
                raise ValueError(f"row count mismatch for {fold_id} {partition} {kind}")
            kind_rows[kind] = rows
            partition_audit[kind] = {
                "path": _relative(project_dir, path),
                "sha256": observed_hash,
                "rows": len(rows),
            }
        rows_by_partition[partition] = (kind_rows["positive"], kind_rows["negative"])
        audit[partition] = partition_audit
    return audit, rows_by_partition


def _checkpoint_prediction_audit(
    *,
    project_dir: Path,
    fold_id: str,
    result_directory: Path,
    fold_config_path: Path,
    expected_spec: Mapping[str, str],
    metrics: Mapping[str, Any],
    stored_predictions: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    fold_config = _load_yaml(fold_config_path)
    config_spec = {
        "tier": str(fold_config["experiment"]["tier"]),
        "positive_split_strategy": str(
            fold_config["experiment"]["positive_split_strategy"]
        ),
        "negative_strategy": str(fold_config["experiment"]["negative_strategy"]),
    }
    if config_spec != dict(expected_spec):
        raise ValueError(f"training config data specification mismatch for {fold_id}")
    if fold_config["experiment"].get("fold_id") != fold_id:
        raise ValueError(f"training config fold mismatch for {fold_id}")

    spec = DatasetSpec(**config_spec)
    bundle = build_training_data_bundle(fold_config, project_dir, spec)
    checkpoint_path = result_directory / "best_model.pt"
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, torch.device("cpu"))
    if checkpoint.get("data_spec") != dict(expected_spec):
        raise ValueError(f"checkpoint data specification mismatch for {fold_id}")
    if int(checkpoint.get("best_epoch", -1)) != int(metrics["best_epoch"]):
        raise ValueError(f"checkpoint best epoch mismatch for {fold_id}")
    if int(checkpoint.get("seed", -1)) != int(metrics["seed"]):
        raise ValueError(f"checkpoint seed mismatch for {fold_id}")
    if checkpoint.get("early_stopping_metric") != metrics.get(
        "early_stopping_metric"
    ):
        raise ValueError(f"checkpoint selection metric mismatch for {fold_id}")
    if checkpoint.get("hla_to_id") != bundle.hla_vocab.hla_to_id:
        raise ValueError(f"checkpoint HLA mapping mismatch for {fold_id}")
    if checkpoint.get("amino_acid_vocab", {}).get(
        "token_to_id"
    ) != bundle.amino_acid_vocab.token_to_id:
        raise ValueError(f"checkpoint amino-acid vocabulary mismatch for {fold_id}")

    expected_model_config = {
        "hla_vocab_size": len(bundle.hla_vocab),
        "aa_vocab_size": int(fold_config["model"]["aa_vocab_size"]),
        "aa_embedding_dim": int(fold_config["model"]["aa_embedding_dim"]),
        "cnn_channels": int(fold_config["model"]["cnn_channels"]),
        "hla_embedding_dim": int(fold_config["model"]["hla_embedding_dim"]),
        "kernel_sizes": list(fold_config["model"]["kernel_sizes"]),
        "mlp_hidden_dims": list(fold_config["model"]["mlp_hidden_dims"]),
        "padding_idx": int(fold_config["model"]["padding_idx"]),
        "dropout": float(fold_config["model"]["dropout"]),
    }
    if checkpoint.get("model_config") != expected_model_config:
        raise ValueError(f"checkpoint model configuration mismatch for {fold_id}")
    for metric in ("loss", "auroc", "auprc"):
        _assert_close(
            float(checkpoint["validation_metrics"][metric]),
            float(metrics["validation"]["pooled"][metric]),
            f"{fold_id} checkpoint validation {metric}",
        )
    if int(checkpoint["validation_metrics"]["sample_count"]) != int(
        metrics["validation"]["pooled"]["sample_count"]
    ):
        raise ValueError(f"checkpoint validation sample count mismatch for {fold_id}")
    for metric in ("auroc", "auprc"):
        _assert_close(
            float(checkpoint["validation_query_macro_metrics"][metric]),
            float(metrics["validation"]["query_macro"][metric]),
            f"{fold_id} checkpoint validation macro {metric}",
        )

    reproduced = evaluate_model(model, bundle.dataloaders["test"], torch.device("cpu"))
    if len(reproduced.predictions) != len(stored_predictions):
        raise ValueError(f"checkpoint test prediction count mismatch for {fold_id}")
    for row_index, (observed, stored) in enumerate(
        zip(reproduced.predictions, stored_predictions, strict=True)
    ):
        for column in PREDICTION_COLUMNS:
            if column in {"logit", "probability"}:
                if not math.isclose(
                    float(observed[column]),
                    float(stored[column]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"checkpoint prediction mismatch for {fold_id} row {row_index} {column}"
                    )
            elif str(observed[column]) != str(stored[column]):
                raise ValueError(
                    f"checkpoint metadata mismatch for {fold_id} row {row_index} {column}"
                )

    return {
        "path": _relative(project_dir, checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "training_config": {
            "path": _relative(project_dir, fold_config_path),
            "sha256": file_sha256(fold_config_path),
        },
        "best_epoch_matches_metrics": True,
        "validation_metrics_match": True,
        "test_predictions_reproduced": True,
        "reproduced_test_rows": len(reproduced.predictions),
    }


def _audit_fold(
    *,
    project_dir: Path,
    results_directory: Path,
    fold_id: str,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    expected_test_queries: set[str],
    expected_spec: Mapping[str, str],
    validation_audit_key: str,
    positive_manifest_key: str,
    fold_config_path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    result_directory = results_directory / fold_id
    metrics_path = result_directory / "metrics.json"
    predictions_path = result_directory / "test_predictions.csv"
    per_query_path = result_directory / "per_query_metrics.csv"
    metrics = _load_json(metrics_path)
    if metrics.get("data_spec") != dict(expected_spec):
        raise ValueError(f"data specification mismatch for {fold_id}")
    if metrics.get("early_stopping_metric") != "validation_query_macro_auprc":
        raise ValueError(f"unexpected stored model-selection metric for {fold_id}")

    recorded = metrics.get("data_validation", {}).get(validation_audit_key, {})
    if recorded.get("fold_id") != fold_id:
        raise ValueError(f"recorded fold ID mismatch for {fold_id}")
    if recorded.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"recorded manifest SHA-256 mismatch for {fold_id}")
    input_audit, partition_rows = _verify_partition_inputs(
        project_dir=project_dir,
        fold_id=fold_id,
        manifest_fold=manifest["folds"][fold_id],
        recorded=recorded,
        positive_manifest_key=positive_manifest_key,
    )

    prediction_columns, predictions = _read_csv(predictions_path)
    if tuple(prediction_columns) != PREDICTION_COLUMNS:
        raise ValueError(f"prediction schema mismatch for {fold_id}")
    positive_rows, negative_rows = partition_rows["test"]
    expected_ids = [row["record_id"] for row in positive_rows] + [
        row["negative_id"] for row in negative_rows
    ]
    expected_labels = ["1"] * len(positive_rows) + ["0"] * len(negative_rows)
    expected_queries = [row["pmhc_id"] for row in positive_rows + negative_rows]
    if [row["sample_id"] for row in predictions] != expected_ids:
        raise ValueError(f"test prediction sample order mismatch for {fold_id}")
    if [row["label"] for row in predictions] != expected_labels:
        raise ValueError(f"test prediction labels mismatch for {fold_id}")
    if [row["pmhc_id"] for row in predictions] != expected_queries:
        raise ValueError(f"test prediction query order mismatch for {fold_id}")
    if any(row["split"] != "test" for row in predictions):
        raise ValueError(f"non-test row found in predictions for {fold_id}")

    labels = [int(row["label"]) for row in predictions]
    probabilities = [float(row["probability"]) for row in predictions]
    pooled = binary_classification_metrics(labels, probabilities)
    query_macro, computed_per_query = query_level_classification_metrics(predictions)
    for metric in ("auroc", "auprc"):
        _assert_close(
            pooled[metric], float(metrics["test"]["pooled"][metric]), f"{fold_id} pooled {metric}"
        )
        _assert_close(
            float(query_macro[metric]),
            float(metrics["test"]["query_macro"][metric]),
            f"{fold_id} macro {metric}",
        )

    _, recorded_per_query = _read_csv(per_query_path)
    if any(row["partition"] not in {"validation", "test"} for row in recorded_per_query):
        raise ValueError(f"unexpected partition in per-query metrics for {fold_id}")
    recorded_test = [row for row in recorded_per_query if row["partition"] == "test"]
    recorded_index = {row["query_id"]: row for row in recorded_test}
    if len(recorded_index) != len(recorded_test) or set(recorded_index) != expected_test_queries:
        raise ValueError(f"test per-query set mismatch for {fold_id}")
    for row in computed_per_query:
        recorded_row = recorded_index[row["query_id"]]
        for count_name in ("positive_count", "negative_count", "sample_count"):
            if int(recorded_row[count_name]) != int(row[count_name]):
                raise ValueError(f"{fold_id} per-query {count_name} mismatch")
        for metric in ("auroc", "auprc"):
            _assert_close(
                float(row[metric]),
                float(recorded_row[metric]),
                f"{fold_id} {row['query_id']} {metric}",
            )

    fold_result = {
        "fold_id": fold_id,
        "best_epoch": int(metrics["best_epoch"]),
        "sample_count": len(predictions),
        "query_count": int(query_macro["query_count"]),
        "pooled_auroc": pooled["auroc"],
        "pooled_auprc": pooled["auprc"],
        "macro_auroc": float(query_macro["auroc"]),
        "macro_auprc": float(query_macro["auprc"]),
    }
    oof_predictions = [{"fold_id": fold_id, **row} for row in predictions]
    oof_queries = [{"fold_id": fold_id, **row} for row in computed_per_query]
    artifact_audit = {
        "metrics": {
            "path": _relative(project_dir, metrics_path),
            "sha256": file_sha256(metrics_path),
        },
        "test_predictions": {
            "path": _relative(project_dir, predictions_path),
            "sha256": file_sha256(predictions_path),
        },
        "per_query_metrics": {
            "path": _relative(project_dir, per_query_path),
            "sha256": file_sha256(per_query_path),
        },
        "partitions": input_audit,
    }
    if fold_config_path is not None:
        artifact_audit["checkpoint"] = _checkpoint_prediction_audit(
            project_dir=project_dir,
            fold_id=fold_id,
            result_directory=result_directory,
            fold_config_path=fold_config_path,
            expected_spec=expected_spec,
            metrics=metrics,
            stored_predictions=predictions,
        )
    return fold_result, oof_predictions, oof_queries, artifact_audit


def _quantile_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        values, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear"
    )
    return float(lower), float(upper)


def query_bootstrap_confidence_intervals(
    predictions: Sequence[Mapping[str, Any]],
    per_query_metrics: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    iterations: int,
    confidence_level: float,
    batch_size: int = 128,
) -> dict[str, Any]:
    if iterations <= 0 or batch_size <= 0:
        raise ValueError("bootstrap iterations and batch size must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between zero and one")
    query_ids = sorted(str(row["query_id"]) for row in per_query_metrics)
    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError("bootstrap queries must be non-empty and unique")
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    per_query_index = {str(row["query_id"]): row for row in per_query_metrics}
    prediction_queries = [str(row["pmhc_id"]).upper() for row in predictions]
    if set(prediction_queries) != set(query_ids):
        raise ValueError("prediction and per-query query sets do not match")

    query_count = len(query_ids)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, query_count, size=(iterations, query_count))
    counts = np.zeros((iterations, query_count), dtype=np.int16)
    np.add.at(
        counts,
        (np.repeat(np.arange(iterations), query_count), draws.reshape(-1)),
        1,
    )

    macro_replicates = {}
    for metric in ("auroc", "auprc"):
        values = np.asarray(
            [float(per_query_index[query_id][metric]) for query_id in query_ids],
            dtype=np.float64,
        )
        macro_replicates[metric] = counts @ values / query_count

    labels = np.asarray([int(row["label"]) for row in predictions], dtype=np.float64)
    probabilities = np.asarray(
        [float(row["probability"]) for row in predictions], dtype=np.float64
    )
    row_query_indices = np.asarray(
        [query_index[query_id] for query_id in prediction_queries], dtype=np.int64
    )
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]
    sorted_query_indices = row_query_indices[order]
    group_starts = np.r_[
        0, np.flatnonzero(sorted_probabilities[1:] != sorted_probabilities[:-1]) + 1
    ]
    pooled_replicates = {
        "auroc": np.empty(iterations, dtype=np.float64),
        "auprc": np.empty(iterations, dtype=np.float64),
    }
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        row_weights = counts[start:stop, sorted_query_indices].astype(np.float64)
        positive_by_score = np.add.reduceat(
            row_weights * sorted_labels, group_starts, axis=1
        )
        negative_by_score = np.add.reduceat(
            row_weights * (1.0 - sorted_labels), group_starts, axis=1
        )
        cumulative_positive = np.cumsum(positive_by_score, axis=1)
        cumulative_negative = np.cumsum(negative_by_score, axis=1)
        total_positive = cumulative_positive[:, -1]
        total_negative = cumulative_negative[:, -1]
        precision = np.divide(
            cumulative_positive,
            cumulative_positive + cumulative_negative,
            out=np.zeros_like(cumulative_positive),
            where=(cumulative_positive + cumulative_negative) > 0,
        )
        pooled_replicates["auprc"][start:stop] = (
            np.sum(positive_by_score * precision, axis=1) / total_positive
        )
        negative_below = total_negative[:, None] - cumulative_negative
        pooled_replicates["auroc"][start:stop] = np.sum(
            positive_by_score * (negative_below + 0.5 * negative_by_score), axis=1
        ) / (total_positive * total_negative)

    estimates = binary_classification_metrics(labels.astype(np.int64), probabilities)
    result: dict[str, Any] = {
        "seed": seed,
        "iterations": iterations,
        "confidence_level": confidence_level,
        "resampling_unit": "query",
        "query_count": query_count,
        "pooled": {},
        "query_macro": {},
    }
    for metric in ("auroc", "auprc"):
        pooled_lower, pooled_upper = _quantile_interval(
            pooled_replicates[metric], confidence_level
        )
        macro_lower, macro_upper = _quantile_interval(
            macro_replicates[metric], confidence_level
        )
        result["pooled"][metric] = {
            "estimate": estimates[metric],
            "ci_lower": pooled_lower,
            "ci_upper": pooled_upper,
        }
        result["query_macro"][metric] = {
            "estimate": float(
                np.mean([float(row[metric]) for row in per_query_metrics])
            ),
            "ci_lower": macro_lower,
            "ci_upper": macro_upper,
        }
    return result


def _five_fold_statistics(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statistics = {}
    for metric in METRIC_NAMES:
        values = np.asarray([float(row[metric]) for row in fold_results], dtype=np.float64)
        statistics[metric] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return statistics


def _summary_csv_rows(
    fold_results: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    oof: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [{"record_type": "fold", **row} for row in fold_results]
    for statistic_name, key in (
        ("five_fold_mean", "mean"),
        ("five_fold_standard_deviation", "standard_deviation"),
        ("five_fold_minimum", "minimum"),
        ("five_fold_maximum", "maximum"),
    ):
        rows.append(
            {
                "record_type": statistic_name,
                **{metric: statistics[metric][key] for metric in METRIC_NAMES},
            }
        )
    bootstrap = oof["bootstrap"]
    for record_type, ci_key in (
        ("oof_estimate", "estimate"),
        ("oof_ci_lower", "ci_lower"),
        ("oof_ci_upper", "ci_upper"),
    ):
        rows.append(
            {
                "record_type": record_type,
                "sample_count": oof["sample_count"],
                "query_count": oof["query_count"],
                "pooled_auroc": bootstrap["pooled"]["auroc"][ci_key],
                "pooled_auprc": bootstrap["pooled"]["auprc"][ci_key],
                "macro_auroc": bootstrap["query_macro"]["auroc"][ci_key],
                "macro_auprc": bootstrap["query_macro"]["auprc"][ci_key],
            }
        )
    return rows


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# {summary['report_title']}",
        "",
        "All five stored best-model test outputs were included. No model reselection or training was performed.",
        "",
        "## Input audit",
        "",
        f"- Manifest SHA-256: `{summary['audit']['manifest']['sha256']}`",
        f"- Queries tested exactly once: {summary['audit']['query_rotation']['query_count']}",
        "- Fold input hashes: passed",
        "- Selection audit: stored validation query-level macro AUPRC selection only; no summary-time selection",
        "",
        "## Fold results",
        "",
        "| Fold | Best epoch | Pooled AUROC | Pooled AUPRC | Macro AUROC | Macro AUPRC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["fold_results"]:
        lines.append(
            f"| {row['fold_id']} | {row['best_epoch']} | {row['pooled_auroc']:.6f} | "
            f"{row['pooled_auprc']:.6f} | {row['macro_auroc']:.6f} | {row['macro_auprc']:.6f} |"
        )
    lines.extend(
        ["", "## Five-fold distribution", "", "Standard deviations are sample standard deviations (ddof=1).", ""]
    )
    for metric in METRIC_NAMES:
        item = summary["five_fold_statistics"][metric]
        lines.append(
            f"- {metric}: {item['mean']:.6f} +/- {item['standard_deviation']:.6f} "
            f"(min {item['minimum']:.6f}, max {item['maximum']:.6f})"
        )
    lines.extend(["", "## Out-of-fold metrics", ""])
    bootstrap = summary["oof"]["bootstrap"]
    for scope in ("pooled", "query_macro"):
        for metric in ("auroc", "auprc"):
            item = bootstrap[scope][metric]
            lines.append(
                f"- {scope} {metric}: {item['estimate']:.6f} "
                f"(95% CI {item['ci_lower']:.6f} to {item['ci_upper']:.6f})"
            )
    lines.extend(
        [
            "",
            f"Confidence intervals use {bootstrap['iterations']:,} fixed-seed bootstrap resamples with query as the resampling unit.",
            "",
        ]
    )
    return "\n".join(lines)


def build_five_fold_summary(
    config_path: Path, project_dir: Path, *, force: bool = False
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    fold_ids = tuple(config["fold_ids"])
    experiment = config.get(
        "experiment",
        {
            "tier": "development",
            "positive_split_strategy": "distance_aware",
            "negative_strategy": "N1",
            "validation_audit_key": "query_balanced_5fold_inputs",
            "manifest_positive_key": "positive_output",
            "query_rotation_source": "manifest_statistics",
        },
    )
    expected_spec = {
        "tier": str(experiment["tier"]),
        "positive_split_strategy": str(experiment["positive_split_strategy"]),
        "negative_strategy": str(experiment["negative_strategy"]),
    }
    validation_audit_key = str(experiment["validation_audit_key"])
    positive_manifest_key = str(experiment["manifest_positive_key"])
    manifest_path = _resolve(project_dir, config["inputs"]["manifest"])
    results_directory = _resolve(project_dir, config["inputs"]["results_directory"])
    configured_fold_configs = config["inputs"].get("fold_configs", {})
    if configured_fold_configs and set(configured_fold_configs) != set(fold_ids):
        raise ValueError("fold training configs do not match configured folds")
    output_directory = _resolve(project_dir, config["outputs"]["directory"])
    output_paths = {
        name: output_directory / config["outputs"][name]
        for name in (
            "summary_csv",
            "summary_json",
            "oof_predictions_csv",
            "oof_per_query_csv",
            "report_md",
        )
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(f"summary outputs already exist: {existing}")
    output_directory.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(manifest_path)
    manifest_sha = file_sha256(manifest_path)
    if experiment["query_rotation_source"] == "positive_input_files":
        rotation_audit = validate_query_test_rotation_from_inputs(
            project_dir,
            manifest,
            fold_ids,
            positive_manifest_key=positive_manifest_key,
        )
    elif experiment["query_rotation_source"] == "manifest_statistics":
        rotation_audit = validate_query_test_rotation(manifest, fold_ids)
    else:
        raise ValueError("unsupported query rotation source")
    fold_results = []
    oof_predictions: list[dict[str, str]] = []
    oof_queries: list[dict[str, Any]] = []
    fold_audits = {}
    for fold_id in fold_ids:
        result, predictions, queries, audit = _audit_fold(
            project_dir=project_dir,
            results_directory=results_directory,
            fold_id=fold_id,
            manifest=manifest,
            manifest_sha256=manifest_sha,
            expected_test_queries=set(rotation_audit["fold_test_queries"][fold_id]),
            expected_spec=expected_spec,
            validation_audit_key=validation_audit_key,
            positive_manifest_key=positive_manifest_key,
            fold_config_path=(
                _resolve(project_dir, configured_fold_configs[fold_id])
                if configured_fold_configs
                else None
            ),
        )
        fold_results.append(result)
        oof_predictions.extend(predictions)
        oof_queries.extend(queries)
        fold_audits[fold_id] = audit

    if len({row["sample_id"] for row in oof_predictions}) != len(oof_predictions):
        raise ValueError("duplicate sample ID in OOF predictions")
    if len({row["query_id"] for row in oof_queries}) != len(oof_queries):
        raise ValueError("duplicate query in OOF per-query metrics")
    if len(oof_queries) != rotation_audit["query_count"]:
        raise ValueError("OOF query count does not match manifest rotation")

    pooled = binary_classification_metrics(
        [int(row["label"]) for row in oof_predictions],
        [float(row["probability"]) for row in oof_predictions],
    )
    query_macro, recomputed_queries = query_level_classification_metrics(oof_predictions)
    recomputed_index = {row["query_id"]: row for row in recomputed_queries}
    for row in oof_queries:
        recomputed = recomputed_index[row["query_id"]]
        for metric in ("auroc", "auprc"):
            _assert_close(
                float(row[metric]), float(recomputed[metric]), f"OOF {row['query_id']} {metric}"
            )
    bootstrap_config = config["bootstrap"]
    bootstrap = query_bootstrap_confidence_intervals(
        oof_predictions,
        oof_queries,
        seed=int(bootstrap_config["seed"]),
        iterations=int(bootstrap_config["iterations"]),
        confidence_level=float(bootstrap_config["confidence_level"]),
        batch_size=int(bootstrap_config["batch_size"]),
    )
    oof = {
        "sample_count": len(oof_predictions),
        "query_count": int(query_macro["query_count"]),
        "pooled": pooled,
        "query_macro": {
            "auroc": float(query_macro["auroc"]),
            "auprc": float(query_macro["auprc"]),
        },
        "bootstrap": bootstrap,
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "report_title": config.get(
            "report_title",
            "Development distance-aware 5-fold query-balanced N1 summary",
        ),
        "experiment": expected_spec,
        "audit": {
            "status": "passed",
            "manifest": {
                "path": _relative(project_dir, manifest_path),
                "sha256": manifest_sha,
            },
            "query_rotation": rotation_audit,
            "fold_inputs": fold_audits,
            "model_selection": {
                "summary_time_reselection_performed": False,
                "validation_metrics_used_for_aggregation": False,
                "test_metrics_used_for_selection": False,
                "stored_selection_metric": "validation_query_macro_auprc",
                "all_folds_included": True,
            },
        },
        "fold_results": fold_results,
        "five_fold_statistics": _five_fold_statistics(fold_results),
        "five_fold_statistics_definition": {
            "standard_deviation": "sample",
            "degrees_of_freedom": 1,
        },
        "oof": oof,
    }

    _write_csv(
        output_paths["summary_csv"],
        SUMMARY_COLUMNS,
        _summary_csv_rows(fold_results, summary["five_fold_statistics"], oof),
    )
    _write_csv(output_paths["oof_predictions_csv"], OOF_PREDICTION_COLUMNS, oof_predictions)
    _write_csv(output_paths["oof_per_query_csv"], OOF_QUERY_COLUMNS, oof_queries)
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
