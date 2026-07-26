from __future__ import annotations

import csv
import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import yaml
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .five_fold_summary import file_sha256, validate_query_test_rotation
from .metrics import binary_classification_metrics, query_level_classification_metrics
from .vocab import STANDARD_AMINO_ACIDS


PARTITIONS = ("train", "validation", "test")
METRIC_NAMES = ("pooled_auroc", "pooled_auprc", "macro_auroc", "macro_auprc")
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
    "logit",
    "probability",
)
PER_QUERY_COLUMNS = (
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
UNK_HLA = "UNK_HLA"


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


def normalized_kmer_frequencies(
    sequence: str, k: int, amino_acids: Sequence[str] = STANDARD_AMINO_ACIDS
) -> dict[int, float]:
    normalized = sequence.strip().upper()
    if not normalized:
        raise ValueError("amino-acid sequence must be non-empty")
    alphabet = tuple(amino_acids)
    invalid = sorted(set(normalized) - set(alphabet))
    if invalid:
        raise ValueError(f"non-standard amino acid(s): {''.join(invalid)}")
    if k <= 0:
        raise ValueError("k must be positive")
    denominator = len(normalized) - k + 1
    if denominator <= 0:
        return {}
    aa_index = {amino_acid: index for index, amino_acid in enumerate(alphabet)}
    counts: Counter[int] = Counter()
    for start in range(denominator):
        index = 0
        for amino_acid in normalized[start : start + k]:
            index = index * len(alphabet) + aa_index[amino_acid]
        counts[index] += 1
    return {index: count / denominator for index, count in counts.items()}


def fit_hla_mapping(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    hlas = sorted({str(row["hla"]).strip().upper() for row in rows})
    if not hlas or "" in hlas or UNK_HLA in hlas:
        raise ValueError("training rows contain an invalid HLA value")
    return {UNK_HLA: 0, **{hla: index + 1 for index, hla in enumerate(hlas)}}


def feature_dimensions(hla_mapping: Mapping[str, int]) -> dict[str, int]:
    alphabet_size = len(STANDARD_AMINO_ACIDS)
    base = 2 * (alphabet_size + alphabet_size**2) + alphabet_size**2 + 2
    if set(hla_mapping.values()) != set(range(len(hla_mapping))):
        raise ValueError("HLA IDs must be contiguous and zero-based")
    return {"sequence_and_interaction": base, "hla": len(hla_mapping), "total": base + len(hla_mapping)}


def build_feature_matrix(
    rows: Sequence[Mapping[str, Any]], hla_mapping: Mapping[str, int]
) -> csr_matrix:
    dimensions = feature_dimensions(hla_mapping)
    alphabet_size = len(STANDARD_AMINO_ACIDS)
    tcr_1_offset = 0
    tcr_2_offset = tcr_1_offset + alphabet_size
    epitope_1_offset = tcr_2_offset + alphabet_size**2
    epitope_2_offset = epitope_1_offset + alphabet_size
    interaction_offset = epitope_2_offset + alphabet_size**2
    length_offset = interaction_offset + alphabet_size**2
    hla_offset = dimensions["sequence_and_interaction"]
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
        tcr_1 = normalized_kmer_frequencies(tcr, 1)
        epitope_1 = normalized_kmer_frequencies(epitope, 1)
        append(row_index, tcr_1_offset, tcr_1)
        append(row_index, tcr_2_offset, normalized_kmer_frequencies(tcr, 2))
        append(row_index, epitope_1_offset, epitope_1)
        append(row_index, epitope_2_offset, normalized_kmer_frequencies(epitope, 2))
        interaction = {
            tcr_index * alphabet_size + epitope_index: tcr_value * epitope_value
            for tcr_index, tcr_value in tcr_1.items()
            for epitope_index, epitope_value in epitope_1.items()
        }
        append(row_index, interaction_offset, interaction)
        append(row_index, length_offset, {0: len(tcr), 1: len(epitope)})
        hla = str(row["hla"]).strip().upper()
        append(row_index, hla_offset, {hla_mapping.get(hla, hla_mapping[UNK_HLA]): 1.0})
    return csr_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(rows), dimensions["total"]),
        dtype=np.float64,
    )


def _fixed_rows(
    positive_rows: Sequence[Mapping[str, str]],
    negative_rows: Sequence[Mapping[str, str]],
    *,
    partition: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, source_rows, identifier in (
        ("positive", positive_rows, "record_id"),
        ("putative_negative", negative_rows, "negative_id"),
    ):
        for row in source_rows:
            label = int(row["label"])
            expected_label = 1 if source == "positive" else 0
            if label != expected_label:
                raise ValueError(f"invalid {source} label in {partition}")
            rows.append(
                {
                    "sample_id": row[identifier],
                    "source": source,
                    "tier": "development",
                    "strategy": "distance_aware",
                    "negative_strategy": "N1",
                    "split": partition,
                    "pmhc_id": row["pmhc_id"],
                    "tcr_cdr3_beta": row["tcr_cdr3_beta"],
                    "epitope": row["epitope"],
                    "hla": row["hla"],
                    "label": label,
                }
            )
    return rows


def _audit_manifest_inputs(
    project_dir: Path,
    manifest: Mapping[str, Any],
    fold_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, dict[str, list[dict[str, Any]]]]]:
    audit: dict[str, Any] = {}
    all_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for fold_id in fold_ids:
        fold_audit: dict[str, Any] = {}
        fold_rows: dict[str, list[dict[str, Any]]] = {}
        for partition in PARTITIONS:
            entry = manifest["folds"][fold_id]["partitions"][partition]
            kind_rows: dict[str, list[dict[str, str]]] = {}
            partition_audit: dict[str, Any] = {}
            for kind, key in (("positive", "positive_output"), ("negative", "negative_output")):
                output = entry[key]
                path = _resolve(project_dir, output["path"])
                observed_sha = file_sha256(path)
                if observed_sha != output["sha256"]:
                    raise ValueError(f"SHA-256 mismatch for {fold_id} {partition} {kind}")
                rows = _read_csv(path)
                if len(rows) != int(output["rows"]):
                    raise ValueError(f"row count mismatch for {fold_id} {partition} {kind}")
                kind_rows[kind] = rows
                partition_audit[kind] = {
                    "path": _relative(project_dir, path),
                    "sha256": observed_sha,
                    "rows": len(rows),
                }
            fixed = _fixed_rows(
                kind_rows["positive"], kind_rows["negative"], partition=partition
            )
            if len({row["sample_id"] for row in fixed}) != len(fixed):
                raise ValueError(f"duplicate sample ID in {fold_id} {partition}")
            fold_rows[partition] = fixed
            fold_audit[partition] = partition_audit
        all_rows[fold_id] = fold_rows
        audit[fold_id] = fold_audit
    return audit, all_rows


def _metrics(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    pooled = binary_classification_metrics(
        [int(row["label"]) for row in rows],
        [float(row["probability"]) for row in rows],
    )
    macro, per_query = query_level_classification_metrics(rows)
    return {
        "pooled_auroc": pooled["auroc"],
        "pooled_auprc": pooled["auprc"],
        "macro_auroc": float(macro["auroc"]),
        "macro_auprc": float(macro["auprc"]),
    }, per_query


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Five-fold k-mer logistic regression baseline",
        "",
        "This report was generated from the frozen cap=100 query-balanced folds. No sampling or hyperparameter selection was performed.",
        "",
        "## Fixed specification",
        "",
        "- Features: normalized TCR and epitope 1-mer/2-mer frequencies, 20x20 amino-acid composition interactions, raw sequence lengths, and train-only HLA one-hot with UNK.",
        "- Model: L2 LogisticRegression, C=1.0, class_weight=None, max_iter=5000, fixed seed.",
        "- Fitting: each fold uses train only; validation and test never fit features or model parameters.",
        "",
        "## Fold test results",
        "",
        "| Fold | Samples | Queries | Pooled AUROC | Pooled AUPRC | Macro AUROC | Macro AUPRC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
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
        lines.append(f"- {metric}: {item['mean']:.6f} +/- {item['standard_deviation']:.6f}")
    for metric, value in summary["oof_metrics"].items():
        lines.append(f"- OOF {metric}: {value:.6f}")
    lines.extend(["", "## Audit", "", "All manifest hashes passed, each query appears as test exactly once, and validation/test fitting flags are false.", ""])
    return "\n".join(lines)


def train_kmer_logistic_five_fold(
    project_dir: Path, config_path: Path
) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    fold_ids = [str(value) for value in config["fold_ids"]]
    if tuple(config["features"]["tcr_kmers"]) != (1, 2) or tuple(
        config["features"]["epitope_kmers"]
    ) != (1, 2):
        raise ValueError("fixed baseline requires 1-mer and 2-mer features")
    expected_model = {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "penalty": "l2",
        "C": 1.0,
        "class_weight": None,
        "max_iter": 5000,
        "solver": "liblinear",
        "random_state_source": "seed",
    }
    if config["model"] != expected_model:
        raise ValueError("logistic regression configuration differs from the fixed baseline")
    output_directory = _resolve(project_dir, config["output"]["directory"])
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_directory}")
    manifest_path = _resolve(project_dir, config["data"]["manifest"])
    manifest = _load_json(manifest_path)
    rotation_audit = validate_query_test_rotation(manifest, fold_ids)
    input_audit, rows_by_fold = _audit_manifest_inputs(project_dir, manifest, fold_ids)
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
            {**row, "fold_id": fold_id, "logit": float(logit), "probability": float(probability)}
            for row, logit, probability in zip(test_rows, logits, probabilities, strict=True)
        ]
        fold_metrics, per_query = _metrics(scored_rows)
        per_query = [{"fold_id": fold_id, **row} for row in per_query]
        _write_csv(fold_directory / "test_predictions.csv", PREDICTION_COLUMNS, scored_rows)
        _write_csv(fold_directory / "per_query_metrics.csv", PER_QUERY_COLUMNS, per_query)
        artifact = {
            "estimator": model,
            "feature_config": config["features"],
            "model_config": config["model"],
            "seed": seed,
            "fold_id": fold_id,
            "hla_mapping": hla_mapping,
            "feature_dimensions": feature_dimensions(hla_mapping),
            "input_audit": input_audit[fold_id],
            "fit_partitions": ["train"],
        }
        joblib.dump(artifact, fold_directory / "model.joblib", compress=3)
        fold_summaries[fold_id] = {
            "train_sample_count": len(train_rows),
            "validation_sample_count": len(rows_by_fold[fold_id]["validation"]),
            "test_sample_count": len(test_rows),
            "test_query_count": len(per_query),
            "known_train_hla_count": len(hla_mapping) - 1,
            "unknown_test_hla_count": sum(
                str(row["hla"]).strip().upper() not in hla_mapping for row in test_rows
            ),
            "feature_count": train_features.shape[1],
            "iterations": int(model.n_iter_[0]),
            "test_metrics": fold_metrics,
            "artifacts": {
                "model": _relative(project_dir, fold_directory / "model.joblib"),
                "test_predictions": _relative(project_dir, fold_directory / "test_predictions.csv"),
                "per_query_metrics": _relative(project_dir, fold_directory / "per_query_metrics.csv"),
            },
        }
        oof_rows.extend(scored_rows)
        oof_query_rows.extend(per_query)

    oof_metrics, recomputed_oof_queries = _metrics(oof_rows)
    if len(recomputed_oof_queries) != len(oof_query_rows):
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
    _write_csv(output_directory / "oof_test_predictions.csv", PREDICTION_COLUMNS, oof_rows)
    _write_csv(output_directory / "oof_per_query_metrics.csv", PER_QUERY_COLUMNS, oof_query_rows)
    summary = {
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
        "audit": {
            "status": "passed",
            "manifest": {
                "path": _relative(project_dir, manifest_path),
                "sha256": file_sha256(manifest_path),
            },
            "config": {
                "path": _relative(project_dir, config_path),
                "sha256": file_sha256(config_path),
            },
            "inputs": input_audit,
            "query_test_rotation": rotation_audit,
            "resampling_performed": False,
            "feature_fit_partitions": ["train"],
            "model_fit_partitions": ["train"],
            "validation_rows_used_for_fitting": False,
            "test_rows_used_for_fitting": False,
            "hyperparameter_tuning_performed": False,
        },
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary
