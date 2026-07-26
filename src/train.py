from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import DatasetSpec, HlaVocab, PartitionPaths
from .metrics import (
    binary_classification_metrics,
    query_level_classification_metrics,
)
from .model_cnn import TwoTowerCnn
from .tcrue.distance_aware_split import cross_partition_maximum_similarity
from .vocab import AminoAcidVocab, PAD_ID, UNK_ID


HISTORY_COLUMNS = (
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_auroc",
    "validation_auprc",
    "validation_query_macro_auroc",
    "validation_query_macro_auprc",
    "early_stopping_metric",
    "early_stopping_value",
    "is_best",
)
DYNAMIC_SAMPLING_COLUMNS = (
    "epoch",
    "sampling_seed",
    "selected_negative_pairs",
    "selected_n1_only",
    "selected_n2_only",
    "selected_both",
    "n1_source_memberships",
    "n2_source_memberships",
    "selection_sha256",
    "overlap_with_previous_ratio",
)
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
PER_QUERY_METRIC_COLUMNS = (
    "partition",
    "query_id",
    "epitope",
    "hla",
    "positive_count",
    "negative_count",
    "sample_count",
    "auroc",
    "auprc",
)


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    auroc: float
    auprc: float
    sample_count: int
    predictions: list[dict[str, Any]]

    def metrics_dict(self) -> dict[str, float | int]:
        return {
            "loss": self.loss,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation: dict[str, Any]
    history: list[dict[str, Any]]
    checkpoint_path: Path
    dynamic_sampling_audit_path: Path | None


def validate_unseen_epitope_isolation(
    partition_rows: Mapping[str, list[dict[str, str]]],
) -> dict[str, Any]:
    required_partitions = ("train", "validation", "test")
    missing = [name for name in required_partitions if name not in partition_rows]
    if missing:
        raise ValueError(f"missing partitions for peptide validation: {missing}")

    peptide_sets = {
        partition: {
            row["epitope"].strip().upper()
            for row in partition_rows[partition]
            if row["epitope"].strip()
        }
        for partition in required_partitions
    }
    pairwise_overlaps: dict[str, int] = {}
    overlap_examples: dict[str, list[str]] = {}
    for index, left in enumerate(required_partitions):
        for right in required_partitions[index + 1 :]:
            key = f"{left}_vs_{right}"
            overlap = peptide_sets[left] & peptide_sets[right]
            pairwise_overlaps[key] = len(overlap)
            if overlap:
                overlap_examples[key] = sorted(overlap)[:5]

    train_negative_peptides = {
        row["epitope"].strip().upper()
        for row in partition_rows["train"]
        if row.get("_source") == "putative_negative" or row.get("label") == "0"
    }
    heldout_peptides = peptide_sets["validation"] | peptide_sets["test"]
    train_negative_overlap = train_negative_peptides & heldout_peptides
    if overlap_examples or train_negative_overlap:
        details = {
            "pairwise_overlap_examples": overlap_examples,
            "train_negative_heldout_examples": sorted(train_negative_overlap)[:5],
        }
        raise ValueError(f"unseen-epitope peptide leakage detected: {details}")

    return {
        "partition_peptide_counts": {
            partition: len(peptide_sets[partition])
            for partition in required_partitions
        },
        "pairwise_overlap_counts": pairwise_overlaps,
        "train_negative_peptide_count": len(train_negative_peptides),
        "train_negative_heldout_overlap_count": 0,
    }


def _interaction_key(row: Mapping[str, str]) -> tuple[str, str, str]:
    return (
        row["tcr_cdr3_beta"].strip().upper(),
        row["epitope"].strip().upper(),
        row["hla"].strip().upper(),
    )


def validate_partition_interaction_isolation(
    partition_rows: Mapping[str, list[dict[str, str]]],
) -> dict[str, Any]:
    partitions = ("train", "validation", "test")
    key_sets: dict[str, set[tuple[str, str, str]]] = {}
    duplicate_counts: dict[str, int] = {}
    for partition in partitions:
        keys = [_interaction_key(row) for row in partition_rows[partition]]
        key_sets[partition] = set(keys)
        duplicate_counts[partition] = len(keys) - len(key_sets[partition])

    overlap_counts: dict[str, int] = {}
    overlap_examples: dict[str, list[tuple[str, str, str]]] = {}
    for index, left in enumerate(partitions):
        for right in partitions[index + 1 :]:
            name = f"{left}_vs_{right}"
            overlap = key_sets[left] & key_sets[right]
            overlap_counts[name] = len(overlap)
            if overlap:
                overlap_examples[name] = sorted(overlap)[:5]

    if any(duplicate_counts.values()) or overlap_examples:
        raise ValueError(
            "partition interaction isolation failed: "
            f"duplicates={duplicate_counts}, overlap_examples={overlap_examples}"
        )
    return {
        "partition_interaction_counts": {
            partition: len(key_sets[partition]) for partition in partitions
        },
        "within_partition_duplicate_counts": duplicate_counts,
        "pairwise_intersection_counts": overlap_counts,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_negative_sampling_manifest(
    *,
    manifest_path: Path,
    negative_directory: Path,
    partition_rows: Mapping[str, list[dict[str, str]]],
    data_spec: DatasetSpec,
    seed: int,
) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("seed") != seed:
        raise ValueError("negative sampling manifest seed mismatch")
    expected_policy = {
        "interaction_key": ["tcr_cdr3_beta", "epitope", "hla"],
        "partition_priority": ["test", "validation", "train"],
    }
    policy_name = f"{data_spec.negative_strategy.lower()}_global_deduplication"
    if manifest.get(policy_name) != expected_policy:
        raise ValueError(
            "negative sampling manifest has the wrong "
            f"{data_spec.negative_strategy} deduplication policy"
        )

    partition_report: dict[str, dict[str, Any]] = {}
    for partition in ("train", "validation", "test"):
        entry = manifest["tiers"][data_spec.tier][
            data_spec.positive_split_strategy
        ][partition]["negatives"][data_spec.negative_strategy]
        expected_name = (
            f"{data_spec.tier}_{data_spec.positive_split_strategy}_{partition}_"
            f"{data_spec.negative_strategy.lower()}.csv"
        )
        if entry["file"] != expected_name:
            raise ValueError(f"negative manifest filename mismatch for {partition}")
        csv_path = negative_directory / entry["file"]
        actual_sha256 = _file_sha256(csv_path)
        if actual_sha256 != entry["sha256"]:
            raise ValueError(f"negative CSV checksum mismatch for {partition}")

        rows = partition_rows[partition]
        positive_count = sum(row.get("label") == "1" for row in rows)
        negative_count = sum(row.get("label") == "0" for row in rows)
        class_balance = entry["class_balance"]
        deduplication = entry["global_deduplication"]
        if entry["generated_negative_pairs"] != negative_count:
            raise ValueError(f"negative manifest count mismatch for {partition}")
        if deduplication["generated_negative_pairs"] != negative_count:
            raise ValueError(f"negative deduplication count mismatch for {partition}")
        expected_fraction = positive_count / (positive_count + negative_count)
        if (
            class_balance["positive_pairs"] != positive_count
            or class_balance["negative_pairs"] != negative_count
            or abs(class_balance["positive_fraction"] - expected_fraction) > 1e-15
        ):
            raise ValueError(f"negative manifest class balance mismatch for {partition}")
        partition_report[partition] = {
            "file": entry["file"],
            "sha256": actual_sha256,
            "negative_count": negative_count,
            "cross_partition_duplicates_removed": deduplication[
                "cross_partition_duplicates_removed"
            ],
            "positive_fraction": expected_fraction,
        }

    return {
        "manifest_file": manifest_path.name,
        "manifest_sha256": _file_sha256(manifest_path),
        "seed": seed,
        policy_name: expected_policy,
        "partitions": partition_report,
    }


def validate_distance_aware_n1_inputs(
    *,
    positive_manifest_path: Path,
    negative_manifest_path: Path,
    positive_directory: Path,
    negative_directory: Path,
    positive_registry_path: Path,
    partition_rows: Mapping[str, list[dict[str, str]]],
    data_spec: DatasetSpec,
    seed: int,
    maximum_similarity: float = 0.5,
) -> dict[str, Any]:
    expected_spec = DatasetSpec("development", "distance_aware", "N1")
    if data_spec != expected_spec:
        raise ValueError("distance-aware N1 validation requires its exact data spec")
    if maximum_similarity != 0.5:
        raise ValueError("distance-aware training requires the fixed 0.5 threshold")

    with positive_manifest_path.open("r", encoding="utf-8") as handle:
        positive_manifest = json.load(handle)
    with negative_manifest_path.open("r", encoding="utf-8") as handle:
        negative_manifest = json.load(handle)

    if (
        positive_manifest.get("seed") != seed
        or positive_manifest.get("tier") != "development"
        or positive_manifest.get("negative_samples_generated") is not False
    ):
        raise ValueError("distance-aware positive manifest experiment mismatch")
    similarity = positive_manifest.get("similarity", {})
    if float(similarity.get("threshold", -1)) != maximum_similarity:
        raise ValueError("distance-aware positive manifest threshold mismatch")
    if (
        negative_manifest.get("seed") != seed
        or negative_manifest.get("tier") != "development"
        or negative_manifest.get("positive_split_strategy") != "distance_aware"
        or negative_manifest.get("negative_strategy") != "N1"
        or negative_manifest.get("negative_to_positive_ratio") != 1
    ):
        raise ValueError("distance-aware N1 manifest experiment mismatch")

    positive_manifest_sha = _file_sha256(positive_manifest_path)
    if (
        negative_manifest["positive_splits_manifest"]["sha256"]
        != positive_manifest_sha
    ):
        raise ValueError("distance-aware positive manifest checksum mismatch")
    registry_sha = _file_sha256(positive_registry_path)
    if negative_manifest["positive_registry"]["sha256"] != registry_sha:
        raise ValueError("distance-aware positive registry checksum mismatch")
    with positive_registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"tcr_cdr3_beta", "epitope", "hla"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("positive registry is missing interaction key columns")
        positive_registry = {_interaction_key(row) for row in reader}
    if len(positive_registry) != int(
        negative_manifest["positive_registry"]["positive_pairs"]
    ):
        raise ValueError("distance-aware positive registry count mismatch")

    positive_rows: dict[str, list[dict[str, str]]] = {}
    partition_report: dict[str, Any] = {}
    negative_key_sets: dict[str, set[tuple[str, str, str]]] = {}
    for partition in ("train", "validation", "test"):
        positive_entry = positive_manifest["partitions"][partition]
        expected_positive_name = f"development_distance_aware_{partition}.csv"
        if positive_entry["file"] != expected_positive_name:
            raise ValueError(f"distance-aware positive filename mismatch for {partition}")
        positive_path = positive_directory / expected_positive_name
        positive_sha = _file_sha256(positive_path)
        if positive_sha != positive_entry["sha256"]:
            raise ValueError(f"distance-aware positive CSV checksum mismatch for {partition}")

        negative_entry = negative_manifest["partitions"][partition]
        output_entry = negative_entry["negative_output"]
        expected_negative_name = f"development_distance_aware_{partition}_n1.csv"
        if output_entry["file"] != expected_negative_name:
            raise ValueError(f"distance-aware N1 filename mismatch for {partition}")
        negative_path = negative_directory / expected_negative_name
        negative_sha = _file_sha256(negative_path)
        if negative_sha != output_entry["sha256"]:
            raise ValueError(f"distance-aware N1 CSV checksum mismatch for {partition}")
        if negative_entry["positive_input"]["sha256"] != positive_sha:
            raise ValueError(f"distance-aware N1 positive reference mismatch for {partition}")

        rows = partition_rows[partition]
        positives = [row for row in rows if row.get("label") == "1"]
        negatives = [row for row in rows if row.get("label") == "0"]
        if len(positives) != int(positive_entry["positive_pairs"]):
            raise ValueError(f"distance-aware positive count mismatch for {partition}")
        if len(negatives) != int(negative_entry["negative_pairs"]):
            raise ValueError(f"distance-aware N1 count mismatch for {partition}")
        if len(positives) != len(negatives):
            raise ValueError(f"distance-aware class balance is not 1:1 for {partition}")

        positive_peptides = {row["epitope"] for row in positives}
        positive_tcrs = {row["tcr_cdr3_beta"] for row in positives}
        positive_ids = {row.get("record_id", "") for row in positives}
        negative_keys = [_interaction_key(row) for row in negatives]
        negative_key_set = set(negative_keys)
        failures = {
            "duplicate_negative_interactions": len(negative_keys)
            - len(negative_key_set),
            "positive_registry_intersections": len(
                negative_key_set & positive_registry
            ),
            "peptides_outside_partition": sum(
                row["epitope"] not in positive_peptides for row in negatives
            ),
            "tcrs_outside_partition": sum(
                row["tcr_cdr3_beta"] not in positive_tcrs for row in negatives
            ),
            "source_records_outside_partition": sum(
                row.get("source_positive_record_id", "") not in positive_ids
                for row in negatives
            ),
            "non_n1_strategy": sum(
                row.get("negative_strategy") != "N1" for row in negatives
            ),
            "unchanged_source_pmhc": sum(
                row.get("pmhc_id") == row.get("source_positive_pmhc_id")
                for row in negatives
            ),
        }
        if any(failures.values()):
            raise ValueError(
                f"distance-aware N1 integrity failure in {partition}: {failures}"
            )
        positive_rows[partition] = positives
        negative_key_sets[partition] = negative_key_set
        partition_report[partition] = {
            "positive_file": expected_positive_name,
            "positive_sha256": positive_sha,
            "negative_file": expected_negative_name,
            "negative_sha256": negative_sha,
            "positive_count": len(positives),
            "negative_count": len(negatives),
            **failures,
        }

    negative_intersections: dict[str, int] = {}
    partitions = ("train", "validation", "test")
    for index, left in enumerate(partitions):
        for right in partitions[index + 1 :]:
            name = f"{left}_vs_{right}"
            overlap = negative_key_sets[left] & negative_key_sets[right]
            negative_intersections[name] = len(overlap)
            if overlap:
                raise ValueError(f"distance-aware N1 partition overlap: {name}")

    cross_similarity = cross_partition_maximum_similarity(positive_rows)
    observed_similarity = float(cross_similarity["maximum_similarity"])
    if observed_similarity > maximum_similarity + 1e-15:
        raise ValueError(
            "distance-aware cross-partition peptide similarity exceeds 0.5"
        )
    manifest_similarity = float(
        positive_manifest["cross_partition_similarity"]["maximum_similarity"]
    )
    if abs(manifest_similarity - observed_similarity) > 1e-15:
        raise ValueError("distance-aware recomputed similarity disagrees with manifest")

    return {
        "positive_manifest_sha256": positive_manifest_sha,
        "negative_manifest_sha256": _file_sha256(negative_manifest_path),
        "positive_registry_sha256": registry_sha,
        "positive_registry_size": len(positive_registry),
        "maximum_allowed_similarity": maximum_similarity,
        "cross_partition_similarity": cross_similarity,
        "negative_pairwise_intersection_counts": negative_intersections,
        "partitions": partition_report,
    }


def validate_distance_aware_five_fold_n1_inputs(
    *,
    fold_id: str,
    positive_manifest_path: Path,
    negative_manifest_path: Path,
    positive_directory: Path,
    negative_directory: Path,
    positive_registry_path: Path,
    partition_paths: Mapping[str, PartitionPaths],
    partition_rows: Mapping[str, list[dict[str, str]]],
    data_spec: DatasetSpec,
    seed: int,
    maximum_similarity: float = 0.5,
) -> dict[str, Any]:
    expected_spec = DatasetSpec("development", "distance_aware", "N1")
    if data_spec != expected_spec:
        raise ValueError("five-fold distance-aware validation requires N1 data")
    if maximum_similarity != 0.5:
        raise ValueError("five-fold distance-aware training requires threshold 0.5")

    with positive_manifest_path.open("r", encoding="utf-8") as handle:
        positive_manifest = json.load(handle)
    with negative_manifest_path.open("r", encoding="utf-8") as handle:
        negative_manifest = json.load(handle)
    if fold_id not in positive_manifest.get("folds", {}):
        raise ValueError(f"unknown rebalanced five-fold ID: {fold_id}")
    if fold_id not in negative_manifest.get("folds", {}):
        raise ValueError(f"N1 manifest is missing fold: {fold_id}")

    if (
        positive_manifest.get("schema_version") != 1
        or positive_manifest.get("seed") != seed
        or positive_manifest.get("tier") != "development"
        or positive_manifest.get("fold_count") != 5
        or positive_manifest.get("status") != "active"
        or positive_manifest.get("negative_samples_generated") is not False
        or positive_manifest.get("bucket_allocation", {}).get("objective")
        != "joint_weighted"
    ):
        raise ValueError("positive manifest is not the active rebalanced five-fold split")
    similarity = positive_manifest.get("similarity", {})
    if float(similarity.get("threshold", -1)) != maximum_similarity:
        raise ValueError("rebalanced five-fold positive threshold mismatch")
    if (
        negative_manifest.get("schema_version") != 1
        or negative_manifest.get("base_seed") != seed
        or negative_manifest.get("tier") != "development"
        or negative_manifest.get("fold_count") != 5
        or negative_manifest.get("positive_split_strategy")
        != "distance_aware_5fold_rebalanced"
        or negative_manifest.get("negative_strategy") != "N1"
        or negative_manifest.get("negative_to_positive_ratio") != 1
        or negative_manifest.get("positive_splits_modified") is not False
    ):
        raise ValueError("rebalanced five-fold N1 manifest experiment mismatch")

    positive_manifest_sha = _file_sha256(positive_manifest_path)
    if (
        negative_manifest["positive_splits_manifest"]["sha256"]
        != positive_manifest_sha
    ):
        raise ValueError("rebalanced five-fold positive manifest checksum mismatch")
    registry_sha = _file_sha256(positive_registry_path)
    registry_entry = negative_manifest["positive_registry"]
    if registry_entry["sha256"] != registry_sha:
        raise ValueError("rebalanced five-fold positive registry checksum mismatch")
    with positive_registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"tcr_cdr3_beta", "epitope", "hla"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("positive registry is missing interaction key columns")
        positive_registry = {_interaction_key(row) for row in reader}
    if len(positive_registry) != int(registry_entry["positive_pairs"]):
        raise ValueError("rebalanced five-fold positive registry count mismatch")

    positive_fold = positive_manifest["folds"][fold_id]
    negative_fold = negative_manifest["folds"][fold_id]
    positive_rows: dict[str, list[dict[str, str]]] = {}
    negative_key_sets: dict[str, set[tuple[str, str, str]]] = {}
    partition_report: dict[str, Any] = {}
    for partition in ("train", "validation", "test"):
        if partition not in partition_paths:
            raise ValueError(f"missing resolved path for {fold_id}/{partition}")
        positive_entry = positive_fold["partitions"][partition]
        negative_entry = negative_fold["partitions"][partition]
        positive_name = (
            f"development_distance_aware_rebalanced_{fold_id}_{partition}.csv"
        )
        negative_name = (
            f"development_distance_aware_rebalanced_{fold_id}_{partition}_n1.csv"
        )
        if positive_entry["file"] != positive_name:
            raise ValueError(f"positive filename mismatch for {fold_id}/{partition}")
        if negative_entry["negative_output"]["file"] != negative_name:
            raise ValueError(f"N1 filename mismatch for {fold_id}/{partition}")

        expected_positive_path = positive_directory / positive_name
        expected_negative_path = negative_directory / negative_name
        resolved = partition_paths[partition]
        if resolved.positive_csv.resolve() != expected_positive_path.resolve():
            raise ValueError(f"configured positive path mismatch for {fold_id}/{partition}")
        if resolved.negative_csv.resolve() != expected_negative_path.resolve():
            raise ValueError(f"configured N1 path mismatch for {fold_id}/{partition}")
        positive_sha = _file_sha256(expected_positive_path)
        negative_sha = _file_sha256(expected_negative_path)
        if positive_sha != positive_entry["sha256"]:
            raise ValueError(f"positive CSV checksum mismatch for {fold_id}/{partition}")
        if negative_sha != negative_entry["negative_output"]["sha256"]:
            raise ValueError(f"N1 CSV checksum mismatch for {fold_id}/{partition}")
        if negative_entry["positive_input"]["sha256"] != positive_sha:
            raise ValueError(f"N1 positive reference mismatch for {fold_id}/{partition}")

        rows = partition_rows[partition]
        positives = [row for row in rows if row.get("label") == "1"]
        negatives = [row for row in rows if row.get("label") == "0"]
        if len(positives) != int(positive_entry["positive_pairs"]):
            raise ValueError(f"positive count mismatch for {fold_id}/{partition}")
        if len(negatives) != int(negative_entry["negative_pairs"]):
            raise ValueError(f"N1 count mismatch for {fold_id}/{partition}")
        if len(positives) != len(negatives):
            raise ValueError(f"class balance is not 1:1 for {fold_id}/{partition}")

        positive_peptides = {row["epitope"] for row in positives}
        positive_tcrs = {row["tcr_cdr3_beta"] for row in positives}
        positive_ids = {row.get("record_id", "") for row in positives}
        negative_keys = [_interaction_key(row) for row in negatives]
        negative_key_set = set(negative_keys)
        failures = {
            "duplicate_negative_interactions": len(negative_keys)
            - len(negative_key_set),
            "positive_registry_intersections": len(
                negative_key_set & positive_registry
            ),
            "peptides_outside_partition": sum(
                row["epitope"] not in positive_peptides for row in negatives
            ),
            "tcrs_outside_partition": sum(
                row["tcr_cdr3_beta"] not in positive_tcrs for row in negatives
            ),
            "source_records_outside_partition": sum(
                row.get("source_positive_record_id", "") not in positive_ids
                for row in negatives
            ),
            "non_n1_strategy": sum(
                row.get("negative_strategy") != "N1" for row in negatives
            ),
            "unchanged_source_pmhc": sum(
                row.get("pmhc_id") == row.get("source_positive_pmhc_id")
                for row in negatives
            ),
        }
        if any(failures.values()):
            raise ValueError(
                f"five-fold N1 integrity failure in {fold_id}/{partition}: "
                f"{failures}"
            )
        positive_rows[partition] = positives
        negative_key_sets[partition] = negative_key_set
        partition_report[partition] = {
            "positive_file": positive_name,
            "positive_sha256": positive_sha,
            "negative_file": negative_name,
            "negative_sha256": negative_sha,
            "positive_count": len(positives),
            "negative_count": len(negatives),
            **failures,
        }

    negative_intersections: dict[str, int] = {}
    partitions = ("train", "validation", "test")
    for index, left in enumerate(partitions):
        for right in partitions[index + 1 :]:
            name = f"{left}_vs_{right}"
            overlap = negative_key_sets[left] & negative_key_sets[right]
            negative_intersections[name] = len(overlap)
            if overlap:
                raise ValueError(f"five-fold N1 partition overlap: {fold_id}/{name}")

    cross_similarity = cross_partition_maximum_similarity(positive_rows)
    observed_similarity = float(cross_similarity["maximum_similarity"])
    if observed_similarity > maximum_similarity + 1e-15:
        raise ValueError("five-fold cross-partition peptide similarity exceeds 0.5")
    manifest_similarity = float(
        positive_fold["cross_partition_similarity"]["maximum_similarity"]
    )
    if abs(manifest_similarity - observed_similarity) > 1e-15:
        raise ValueError("five-fold recomputed similarity disagrees with manifest")

    return {
        "fold_id": fold_id,
        "positive_manifest_sha256": positive_manifest_sha,
        "negative_manifest_sha256": _file_sha256(negative_manifest_path),
        "positive_registry_sha256": registry_sha,
        "positive_registry_size": len(positive_registry),
        "maximum_allowed_similarity": maximum_similarity,
        "cross_partition_similarity": cross_similarity,
        "negative_pairwise_intersection_counts": negative_intersections,
        "partitions": partition_report,
    }


def validate_query_balanced_five_fold_n1_inputs(
    *,
    project_dir: Path,
    fold_id: str,
    manifest_path: Path,
    positive_registry_path: Path,
    partition_paths: Mapping[str, PartitionPaths],
    partition_rows: Mapping[str, list[dict[str, str]]],
    data_spec: DatasetSpec,
    seed: int,
    per_query_cap: int = 100,
    maximum_similarity: float = 0.5,
) -> dict[str, Any]:
    expected_spec = DatasetSpec("development", "distance_aware", "N1")
    if data_spec != expected_spec:
        raise ValueError("query-balanced five-fold validation requires N1 data")
    if per_query_cap != 100 or maximum_similarity != 0.5:
        raise ValueError("query-balanced validation requires cap=100 and threshold=0.5")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_invariants = {
        "all_original_queries_retained": True,
        "all_queries_positive_negative_one_to_one": True,
        "maximum_per_query_per_class": per_query_cap,
        "known_positive_registry_intersections": 0,
        "duplicate_interactions": 0,
        "within_fold_cross_partition_interactions": 0,
        "within_fold_cross_partition_peptides": 0,
        "models_trained": False,
    }
    invariants = manifest.get("invariants", {})
    if (
        manifest.get("schema_version") != 1
        or manifest.get("tier") != "development"
        or manifest.get("fold_count") != 5
        or manifest.get("base_seed") != seed
        or manifest.get("per_query_cap") != per_query_cap
        or manifest.get("query_key") != ["epitope", "hla"]
        or manifest.get("interaction_key")
        != ["tcr_cdr3_beta", "epitope", "hla"]
        or any(invariants.get(key) != value for key, value in expected_invariants.items())
    ):
        raise ValueError("query-balanced manifest experiment or invariants mismatch")
    if fold_id not in manifest.get("folds", {}):
        raise ValueError(f"query-balanced manifest is missing fold: {fold_id}")

    registry_sha = _file_sha256(positive_registry_path)
    if registry_sha != manifest["inputs"]["positive_registry"]["sha256"]:
        raise ValueError("query-balanced positive registry checksum mismatch")
    with positive_registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"tcr_cdr3_beta", "epitope", "hla"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("positive registry is missing interaction key columns")
        positive_registry = {_interaction_key(row) for row in reader}

    source_manifest_entry = manifest["inputs"]["positive_manifest"]
    source_manifest_path = project_dir / source_manifest_entry["path"]
    if _file_sha256(source_manifest_path) != source_manifest_entry["sha256"]:
        raise ValueError("query-balanced source positive manifest checksum mismatch")
    with source_manifest_path.open("r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    if (
        source_manifest.get("status") != "active"
        or source_manifest.get("negative_samples_generated") is not False
        or fold_id not in source_manifest.get("folds", {})
    ):
        raise ValueError("query-balanced source positive manifest is invalid")

    fold = manifest["folds"][fold_id]
    interaction_sets: dict[str, set[tuple[str, str, str]]] = {}
    peptide_sets: dict[str, set[str]] = {}
    positive_rows: dict[str, list[dict[str, str]]] = {}
    partition_report: dict[str, Any] = {}
    for partition in ("train", "validation", "test"):
        entry = fold["partitions"][partition]
        configured_paths = partition_paths[partition]
        expected_positive_path = project_dir / entry["positive_output"]["path"]
        expected_negative_path = project_dir / entry["negative_output"]["path"]
        if configured_paths.positive_csv.resolve() != expected_positive_path.resolve():
            raise ValueError(f"query-balanced positive path mismatch: {partition}")
        if configured_paths.negative_csv.resolve() != expected_negative_path.resolve():
            raise ValueError(f"query-balanced negative path mismatch: {partition}")
        positive_sha = _file_sha256(expected_positive_path)
        negative_sha = _file_sha256(expected_negative_path)
        if positive_sha != entry["positive_output"]["sha256"]:
            raise ValueError(f"query-balanced positive checksum mismatch: {partition}")
        if negative_sha != entry["negative_output"]["sha256"]:
            raise ValueError(f"query-balanced negative checksum mismatch: {partition}")

        rows = partition_rows[partition]
        positives = [row for row in rows if row.get("label") == "1"]
        negatives = [row for row in rows if row.get("label") == "0"]
        if len(positives) != int(entry["positive_output"]["rows"]):
            raise ValueError(f"query-balanced positive count mismatch: {partition}")
        if len(negatives) != int(entry["negative_output"]["rows"]):
            raise ValueError(f"query-balanced negative count mismatch: {partition}")

        positive_counts = Counter(
            (row["epitope"], row["hla"]) for row in positives
        )
        negative_counts = Counter(
            (row["epitope"], row["hla"]) for row in negatives
        )
        if (
            positive_counts != negative_counts
            or not positive_counts
            or max(positive_counts.values()) > per_query_cap
            or len(positive_counts) != int(entry["audit"]["query_count"])
        ):
            raise ValueError(f"query-balanced class balance failed: {partition}")
        query_statistics = {
            (values["epitope"], values["hla"]): values
            for values in entry["query_statistics"]
        }
        if set(query_statistics) != set(positive_counts):
            raise ValueError(f"query-balanced query manifest mismatch: {partition}")
        for query, count in positive_counts.items():
            values = query_statistics[query]
            if (
                int(values["retained_positive_count"]) != count
                or int(values["retained_negative_count"]) != count
            ):
                raise ValueError(f"query-balanced retained count mismatch: {query}")

        positive_keys = [_interaction_key(row) for row in positives]
        negative_keys = [_interaction_key(row) for row in negatives]
        combined_keys = positive_keys + negative_keys
        if len(combined_keys) != len(set(combined_keys)):
            raise ValueError(f"query-balanced duplicate interaction: {partition}")
        registry_intersections = set(negative_keys) & positive_registry
        if registry_intersections:
            raise ValueError(
                f"query-balanced negative intersects positive registry: {partition}"
            )
        for query in negative_counts:
            query_tcrs = {
                row["tcr_cdr3_beta"]
                for row in negatives
                if (row["epitope"], row["hla"]) == query
            }
            if len(query_tcrs) != negative_counts[query]:
                raise ValueError(f"duplicate negative TCR within query: {query}")

        interaction_sets[partition] = set(combined_keys)
        peptide_sets[partition] = {row["epitope"] for row in rows}
        positive_rows[partition] = positives
        partition_report[partition] = {
            "positive_file": expected_positive_path.name,
            "positive_sha256": positive_sha,
            "negative_file": expected_negative_path.name,
            "negative_sha256": negative_sha,
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "query_count": len(positive_counts),
            "maximum_per_query_per_class": max(positive_counts.values()),
            "positive_registry_intersections": 0,
        }

    interaction_intersections = {}
    peptide_intersections = {}
    partitions = ("train", "validation", "test")
    for index, left in enumerate(partitions):
        for right in partitions[index + 1 :]:
            name = f"{left}_vs_{right}"
            interaction_intersections[name] = len(
                interaction_sets[left] & interaction_sets[right]
            )
            peptide_intersections[name] = len(peptide_sets[left] & peptide_sets[right])
    if any(interaction_intersections.values()) or any(peptide_intersections.values()):
        raise ValueError("query-balanced cross-partition leakage detected")
    if (
        fold["invariants"]["partition_interaction_intersections"]
        != interaction_intersections
        or fold["invariants"]["partition_peptide_intersections"]
        != peptide_intersections
    ):
        raise ValueError("query-balanced leakage audit disagrees with manifest")

    cross_similarity = cross_partition_maximum_similarity(positive_rows)
    observed_similarity = float(cross_similarity["maximum_similarity"])
    if observed_similarity > maximum_similarity + 1e-15:
        raise ValueError("query-balanced peptide similarity exceeds 0.5")
    expected_similarity = float(
        source_manifest["folds"][fold_id]["cross_partition_similarity"][
            "maximum_similarity"
        ]
    )
    if abs(observed_similarity - expected_similarity) > 1e-15:
        raise ValueError("query-balanced similarity disagrees with source manifest")

    return {
        "fold_id": fold_id,
        "manifest_sha256": _file_sha256(manifest_path),
        "positive_registry_sha256": registry_sha,
        "positive_registry_size": len(positive_registry),
        "per_query_cap": per_query_cap,
        "cross_partition_similarity": cross_similarity,
        "partition_interaction_intersections": interaction_intersections,
        "partition_peptide_intersections": peptide_intersections,
        "partitions": partition_report,
    }


def validate_degree_matched_five_fold_inputs(
    *,
    project_dir: Path,
    fold_id: str,
    manifest_path: Path,
    expected_manifest_sha256: str,
    positive_registry_path: Path,
    partition_paths: Mapping[str, PartitionPaths],
    partition_rows: Mapping[str, list[dict[str, str]]],
    data_spec: DatasetSpec,
) -> dict[str, Any]:
    expected_spec = DatasetSpec(
        "development", "distance_aware", "DEGREE_MATCHED"
    )
    if data_spec != expected_spec:
        raise ValueError("degree-matched five-fold validation requires exact data")
    manifest_sha = _file_sha256(manifest_path)
    if manifest_sha != expected_manifest_sha256:
        raise ValueError("degree-matched manifest checksum mismatch")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    invariants = manifest.get("invariants", {})
    if (
        manifest.get("schema_version") != 1
        or manifest.get("tier") != "development"
        or manifest.get("fold_count") != 5
        or manifest.get("query_key") != ["epitope", "hla"]
        or manifest.get("interaction_key")
        != ["tcr_cdr3_beta", "epitope", "hla"]
        or invariants.get("all_fold_partitions_have_exact_solution") is not True
        or invariants.get("all_tcr_degrees_exact") is not True
        or invariants.get("all_query_degrees_exact") is not True
        or invariants.get("known_positive_registry_intersections") != 0
    ):
        raise ValueError("degree-matched manifest experiment or invariants mismatch")
    if fold_id not in manifest.get("folds", {}):
        raise ValueError(f"degree-matched manifest is missing fold: {fold_id}")

    registry_source = manifest["inputs"]["positive_registry"]
    expected_registry_path = project_dir / registry_source["path"]
    if positive_registry_path.resolve() != expected_registry_path.resolve():
        raise ValueError("degree-matched positive registry path mismatch")
    registry_sha = _file_sha256(positive_registry_path)
    if registry_sha != registry_source["sha256"]:
        raise ValueError("degree-matched positive registry checksum mismatch")
    with positive_registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"tcr_cdr3_beta", "epitope", "hla"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("positive registry is missing interaction key columns")
        positive_registry = {_interaction_key(row) for row in reader}

    fold = manifest["folds"][fold_id]
    partition_report: dict[str, Any] = {}
    interaction_sets: dict[str, set[tuple[str, str, str]]] = {}
    peptide_sets: dict[str, set[str]] = {}
    for partition in ("train", "validation", "test"):
        entry = fold["partitions"][partition]
        configured = partition_paths[partition]
        positive_source = entry["positive_input"]
        negative_source = entry["negative_output"]
        positive_path = project_dir / positive_source["path"]
        negative_path = project_dir / negative_source["path"]
        if configured.positive_csv.resolve() != positive_path.resolve():
            raise ValueError(f"degree-matched positive path mismatch: {partition}")
        if configured.negative_csv.resolve() != negative_path.resolve():
            raise ValueError(f"degree-matched negative path mismatch: {partition}")
        positive_sha = _file_sha256(positive_path)
        negative_sha = _file_sha256(negative_path)
        if positive_sha != positive_source["sha256"]:
            raise ValueError(f"degree-matched positive checksum mismatch: {partition}")
        if negative_sha != negative_source["sha256"]:
            raise ValueError(f"degree-matched negative checksum mismatch: {partition}")

        rows = partition_rows[partition]
        positives = [row for row in rows if row.get("label") == "1"]
        negatives = [row for row in rows if row.get("label") == "0"]
        if len(positives) != int(positive_source["rows"]):
            raise ValueError(f"degree-matched positive count mismatch: {partition}")
        if len(negatives) != int(negative_source["rows"]):
            raise ValueError(f"degree-matched negative count mismatch: {partition}")
        if any(
            row.get("negative_strategy", "").strip().lower()
            != "degree_matched"
            for row in negatives
        ):
            raise ValueError(f"degree-matched strategy mismatch: {partition}")

        positive_tcr_degree = Counter(row["tcr_cdr3_beta"] for row in positives)
        negative_tcr_degree = Counter(row["tcr_cdr3_beta"] for row in negatives)
        positive_query_degree = Counter(
            (row["epitope"], row["hla"]) for row in positives
        )
        negative_query_degree = Counter(
            (row["epitope"], row["hla"]) for row in negatives
        )
        if positive_tcr_degree != negative_tcr_degree:
            raise ValueError(f"degree-matched TCR degree mismatch: {partition}")
        if positive_query_degree != negative_query_degree:
            raise ValueError(f"degree-matched query degree mismatch: {partition}")

        positive_keys = [_interaction_key(row) for row in positives]
        negative_keys = [_interaction_key(row) for row in negatives]
        combined_keys = positive_keys + negative_keys
        if len(combined_keys) != len(set(combined_keys)):
            raise ValueError(f"degree-matched duplicate interaction: {partition}")
        registry_intersections = set(negative_keys) & positive_registry
        if registry_intersections:
            raise ValueError(
                f"degree-matched negative intersects positive registry: {partition}"
            )
        audit = entry["audit"]
        if (
            entry.get("exact_solution") is not True
            or audit.get("tcr_degree_exact") is not True
            or audit.get("query_degree_exact") is not True
            or int(audit.get("positive_count", -1)) != len(positives)
            or int(audit.get("negative_count", -1)) != len(negatives)
        ):
            raise ValueError(f"degree-matched manifest audit mismatch: {partition}")

        interaction_sets[partition] = set(combined_keys)
        peptide_sets[partition] = {row["epitope"] for row in rows}
        partition_report[partition] = {
            "positive_file": positive_path.name,
            "positive_sha256": positive_sha,
            "negative_file": negative_path.name,
            "negative_sha256": negative_sha,
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "unique_tcr_count": len(positive_tcr_degree),
            "query_count": len(positive_query_degree),
            "tcr_degree_exact": True,
            "query_degree_exact": True,
            "positive_registry_intersections": 0,
        }

    interaction_intersections: dict[str, int] = {}
    peptide_intersections: dict[str, int] = {}
    partitions = ("train", "validation", "test")
    for index, left in enumerate(partitions):
        for right in partitions[index + 1 :]:
            name = f"{left}_vs_{right}"
            interaction_intersections[name] = len(
                interaction_sets[left] & interaction_sets[right]
            )
            peptide_intersections[name] = len(peptide_sets[left] & peptide_sets[right])
    if any(interaction_intersections.values()) or any(peptide_intersections.values()):
        raise ValueError("degree-matched cross-partition leakage detected")

    return {
        "fold_id": fold_id,
        "manifest_sha256": manifest_sha,
        "positive_registry_sha256": registry_sha,
        "positive_registry_size": len(positive_registry),
        "partition_interaction_intersections": interaction_intersections,
        "partition_peptide_intersections": peptide_intersections,
        "partitions": partition_report,
    }


def validate_n2_integrity(
    partition_rows: Mapping[str, list[dict[str, str]]],
    positive_registry_path: Path,
) -> dict[str, Any]:
    with positive_registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"tcr_cdr3_beta", "epitope", "hla"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("positive registry is missing interaction key columns")
        positive_registry = {_interaction_key(row) for row in reader}

    partition_balance: dict[str, dict[str, float | int]] = {}
    hla_mismatches: list[str] = []
    same_peptides: list[str] = []
    strategy_mismatches: list[str] = []
    known_positive_intersections: list[str] = []
    negative_count = 0
    for partition in ("train", "validation", "test"):
        rows = partition_rows[partition]
        positives = [row for row in rows if row.get("label") == "1"]
        negatives = [row for row in rows if row.get("label") == "0"]
        total = len(positives) + len(negatives)
        if total != len(rows) or total == 0:
            raise ValueError(f"invalid or empty labels in {partition} partition")
        partition_balance[partition] = {
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "positive_fraction": len(positives) / total,
        }
        negative_count += len(negatives)

        for row in negatives:
            sample_id = row.get("negative_id", "")
            if row.get("negative_strategy") != "N2":
                strategy_mismatches.append(sample_id)
            if row["hla"].strip().upper() != row.get(
                "source_positive_hla", ""
            ).strip().upper():
                hla_mismatches.append(sample_id)
            if row["epitope"].strip().upper() == row.get(
                "source_positive_epitope", ""
            ).strip().upper():
                same_peptides.append(sample_id)
            if _interaction_key(row) in positive_registry:
                known_positive_intersections.append(sample_id)

    failure_counts = {
        "non_n2_strategy_count": len(strategy_mismatches),
        "hla_mismatch_count": len(hla_mismatches),
        "same_peptide_count": len(same_peptides),
        "positive_registry_intersection_count": len(known_positive_intersections),
    }
    if any(failure_counts.values()):
        examples = {
            "non_n2_strategy": strategy_mismatches[:5],
            "hla_mismatch": hla_mismatches[:5],
            "same_peptide": same_peptides[:5],
            "positive_registry_intersection": known_positive_intersections[:5],
        }
        raise ValueError(
            f"N2 integrity validation failed: counts={failure_counts}, examples={examples}"
        )

    return {
        "positive_registry_size": len(positive_registry),
        "validated_negative_count": negative_count,
        "partition_class_balance": partition_balance,
        **failure_counts,
    }


def validate_mixed_negative_manifest(
    *,
    manifest_path: Path,
    positive_directory: Path,
    negative_directory: Path,
    positive_registry_path: Path,
    partition_rows: Mapping[str, list[dict[str, str]]],
    data_spec: DatasetSpec,
    seed: int,
) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_experiment = {
        "tier": "development",
        "positive_split_strategy": "unseen_epitope",
    }
    if data_spec != DatasetSpec("development", "unseen_epitope", "MIXED"):
        raise ValueError("mixed manifest validation requires the mixed data spec")
    if manifest.get("seed") != seed or manifest.get("experiment") != expected_experiment:
        raise ValueError("mixed manifest experiment or seed mismatch")
    if manifest.get("interaction_key") != ["tcr_cdr3_beta", "epitope", "hla"]:
        raise ValueError("mixed manifest interaction key mismatch")
    if manifest.get("partition_priority") != ["test", "validation", "train"]:
        raise ValueError("mixed manifest partition priority mismatch")
    if manifest.get("candidate_negatives_regenerated") is not False:
        raise ValueError("mixed manifest does not certify existing negative inputs")

    registry_entry = manifest["input_manifests"]["positive_registry"]
    registry_sha = _file_sha256(positive_registry_path)
    if registry_sha != registry_entry["sha256"]:
        raise ValueError("mixed manifest positive registry checksum mismatch")
    with positive_registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        positive_registry = {_interaction_key(row) for row in reader}
    if len(positive_registry) != int(registry_entry["positive_pairs"]):
        raise ValueError("mixed manifest positive registry count mismatch")

    partition_report: dict[str, Any] = {}
    all_negative_keys: dict[str, set[tuple[str, str, str]]] = {}
    for partition in ("train", "validation", "test"):
        entry = manifest["partitions"][partition]
        output = entry["output"]
        expected_negative_name = (
            f"development_unseen_epitope_{partition}_mixed.csv"
        )
        if output["file"] != expected_negative_name:
            raise ValueError(f"mixed filename mismatch for {partition}")
        negative_path = negative_directory / output["file"]
        negative_sha = _file_sha256(negative_path)
        if negative_sha != output["sha256"]:
            raise ValueError(f"mixed CSV checksum mismatch for {partition}")

        positive_entry = entry["positive_input"]
        positive_path = positive_directory / Path(positive_entry["path"]).name
        positive_sha = _file_sha256(positive_path)
        if positive_sha != positive_entry["sha256"]:
            raise ValueError(f"positive CSV checksum mismatch for {partition}")

        rows = partition_rows[partition]
        positives = [row for row in rows if row.get("label") == "1"]
        negatives = [row for row in rows if row.get("label") == "0"]
        if len(positives) != int(positive_entry["positive_pairs"]):
            raise ValueError(f"mixed positive count mismatch for {partition}")
        if len(negatives) != int(output["rows"]):
            raise ValueError(f"mixed negative count mismatch for {partition}")
        selection = entry["selection"]
        if len(negatives) != int(selection["selected_negative_pairs"]):
            raise ValueError(f"mixed selection count mismatch for {partition}")
        if len(positives) != len(negatives):
            raise ValueError(f"mixed class balance is not 1:1 for {partition}")

        keys = [_interaction_key(row) for row in negatives]
        key_set = set(keys)
        if len(keys) != len(key_set):
            raise ValueError(f"duplicate mixed negatives in {partition}")
        intersections = key_set & positive_registry
        if intersections:
            raise ValueError(f"mixed negatives intersect positive registry in {partition}")
        source_counts = Counter(row.get("source", "") for row in negatives)
        if set(source_counts) - {"N1", "N2", "both"}:
            raise ValueError(f"invalid mixed source in {partition}")
        expected_counts = {
            "N1": int(selection["selected_n1_only"]),
            "N2": int(selection["selected_n2_only"]),
            "both": int(selection["selected_both"]),
        }
        if any(source_counts[source] != count for source, count in expected_counts.items()):
            raise ValueError(f"mixed source count mismatch for {partition}")
        if source_counts["N1"] != source_counts["N2"]:
            raise ValueError(f"mixed N1:N2 ratio mismatch for {partition}")
        for row in negatives:
            n1_id = row.get("source_n1_negative_id", "")
            n2_id = row.get("source_n2_negative_id", "")
            source = row["source"]
            if (
                (source == "N1" and (not n1_id or n2_id))
                or (source == "N2" and (not n2_id or n1_id))
                or (source == "both" and (not n1_id or not n2_id))
            ):
                raise ValueError(f"mixed source provenance mismatch in {partition}")
        all_negative_keys[partition] = key_set
        partition_report[partition] = {
            "positive_file": positive_path.name,
            "positive_sha256": positive_sha,
            "negative_file": negative_path.name,
            "negative_sha256": negative_sha,
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "source_counts": dict(sorted(source_counts.items())),
            "positive_registry_intersection_count": 0,
        }

    pairwise_intersections: dict[str, int] = {}
    partitions = ("train", "validation", "test")
    for index, left in enumerate(partitions):
        for right in partitions[index + 1 :]:
            name = f"{left}_vs_{right}"
            overlap = all_negative_keys[left] & all_negative_keys[right]
            pairwise_intersections[name] = len(overlap)
            if overlap:
                raise ValueError(f"mixed negative partition overlap: {name}")
    return {
        "manifest_file": manifest_path.name,
        "manifest_sha256": _file_sha256(manifest_path),
        "seed": seed,
        "candidate_negatives_regenerated": False,
        "positive_registry_sha256": registry_sha,
        "pairwise_negative_intersection_counts": pairwise_intersections,
        "partitions": partition_report,
    }


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0
    for batch in dataloader:
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["tcr_ids"].to(device, non_blocking=True),
            batch["epitope_ids"].to(device, non_blocking=True),
            batch["hla_id"].to(device, non_blocking=True),
        )
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("training dataloader is empty")
    return total_loss / sample_count


def _metadata_at(metadata: Mapping[str, Any], key: str, index: int) -> str:
    values = metadata.get(key)
    if values is None:
        return ""
    return str(values[index])


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_function: nn.Module | None = None,
) -> EvaluationResult:
    model.eval()
    criterion = loss_function or nn.BCEWithLogitsLoss()
    labels_all: list[int] = []
    probabilities_all: list[float] = []
    predictions: list[dict[str, Any]] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(
                batch["tcr_ids"].to(device, non_blocking=True),
                batch["epitope_ids"].to(device, non_blocking=True),
                batch["hla_id"].to(device, non_blocking=True),
            )
            probabilities = torch.sigmoid(logits)
            batch_size = labels.shape[0]
            total_loss += float(criterion(logits, labels).item()) * batch_size

            label_values = labels.cpu().to(torch.int64).tolist()
            logit_values = logits.cpu().tolist()
            probability_values = probabilities.cpu().tolist()
            metadata = batch.get("metadata", {})
            for index in range(batch_size):
                label = int(label_values[index])
                probability = float(probability_values[index])
                labels_all.append(label)
                probabilities_all.append(probability)
                predictions.append(
                    {
                        "sample_id": _metadata_at(metadata, "sample_id", index),
                        "source": _metadata_at(metadata, "source", index),
                        "tier": _metadata_at(metadata, "tier", index),
                        "strategy": _metadata_at(metadata, "strategy", index),
                        "negative_strategy": _metadata_at(
                            metadata, "negative_strategy", index
                        ),
                        "split": _metadata_at(metadata, "split", index),
                        "pmhc_id": _metadata_at(metadata, "pmhc_id", index),
                        "label": label,
                        "logit": float(logit_values[index]),
                        "probability": probability,
                    }
                )

    if not labels_all:
        raise ValueError("evaluation dataloader is empty")
    metrics = binary_classification_metrics(labels_all, probabilities_all)
    return EvaluationResult(
        loss=total_loss / len(labels_all),
        auroc=metrics["auroc"],
        auprc=metrics["auprc"],
        sample_count=len(labels_all),
        predictions=predictions,
    )


def _checkpoint_payload(
    model: nn.Module,
    model_config: Mapping[str, Any],
    amino_acid_vocab: AminoAcidVocab,
    hla_vocab: HlaVocab,
    seed: int,
    data_spec: DatasetSpec,
    epoch: int,
    validation: EvaluationResult,
    optimizer: torch.optim.Optimizer,
    dynamic_sampling_state: Mapping[str, Any] | None = None,
    dynamic_sampling_audit: Mapping[str, Any] | None = None,
    validation_query_macro: Mapping[str, float | int] | None = None,
    early_stopping_metric: str = "validation_auprc",
) -> dict[str, Any]:
    state_dict = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    payload = {
        "schema_version": 2,
        "model_state_dict": state_dict,
        "optimizer_state_dict": _optimizer_state_to_cpu(optimizer.state_dict()),
        "rng_state": capture_rng_state(),
        "model_config": dict(model_config),
        "amino_acid_vocab": {
            "token_to_id": dict(amino_acid_vocab.token_to_id),
            "padding_id": PAD_ID,
            "unknown_id": UNK_ID,
        },
        "hla_to_id": dict(hla_vocab.hla_to_id),
        "seed": seed,
        "data_spec": asdict(data_spec),
        "epoch": epoch,
        "best_epoch": epoch,
        "validation_metrics": validation.metrics_dict(),
        "early_stopping_metric": early_stopping_metric,
    }
    if validation_query_macro is not None:
        payload["validation_query_macro_metrics"] = dict(validation_query_macro)
    if dynamic_sampling_state is not None:
        payload["dynamic_sampling_state"] = dict(dynamic_sampling_state)
    if dynamic_sampling_audit is not None:
        payload["dynamic_sampling_audit"] = dict(dynamic_sampling_audit)
    return payload


def _optimizer_state_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _optimizer_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_optimizer_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_optimizer_state_to_cpu(item) for item in value)
    return value


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise ValueError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def dynamic_sampling_checkpoint_state(
    train_loader: DataLoader,
) -> dict[str, Any] | None:
    state_method = getattr(train_loader, "state_dict", None)
    if not callable(state_method):
        return None
    state = state_method()
    if not isinstance(state, dict):
        raise ValueError("dynamic DataLoader state_dict must return a mapping")
    return state


def restore_dynamic_sampling_state(
    checkpoint: Mapping[str, Any],
    train_loader: DataLoader,
    *,
    advance_to_next_epoch: bool = True,
) -> int:
    state = checkpoint.get("dynamic_sampling_state")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint has no dynamic sampling state")
    load_method = getattr(train_loader, "load_state_dict", None)
    if not callable(load_method):
        raise ValueError("train DataLoader cannot restore dynamic sampling state")
    load_method(state)
    dataset_state = state.get("dataset")
    if not isinstance(dataset_state, Mapping):
        raise ValueError("checkpoint dynamic dataset state is invalid")
    next_epoch = int(dataset_state["next_epoch"])
    if advance_to_next_epoch:
        set_epoch_method = getattr(train_loader, "set_epoch", None)
        if not callable(set_epoch_method):
            raise ValueError("train DataLoader cannot advance dynamic epoch")
        set_epoch_method(next_epoch)
    return next_epoch


def restore_training_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "optimizer_state_dict" not in checkpoint:
        raise ValueError("checkpoint has no optimizer state")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    next_epoch = restore_dynamic_sampling_state(checkpoint, train_loader)
    if int(checkpoint.get("epoch", -1)) + 1 != next_epoch:
        raise ValueError("checkpoint epoch and dynamic sampling state disagree")
    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, Mapping):
        raise ValueError("checkpoint has no RNG state")
    restore_rng_state(rng_state)
    return checkpoint, next_epoch


def write_training_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    temporary_path.replace(path)


def write_dynamic_sampling_audit(
    path: Path, entries: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=DYNAMIC_SAMPLING_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(entries)
    temporary_path.replace(path)


def _dynamic_sampling_audit_entry(
    train_loader: DataLoader,
    previous_keys: set[tuple[str, str, str]] | None,
) -> tuple[dict[str, Any] | None, set[tuple[str, str, str]] | None]:
    state = dynamic_sampling_checkpoint_state(train_loader)
    if state is None:
        return None, previous_keys
    dataset = getattr(train_loader, "dataset", None)
    selected_keys = getattr(dataset, "selected_negative_keys", None)
    statistics = getattr(dataset, "selection_statistics", None)
    if selected_keys is None or not isinstance(statistics, Mapping):
        raise ValueError("dynamic DataLoader does not expose sampling audit data")
    keys = set(selected_keys)
    if len(keys) != len(selected_keys):
        raise ValueError("dynamic epoch contains duplicate negative interactions")
    if len(keys) != int(statistics.get("selected_negative_pairs", -1)):
        raise ValueError("dynamic sampling statistics disagree with selected rows")
    overlap = None
    if previous_keys is not None:
        overlap = len(keys & previous_keys) / len(keys)
        if overlap >= 1.0:
            raise ValueError("dynamic negative selection did not change between epochs")
    dataset_state = state["dataset"]
    entry = {
        "epoch": int(dataset_state["current_epoch"]),
        "sampling_seed": int(dataset_state["base_seed"])
        + int(dataset_state["current_epoch"]),
        **{column: int(statistics[column]) for column in DYNAMIC_SAMPLING_COLUMNS[2:8]},
        "selection_sha256": str(dataset_state["selection_sha256"]),
        "overlap_with_previous_ratio": overlap,
    }
    return entry, keys


def fit_with_early_stopping(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    model_config: Mapping[str, Any],
    amino_acid_vocab: AminoAcidVocab,
    hla_vocab: HlaVocab,
    data_spec: DatasetSpec,
    seed: int,
    learning_rate: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 5,
    early_stopping_metric: str = "validation_auprc",
) -> TrainingResult:
    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs and patience must be positive")
    if early_stopping_metric not in {
        "validation_auprc",
        "validation_query_macro_auprc",
    }:
        raise ValueError(f"unsupported early stopping metric: {early_stopping_metric}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    history_path = output_dir / "training_history.csv"
    dynamic_sampling_audit_path = output_dir / "negative_sampling_history.csv"
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_score = -float("inf")
    best_epoch = 0
    best_validation: dict[str, float | int] = {}
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    dynamic_sampling_audit: list[dict[str, Any]] = []
    previous_dynamic_keys: set[tuple[str, str, str]] | None = None

    for epoch in range(1, max_epochs + 1):
        set_epoch_method = getattr(train_loader, "set_epoch", None)
        if callable(set_epoch_method):
            set_epoch_method(epoch)
        sampling_entry, previous_dynamic_keys = _dynamic_sampling_audit_entry(
            train_loader, previous_dynamic_keys
        )
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_function, device
        )
        validation = evaluate_model(
            model, validation_loader, device, loss_function=loss_function
        )
        validation_query_macro = None
        if early_stopping_metric == "validation_query_macro_auprc":
            validation_query_macro, _ = query_level_classification_metrics(
                validation.predictions
            )
            selection_value = float(validation_query_macro["auprc"])
        else:
            selection_value = validation.auprc
        is_best = selection_value > best_score
        if is_best:
            best_score = selection_value
            best_epoch = epoch
            best_validation = validation.metrics_dict()
            if validation_query_macro is not None:
                best_validation["query_macro"] = dict(validation_query_macro)
            best_validation["early_stopping_metric"] = early_stopping_metric
            best_validation["early_stopping_value"] = selection_value
            epochs_without_improvement = 0
            payload = _checkpoint_payload(
                model,
                model_config,
                amino_acid_vocab,
                hla_vocab,
                seed,
                data_spec,
                epoch,
                validation,
                optimizer,
                dynamic_sampling_checkpoint_state(train_loader),
                sampling_entry,
                validation_query_macro,
                early_stopping_metric,
            )
            temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary_checkpoint)
            temporary_checkpoint.replace(checkpoint_path)
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation.loss,
                "validation_auroc": validation.auroc,
                "validation_auprc": validation.auprc,
                "validation_query_macro_auroc": (
                    validation_query_macro["auroc"]
                    if validation_query_macro is not None
                    else None
                ),
                "validation_query_macro_auprc": (
                    validation_query_macro["auprc"]
                    if validation_query_macro is not None
                    else None
                ),
                "early_stopping_metric": early_stopping_metric,
                "early_stopping_value": selection_value,
                "is_best": is_best,
            }
        )
        write_training_history(history_path, history)
        if sampling_entry is not None:
            dynamic_sampling_audit.append(sampling_entry)
            write_dynamic_sampling_audit(
                dynamic_sampling_audit_path, dynamic_sampling_audit
            )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation": validation.metrics_dict(),
                    "validation_query_macro": validation_query_macro,
                    "early_stopping_metric": early_stopping_metric,
                    "early_stopping_value": selection_value,
                    "is_best": is_best,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if epochs_without_improvement >= patience:
            break

    return TrainingResult(
        best_epoch=best_epoch,
        best_validation=best_validation,
        history=history,
        checkpoint_path=checkpoint_path,
        dynamic_sampling_audit_path=(
            dynamic_sampling_audit_path if dynamic_sampling_audit else None
        ),
    )


def load_model_from_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[TwoTowerCnn, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TwoTowerCnn(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PREDICTION_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(predictions)


def write_per_query_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PER_QUERY_METRIC_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
