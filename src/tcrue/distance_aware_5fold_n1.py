from __future__ import annotations

import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data_tiers import PAIR_KEY_COLUMNS
from .distance_aware_5fold import FOLD_NAMES
from .distance_aware_n1 import (
    _partition_audit,
    canonical_negative_sha256,
)
from .negative_sampling import (
    GLOBAL_PARTITION_PRIORITY,
    PARTITIONS,
    _read_positive_csv,
    _write_negative_csv,
    deduplicate_n1_across_partitions,
    generate_partition_negatives,
    load_positive_registry,
)
from .vdjdb import file_sha256


PairKey = tuple[str, str, str]
Pmhc = tuple[str, str]
SEED_MODULUS = 2_147_483_647


def _pair_key(record: Mapping[str, Any]) -> PairKey:
    return tuple(str(record[column]) for column in PAIR_KEY_COLUMNS)  # type: ignore[return-value]


def derive_partition_seed(base_seed: int, fold_id: str, partition: str) -> int:
    if fold_id not in FOLD_NAMES:
        raise ValueError(f"unknown five-fold ID: {fold_id}")
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition: {partition}")
    payload = f"{base_seed}\0{fold_id}\0{partition}\0N1".encode("ascii")
    derived = int(hashlib.sha256(payload).hexdigest()[:16], 16) % SEED_MODULUS
    return derived or 1


def _positive_conflict_candidate_count(
    positives: Sequence[Mapping[str, str]], positive_registry: set[PairKey]
) -> int:
    pmhc_pool = {(record["epitope"], record["hla"]) for record in positives}
    registry_by_tcr: dict[str, set[Pmhc]] = defaultdict(set)
    for tcr, epitope, hla in positive_registry:
        registry_by_tcr[tcr].add((epitope, hla))
    rejected = 0
    for anchor in positives:
        source_pmhc = (anchor["epitope"], anchor["hla"])
        rejected += len((registry_by_tcr[anchor["tcr_cdr3_beta"]] & pmhc_pool) - {source_pmhc})
    return rejected


def generate_fold_n1(
    partition_positives: Mapping[str, list[dict[str, str]]],
    positive_registry: set[PairKey],
    *,
    base_seed: int,
    fold_id: str,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, int]],
    dict[str, dict[str, Any]],
    dict[str, int],
]:
    if set(partition_positives) != set(PARTITIONS):
        raise ValueError("five-fold N1 requires train, validation, and test")
    generated: dict[str, list[dict[str, Any]]] = {}
    rejection_stats: dict[str, dict[str, Any]] = {}
    derived_seeds = {
        partition: derive_partition_seed(base_seed, fold_id, partition)
        for partition in PARTITIONS
    }
    for partition in PARTITIONS:
        positives = partition_positives[partition]
        generated[partition], skipped = generate_partition_negatives(
            positives,
            positive_registry,
            strategy="N1",
            seed=derived_seeds[partition],
            context={
                "tier": "development",
                "positive_split_strategy": (
                    f"distance_aware_5fold_rebalanced:{fold_id}"
                ),
                "partition": partition,
            },
        )
        rejection_stats[partition] = {
            "positive_conflict_rejections": _positive_conflict_candidate_count(
                positives, positive_registry
            ),
            "unmatched_anchor_rejections": sum(skipped.values()),
            "unmatched_anchors_by_reason": skipped,
        }

    deduplicated, deduplication = deduplicate_n1_across_partitions(generated)
    for partition in PARTITIONS:
        duplicate_rejections = deduplication[partition][
            "cross_partition_duplicates_removed"
        ]
        rejection_stats[partition]["duplicate_rejections"] = duplicate_rejections
        expected = len(partition_positives[partition])
        observed = len(deduplicated[partition])
        if observed != expected:
            raise ValueError(
                f"five-fold N1 failed to retain 1:1 in {fold_id}/{partition}: "
                f"{observed}/{expected}; no refill or fallback is allowed"
            )
    return deduplicated, deduplication, rejection_stats, derived_seeds


def _validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "tier": "development",
        "fold_count": 5,
        "positive_split_strategy": "distance_aware_5fold_rebalanced",
        "negative_strategy": "N1",
        "negative_to_positive_ratio": 1,
        "partitions": list(PARTITIONS),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"five-fold N1 config requires {key}={value!r}")
    if not isinstance(config.get("base_seed"), int):
        raise ValueError("five-fold N1 base_seed must be an integer")
    if config["seed_derivation"] != {
        "algorithm": "sha256",
        "namespace": "base_seed|fold_id|partition|N1",
        "modulus": SEED_MODULUS,
    }:
        raise ValueError("five-fold N1 seed derivation configuration mismatch")
    sampling = config["sampling"]
    required_sampling = {
        "definition": "random_other_pmhc",
        "tcr_source": "current_fold_current_partition_only",
        "pmhc_source": "current_fold_current_partition_only",
        "known_positive_exclusion": "complete_positive_registry",
        "on_shortfall": "error",
        "allow_fallback": False,
        "cross_fold_interaction_exclusion": False,
    }
    for key, value in required_sampling.items():
        if sampling.get(key) != value:
            raise ValueError(f"five-fold N1 sampling requires {key}={value!r}")
    deduplication = sampling["within_fold_deduplication"]
    if deduplication["interaction_key"] != list(PAIR_KEY_COLUMNS):
        raise ValueError("five-fold N1 interaction key mismatch")
    if deduplication["partition_priority"] != list(GLOBAL_PARTITION_PRIORITY):
        raise ValueError("five-fold N1 partition priority mismatch")


def _load_fold_positives(
    fold_id: str,
    positive_manifest: Mapping[str, Any],
    positive_directory: Path,
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, Path],
    dict[str, str],
]:
    partition_positives = {}
    positive_paths = {}
    hashes = {}
    for partition in PARTITIONS:
        source = positive_manifest["folds"][fold_id]["partitions"][partition]
        path = positive_directory / source["file"]
        actual_sha = file_sha256(path)
        if actual_sha != source["sha256"]:
            raise ValueError(f"positive split SHA-256 mismatch: {fold_id}/{partition}")
        records = _read_positive_csv(path)
        if len(records) != source["positive_pairs"]:
            raise ValueError(f"positive split row-count mismatch: {fold_id}/{partition}")
        partition_positives[partition] = records
        positive_paths[partition] = path
        hashes[partition] = actual_sha
    peptide_sets = {
        partition: {record["epitope"] for record in records}
        for partition, records in partition_positives.items()
    }
    for index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[index + 1 :]:
            if peptide_sets[left] & peptide_sets[right]:
                raise ValueError(f"positive peptide leakage in {fold_id}: {left}/{right}")
    return partition_positives, positive_paths, hashes


def render_report(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Development rebalanced distance-aware five-fold N1 negatives",
        "",
        "## Definition",
        "",
        f"- Base seed: `{manifest['base_seed']}`",
        "- Fold/partition seeds are derived with SHA-256.",
        "- TCR and pMHC sources are restricted to the current fold and partition.",
        "- Every candidate in the complete positive registry is excluded.",
        "- Deduplication applies within each fold; interactions may recur across folds.",
        "- No refill, fallback, N2 generation, training, or positive modification is allowed.",
        "",
        "## Fold and partition statistics",
        "",
        "| Fold | Partition | Positives | N1 negatives | Positive-conflict rejections | Duplicate rejections | Output SHA-256 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for fold_id in FOLD_NAMES:
        for partition in PARTITIONS:
            values = manifest["folds"][fold_id]["partitions"][partition]
            lines.append(
                f"| {fold_id} | {partition} | {values['positive_pairs']:,} | "
                f"{values['negative_pairs']:,} | "
                f"{values['positive_conflict_rejections']:,} | "
                f"{values['duplicate_rejections']:,} | "
                f"`{values['negative_output']['sha256']}` |"
            )
    lines.extend(
        [
            "",
            "## Audit",
            "",
            "- Every fold/partition is exactly 1:1.",
            "- Negative intersections with the complete positive registry: 0.",
            "- Duplicate interaction keys within partitions and across partitions of one fold: 0.",
            "- Peptide uses outside the corresponding partition: 0.",
            "- Positive input SHA-256 values are unchanged.",
            "- Two complete deterministic rebuilds produced identical SHA-256 values.",
            "",
            "This report is generated by `scripts/build_distance_aware_5fold_n1.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_distance_aware_five_fold_n1(
    config_path: Path, project_dir: Path, *, force: bool = False
) -> dict[str, Any]:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("five-fold N1 config must be a mapping")
    _validate_config(config)
    registry_path = project_dir / config["positive_registry"]["path"]
    positive_manifest_path = project_dir / config["positive_splits"]["manifest"]
    positive_directory = project_dir / config["positive_splits"]["directory"]
    if file_sha256(registry_path) != config["positive_registry"]["sha256"]:
        raise ValueError("complete positive registry SHA-256 mismatch")
    if file_sha256(positive_manifest_path) != config["positive_splits"][
        "manifest_sha256"
    ]:
        raise ValueError("rebalanced five-fold positive manifest SHA-256 mismatch")
    with positive_manifest_path.open("r", encoding="utf-8") as handle:
        positive_manifest = json.load(handle)
    if (
        positive_manifest.get("tier") != "development"
        or positive_manifest.get("status") != "active"
        or positive_manifest["bucket_allocation"]["objective"] != "joint_weighted"
        or positive_manifest.get("negative_samples_generated") is not False
    ):
        raise ValueError("positive manifest is not the active rebalanced positive-only split")

    output_directory = project_dir / config["outputs"]["directory"]
    pattern = config["outputs"]["csv_pattern"]
    output_paths = {
        fold_id: {
            partition: output_directory
            / pattern.format(fold=fold_id, partition=partition)
            for partition in PARTITIONS
        }
        for fold_id in FOLD_NAMES
    }
    manifest_path = project_dir / config["outputs"]["manifest_json"]
    report_path = project_dir / config["outputs"]["report_md"]
    all_outputs = [
        path for values in output_paths.values() for path in values.values()
    ] + [manifest_path, report_path]
    if not force and any(path.exists() for path in all_outputs):
        raise FileExistsError("refusing to overwrite five-fold N1 outputs")

    positive_registry = load_positive_registry(registry_path)
    base_seed = int(config["base_seed"])
    folds_manifest: dict[str, Any] = {}
    positive_hashes_before: dict[str, dict[str, str]] = {}
    first_rebuild_hashes: dict[str, dict[str, str]] = {}
    second_rebuild_hashes: dict[str, dict[str, str]] = {}
    for fold_id in FOLD_NAMES:
        positives, positive_paths, positive_hashes = _load_fold_positives(
            fold_id, positive_manifest, positive_directory
        )
        positive_hashes_before[fold_id] = positive_hashes
        partition_peptides = {
            partition: {record["epitope"] for record in records}
            for partition, records in positives.items()
        }
        first, deduplication, rejection_stats, derived_seeds = generate_fold_n1(
            positives,
            positive_registry,
            base_seed=base_seed,
            fold_id=fold_id,
        )
        first_rebuild_hashes[fold_id] = {
            partition: canonical_negative_sha256(first[partition])
            for partition in PARTITIONS
        }

        partition_manifest: dict[str, Any] = {}
        key_sets: dict[str, set[PairKey]] = {}
        output_directory.mkdir(parents=True, exist_ok=True)
        for partition in PARTITIONS:
            negatives = first[partition]
            path = output_paths[fold_id][partition]
            _write_negative_csv(path, negatives)
            if file_sha256(path) != first_rebuild_hashes[fold_id][partition]:
                raise ValueError(f"non-canonical N1 output: {fold_id}/{partition}")
            audit = _partition_audit(
                partition,
                positives[partition],
                negatives,
                positive_registry,
                partition_peptides,
            )
            keys = {_pair_key(record) for record in negatives}
            key_sets[partition] = keys
            positive_count = len(positives[partition])
            rejection = rejection_stats[partition]
            partition_manifest[partition] = {
                "derived_seed": derived_seeds[partition],
                "positive_input": {
                    "file": positive_paths[partition].name,
                    "sha256": positive_hashes[partition],
                },
                "negative_output": {
                    "file": path.name,
                    "sha256": file_sha256(path),
                },
                "positive_pairs": positive_count,
                "negative_pairs": len(negatives),
                "positive_fraction": positive_count / (positive_count + len(negatives)),
                "positive_conflict_rejections": rejection[
                    "positive_conflict_rejections"
                ],
                "duplicate_rejections": rejection["duplicate_rejections"],
                "unmatched_anchor_rejections": rejection[
                    "unmatched_anchor_rejections"
                ],
                "unmatched_anchors_by_reason": rejection[
                    "unmatched_anchors_by_reason"
                ],
                "generated_before_within_fold_deduplication": deduplication[
                    partition
                ]["generated_before_global_deduplication"],
                "audit": audit,
            }
        cross_intersections = {}
        for index, left in enumerate(PARTITIONS):
            for right in PARTITIONS[index + 1 :]:
                overlap = len(key_sets[left] & key_sets[right])
                cross_intersections[f"{left}_vs_{right}"] = overlap
                if overlap:
                    raise ValueError(
                        f"within-fold N1 interaction overlap: {fold_id}/{left}/{right}"
                    )
        folds_manifest[fold_id] = {
            "derived_seeds": derived_seeds,
            "partitions": partition_manifest,
            "invariants": {
                "partition_interaction_key_intersections": cross_intersections,
                "known_positive_registry_intersections": 0,
                "peptide_leakage_uses": 0,
                "all_partitions_one_to_one": True,
            },
        }
        del first, negatives, key_sets
        gc.collect()

        second, second_deduplication, second_rejections, second_seeds = generate_fold_n1(
            positives,
            positive_registry,
            base_seed=base_seed,
            fold_id=fold_id,
        )
        second_rebuild_hashes[fold_id] = {
            partition: canonical_negative_sha256(second[partition])
            for partition in PARTITIONS
        }
        if (
            first_rebuild_hashes[fold_id] != second_rebuild_hashes[fold_id]
            or deduplication != second_deduplication
            or rejection_stats != second_rejections
            or derived_seeds != second_seeds
        ):
            raise ValueError(f"five-fold N1 reproducibility failure: {fold_id}")
        del second, positives, partition_peptides
        gc.collect()

    positive_hashes_after = {
        fold_id: {
            partition: file_sha256(
                positive_directory
                / positive_manifest["folds"][fold_id]["partitions"][partition]["file"]
            )
            for partition in PARTITIONS
        }
        for fold_id in FOLD_NAMES
    }
    if positive_hashes_after != positive_hashes_before:
        raise ValueError("rebalanced five-fold positive inputs changed during N1 generation")
    if first_rebuild_hashes != second_rebuild_hashes:
        raise ValueError("complete N1 rebuild SHA-256 mismatch")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "base_seed": base_seed,
        "tier": "development",
        "fold_count": 5,
        "positive_split_strategy": "distance_aware_5fold_rebalanced",
        "negative_strategy": "N1",
        "negative_to_positive_ratio": 1,
        "config": {
            "path": config_path.relative_to(project_dir).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "positive_registry": {
            "path": config["positive_registry"]["path"],
            "sha256": file_sha256(registry_path),
            "positive_pairs": len(positive_registry),
        },
        "positive_splits_manifest": {
            "path": config["positive_splits"]["manifest"],
            "sha256": file_sha256(positive_manifest_path),
        },
        "seed_derivation": config["seed_derivation"],
        "sampling": config["sampling"],
        "folds": folds_manifest,
        "reproducibility": {
            "complete_rebuild_count": 2,
            "sha256_identical": True,
            "negative_csv_sha256": first_rebuild_hashes,
        },
        "invariants": {
            "positive_inputs_unchanged": True,
            "all_folds_one_to_one": True,
            "known_positive_registry_intersections": 0,
            "within_fold_duplicate_interactions": 0,
            "peptide_leakage_uses": 0,
            "cross_fold_interaction_exclusion_required": False,
        },
        "artifacts": {
            "manifest_json": config["outputs"]["manifest_json"],
            "report_md": config["outputs"]["report_md"],
        },
        "positive_splits_modified": False,
        "n2_generated": False,
        "models_trained_or_evaluated": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(manifest), encoding="utf-8")
    return manifest
