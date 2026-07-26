from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data_tiers import PAIR_KEY_COLUMNS
from .negative_sampling import (
    GLOBAL_PARTITION_PRIORITY,
    NEGATIVE_COLUMNS,
    PARTITIONS,
    _negative_invariants,
    _read_positive_csv,
    _write_negative_csv,
    deduplicate_n1_across_partitions,
    generate_partition_negatives,
    load_positive_registry,
)
from .vdjdb import file_sha256


PairKey = tuple[str, str, str]


def _pair_key(record: Mapping[str, Any]) -> PairKey:
    return tuple(str(record[column]) for column in PAIR_KEY_COLUMNS)  # type: ignore[return-value]


def _csv_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    import csv

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=NEGATIVE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode("utf-8")


def canonical_negative_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_csv_bytes(records)).hexdigest()


def _validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "tier": "development",
        "positive_split_strategy": "distance_aware",
        "negative_strategy": "N1",
        "negative_to_positive_ratio": 1,
        "partitions": list(PARTITIONS),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"distance-aware N1 config requires {key}={value!r}")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")

    sampling = config.get("sampling", {})
    if sampling.get("definition") != "random_other_pmhc":
        raise ValueError("N1 definition must be random_other_pmhc")
    if sampling.get("tcr_source") != "current_partition_only":
        raise ValueError("N1 TCRs must come from the current partition")
    if sampling.get("pmhc_source") != "current_partition_only":
        raise ValueError("N1 pMHCs must come from the current partition")
    if sampling.get("known_positive_exclusion") != "complete_positive_registry":
        raise ValueError("N1 must use the complete positive registry")
    if sampling.get("on_shortfall") != "error":
        raise ValueError("N1 shortfall handling must be error")
    if sampling.get("allow_fallback") is not False:
        raise ValueError("N1 fallback must be disabled")
    deduplication = sampling.get("global_deduplication", {})
    if deduplication.get("interaction_key") != list(PAIR_KEY_COLUMNS):
        raise ValueError("global deduplication must use the complete interaction key")
    if deduplication.get("partition_priority") != list(GLOBAL_PARTITION_PRIORITY):
        raise ValueError("partition priority must be test, validation, train")


def generate_distance_aware_n1_partitions(
    partition_positives: Mapping[str, list[dict[str, str]]],
    positive_registry: set[PairKey],
    *,
    seed: int,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
]:
    if set(partition_positives) != set(PARTITIONS):
        raise ValueError("distance-aware N1 requires train, validation, and test")

    generated: dict[str, list[dict[str, Any]]] = {}
    rejected: dict[str, dict[str, int]] = {}
    for partition in PARTITIONS:
        positives = partition_positives[partition]
        generated[partition], rejected[partition] = generate_partition_negatives(
            positives,
            positive_registry,
            strategy="N1",
            seed=seed,
            context={
                "tier": "development",
                "positive_split_strategy": "distance_aware",
                "partition": partition,
            },
        )

    deduplicated, deduplication = deduplicate_n1_across_partitions(generated)
    for partition in PARTITIONS:
        expected = len(partition_positives[partition])
        observed = len(deduplicated[partition])
        if observed != expected:
            raise ValueError(
                f"distance-aware N1 failed to retain 1:1 in {partition}: "
                f"{observed}/{expected}; no resampling or fallback is allowed"
            )
    return deduplicated, deduplication, rejected


def _partition_audit(
    partition: str,
    positives: list[dict[str, str]],
    negatives: list[dict[str, Any]],
    positive_registry: set[PairKey],
    partition_peptides: Mapping[str, set[str]],
) -> dict[str, int]:
    forbidden_peptides = set().union(
        *(peptides for name, peptides in partition_peptides.items() if name != partition)
    )
    invariants = _negative_invariants(
        negatives,
        positive_registry,
        partition_peptides[partition],
        forbidden_peptides,
        strategy="N1",
        unseen_epitope=True,
    )
    tcrs = {record["tcr_cdr3_beta"] for record in positives}
    pmhcs = {(record["epitope"], record["hla"]) for record in positives}
    positive_ids = {record["record_id"] for record in positives}
    provenance = {
        "negative_tcrs_outside_partition": sum(
            record["tcr_cdr3_beta"] not in tcrs for record in negatives
        ),
        "negative_pmhcs_outside_partition": sum(
            (record["epitope"], record["hla"]) not in pmhcs
            for record in negatives
        ),
        "source_records_outside_partition": sum(
            record["source_positive_record_id"] not in positive_ids
            for record in negatives
        ),
    }
    if any(provenance.values()):
        raise ValueError(f"distance-aware N1 provenance failure: {provenance}")
    return {**invariants, **provenance}


def render_distance_aware_n1_report(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Development distance-aware N1 negative sampling",
        "",
        "## Definition",
        "",
        f"- Seed: `{manifest['seed']}`",
        "- N1 pairs each source TCR with another pMHC from the same partition.",
        "- Candidates found in the complete positive registry are rejected.",
        "- No resampling, fallback, N2 generation, or positive-split modification is allowed.",
        "- Interaction key: `(tcr_cdr3_beta, epitope, hla)`.",
        "",
        "## Partition statistics",
        "",
        "| Partition | Positives | N1 negatives | Positive fraction | Cross-partition duplicates removed | Rejected anchors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for partition in PARTITIONS:
        entry = manifest["partitions"][partition]
        lines.append(
            f"| {partition} | {entry['positive_pairs']:,} | "
            f"{entry['negative_pairs']:,} | {entry['positive_fraction']:.4f} | "
            f"{entry['cross_partition_duplicates_removed']:,} | "
            f"{entry['rejected_anchor_count']:,} |"
        )
    lines.extend(
        [
            "",
            "## Audit",
            "",
            "- Positive input files unchanged: yes",
            "- Duplicate interaction keys within or across partitions: 0",
            "- Intersections with the complete positive registry: 0",
            "- Peptide uses outside the corresponding partition: 0",
            "- N1 TCR and pMHC sources outside the corresponding partition: 0",
            "- Every partition retains a 1:1 positive-to-negative ratio",
            "",
            "This report is generated by `scripts/build_distance_aware_n1.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_distance_aware_n1(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    positive_registry_path: Path,
    positive_manifest_path: Path,
    positive_directory: Path,
    output_directory: Path,
    manifest_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    _validate_config(config)
    expected_registry_sha = config["positive_registry"]["sha256"]
    expected_manifest_sha = config["positive_splits"]["manifest_sha256"]
    if file_sha256(positive_registry_path) != expected_registry_sha:
        raise ValueError("complete positive registry checksum mismatch")
    if file_sha256(positive_manifest_path) != expected_manifest_sha:
        raise ValueError("distance-aware positive manifest checksum mismatch")

    positive_manifest = json.loads(positive_manifest_path.read_text(encoding="utf-8"))
    if positive_manifest.get("tier") != "development":
        raise ValueError("positive manifest is not the Development tier")
    if positive_manifest.get("negative_samples_generated") is not False:
        raise ValueError("positive manifest does not describe a positive-only split")

    partition_positives: dict[str, list[dict[str, str]]] = {}
    positive_paths: dict[str, Path] = {}
    positive_hashes_before: dict[str, str] = {}
    for partition in PARTITIONS:
        source = positive_manifest["partitions"][partition]
        source_path = positive_directory / source["file"]
        source_sha = file_sha256(source_path)
        if source_sha != source["sha256"]:
            raise ValueError(f"positive split checksum mismatch: {source_path}")
        records = _read_positive_csv(source_path)
        if len(records) != source["positive_pairs"]:
            raise ValueError(f"positive split row-count mismatch: {source_path}")
        partition_positives[partition] = records
        positive_paths[partition] = source_path
        positive_hashes_before[partition] = source_sha

    partition_peptides = {
        partition: {record["epitope"] for record in records}
        for partition, records in partition_positives.items()
    }
    for index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[index + 1 :]:
            if partition_peptides[left] & partition_peptides[right]:
                raise ValueError(f"positive peptide leakage: {left} vs {right}")

    positive_registry = load_positive_registry(positive_registry_path)
    seed = int(config["seed"])
    negatives, deduplication, rejected = generate_distance_aware_n1_partitions(
        partition_positives, positive_registry, seed=seed
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    partition_manifest: dict[str, Any] = {}
    key_sets: dict[str, set[PairKey]] = {}
    for partition in PARTITIONS:
        records = negatives[partition]
        output_path = output_directory / (
            f"development_distance_aware_{partition}_n1.csv"
        )
        _write_negative_csv(output_path, records)
        if canonical_negative_sha256(records) != file_sha256(output_path):
            raise ValueError(f"non-canonical N1 CSV output: {output_path}")
        audit = _partition_audit(
            partition,
            partition_positives[partition],
            records,
            positive_registry,
            partition_peptides,
        )
        keys = {_pair_key(record) for record in records}
        key_sets[partition] = keys
        rejected_count = sum(rejected[partition].values())
        positive_count = len(partition_positives[partition])
        partition_manifest[partition] = {
            "positive_input": {
                "file": positive_paths[partition].name,
                "sha256": positive_hashes_before[partition],
            },
            "negative_output": {
                "file": output_path.name,
                "sha256": file_sha256(output_path),
            },
            "positive_pairs": positive_count,
            "negative_pairs": len(records),
            "positive_fraction": positive_count / (positive_count + len(records)),
            "generated_before_global_deduplication": deduplication[partition][
                "generated_before_global_deduplication"
            ],
            "cross_partition_duplicates_removed": deduplication[partition][
                "cross_partition_duplicates_removed"
            ],
            "rejected_anchor_count": rejected_count,
            "rejected_anchors_by_reason": rejected[partition],
            "audit": audit,
        }

    cross_intersections: dict[str, int] = {}
    for index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[index + 1 :]:
            overlap = len(key_sets[left] & key_sets[right])
            cross_intersections[f"{left}_vs_{right}"] = overlap
            if overlap:
                raise ValueError(f"cross-partition N1 duplication: {left} vs {right}")

    positive_hashes_after = {
        partition: file_sha256(path) for partition, path in positive_paths.items()
    }
    if positive_hashes_after != positive_hashes_before:
        raise ValueError("distance-aware positive split changed during N1 generation")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "tier": "development",
        "positive_split_strategy": "distance_aware",
        "negative_strategy": "N1",
        "negative_to_positive_ratio": 1,
        "config": {
            "file": config_path.name,
            "sha256": file_sha256(config_path),
        },
        "positive_registry": {
            "path": str(config["positive_registry"]["path"]),
            "sha256": file_sha256(positive_registry_path),
            "positive_pairs": len(positive_registry),
        },
        "positive_splits_manifest": {
            "path": str(config["positive_splits"]["manifest"]),
            "sha256": file_sha256(positive_manifest_path),
        },
        "sampling": config["sampling"],
        "partitions": partition_manifest,
        "invariants": {
            "positive_inputs_unchanged": True,
            "cross_partition_interaction_key_intersections": cross_intersections,
            "known_positive_registry_intersections": 0,
            "peptide_leakage_uses": 0,
            "all_partitions_one_to_one": True,
        },
        "models_trained_or_evaluated": False,
        "n2_generated": False,
        "artifacts": {
            "manifest_json": str(config["outputs"]["manifest_json"]),
            "report_md": str(config["outputs"]["report_md"]),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_distance_aware_n1_report(manifest), encoding="utf-8")
    return manifest
