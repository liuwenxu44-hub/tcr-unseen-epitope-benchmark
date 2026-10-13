from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data_tiers import PAIR_KEY_COLUMNS
from .negative_sampling import load_positive_registry
from .vdjdb import file_sha256


PARTITIONS = ("train", "validation", "test")
PARTITION_PRIORITY = ("test", "validation", "train")
SOURCE_STRATEGIES = ("N1", "N2")
SOURCE_VALUES = ("N1", "N2", "both")
MIXED_COLUMNS = [
    "negative_id",
    "tcr_cdr3_beta",
    "epitope",
    "hla",
    "pmhc_id",
    "label",
    "source",
    "tier",
    "positive_split_strategy",
    "partition",
    "sampling_seed",
    "source_n1_negative_id",
    "source_n2_negative_id",
]

PairKey = tuple[str, str, str]


@dataclass(frozen=True)
class MixedCandidate:
    key: PairKey
    partition: str
    source: str
    n1_negative_id: str = ""
    n2_negative_id: str = ""


def _pair_key(record: Mapping[str, str]) -> PairKey:
    return (
        record["tcr_cdr3_beta"].strip().upper(),
        record["epitope"].strip().upper(),
        record["hla"].strip().upper(),
    )


def _stable_hash(seed: int, *parts: str) -> str:
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def _relative(project_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(project_dir.resolve()).as_posix()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input has no header: {path}")
        return list(reader)


def _verify_source_rows(
    path: Path,
    entry: Mapping[str, Any],
    *,
    strategy: str,
    partition: str,
) -> list[dict[str, str]]:
    observed_sha = file_sha256(path)
    if observed_sha != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch for existing {strategy} {partition} input")
    rows = _read_csv(path)
    if len(rows) != int(entry["generated_negative_pairs"]):
        raise ValueError(f"row count mismatch for existing {strategy} {partition} input")
    keys = [_pair_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate keys within existing {strategy} {partition} input")
    for row in rows:
        if row["label"] != "0":
            raise ValueError(f"non-negative label in existing {strategy} input")
        if row["negative_strategy"] != strategy:
            raise ValueError(f"strategy mismatch in existing {strategy} input")
        if row["partition"] != partition:
            raise ValueError(f"partition mismatch in existing {strategy} input")
        if row["tier"] != "development" or row["positive_split_strategy"] != "unseen_epitope":
            raise ValueError("existing negative input belongs to the wrong experiment")
    return rows


def merge_and_deduplicate_candidates(
    partition_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]],
) -> tuple[dict[str, list[MixedCandidate]], dict[str, dict[str, int]]]:
    if set(partition_rows) != set(PARTITIONS):
        raise ValueError("mixed deduplication requires train, validation, and test")
    priority_index = {partition: index for index, partition in enumerate(PARTITION_PRIORITY)}
    occurrences: dict[PairKey, list[tuple[str, str, Mapping[str, str]]]] = defaultdict(list)
    for partition in PARTITIONS:
        if set(partition_rows[partition]) != set(SOURCE_STRATEGIES):
            raise ValueError(f"mixed inputs are incomplete for {partition}")
        for strategy in SOURCE_STRATEGIES:
            seen: set[PairKey] = set()
            for row in partition_rows[partition][strategy]:
                key = _pair_key(row)
                if key in seen:
                    raise ValueError(f"duplicate {strategy} key within {partition}")
                seen.add(key)
                occurrences[key].append((partition, strategy, row))

    retained: dict[str, list[MixedCandidate]] = {partition: [] for partition in PARTITIONS}
    statistics = {
        partition: {
            "input_n1_rows": len(partition_rows[partition]["N1"]),
            "input_n2_rows": len(partition_rows[partition]["N2"]),
            "within_partition_n1_n2_overlap_keys": 0,
            "same_partition_duplicate_rows_removed": 0,
            "cross_partition_duplicate_rows_removed": 0,
            "total_duplicate_rows_removed": 0,
            "retained_unique_candidates": 0,
        }
        for partition in PARTITIONS
    }
    for key in sorted(occurrences):
        rows = occurrences[key]
        owner = min((partition for partition, _, _ in rows), key=priority_index.get)
        strategies = {strategy for _, strategy, _ in rows}
        source = "both" if strategies == set(SOURCE_STRATEGIES) else next(iter(strategies))
        n1_rows = [row for _, strategy, row in rows if strategy == "N1"]
        n2_rows = [row for _, strategy, row in rows if strategy == "N2"]
        retained[owner].append(
            MixedCandidate(
                key=key,
                partition=owner,
                source=source,
                n1_negative_id=sorted(row["negative_id"] for row in n1_rows)[0]
                if n1_rows
                else "",
                n2_negative_id=sorted(row["negative_id"] for row in n2_rows)[0]
                if n2_rows
                else "",
            )
        )

        by_partition: dict[str, list[str]] = defaultdict(list)
        for partition, strategy, _ in rows:
            by_partition[partition].append(strategy)
        for partition, partition_strategies in by_partition.items():
            if set(partition_strategies) == set(SOURCE_STRATEGIES):
                statistics[partition]["within_partition_n1_n2_overlap_keys"] += 1
            if partition == owner:
                statistics[partition]["same_partition_duplicate_rows_removed"] += (
                    len(partition_strategies) - 1
                )
            else:
                statistics[partition]["cross_partition_duplicate_rows_removed"] += len(
                    partition_strategies
                )

    for partition in PARTITIONS:
        retained[partition].sort(key=lambda candidate: candidate.key)
        statistics[partition]["retained_unique_candidates"] = len(retained[partition])
        statistics[partition]["total_duplicate_rows_removed"] = (
            statistics[partition]["same_partition_duplicate_rows_removed"]
            + statistics[partition]["cross_partition_duplicate_rows_removed"]
        )
    key_sets = {
        partition: {candidate.key for candidate in candidates}
        for partition, candidates in retained.items()
    }
    for index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[index + 1 :]:
            if key_sets[left] & key_sets[right]:
                raise ValueError(f"mixed candidate deduplication failed: {left} vs {right}")
    return retained, statistics


def _deterministic_select(
    candidates: Sequence[MixedCandidate],
    count: int,
    *,
    seed: int,
    partition: str,
    source: str,
) -> list[MixedCandidate]:
    if count > len(candidates):
        raise ValueError(f"insufficient {source} candidates in {partition}")
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            _stable_hash(seed, partition, source, *candidate.key),
            candidate.key,
        ),
    )
    return ranked[:count]


def select_mixed_candidates(
    candidates: Sequence[MixedCandidate],
    target: int,
    *,
    seed: int,
    partition: str,
) -> tuple[list[MixedCandidate], dict[str, int]]:
    buckets = {
        source: [candidate for candidate in candidates if candidate.source == source]
        for source in SOURCE_VALUES
    }
    feasible_both = [
        both_count
        for both_count in range(min(len(buckets["both"]), target), -1, -1)
        if (target - both_count) % 2 == 0
        and (target - both_count) // 2 <= len(buckets["N1"])
        and (target - both_count) // 2 <= len(buckets["N2"])
    ]
    if not feasible_both:
        raise ValueError(
            f"cannot satisfy exact mixed quota in {partition}: target={target}, "
            f"available={{'N1': {len(buckets['N1'])}, 'N2': {len(buckets['N2'])}, "
            f"'both': {len(buckets['both'])}}}"
        )
    both_count = feasible_both[0]
    exclusive_count = (target - both_count) // 2
    selected = []
    selected.extend(
        _deterministic_select(
            buckets["N1"], exclusive_count, seed=seed, partition=partition, source="N1"
        )
    )
    selected.extend(
        _deterministic_select(
            buckets["N2"], exclusive_count, seed=seed, partition=partition, source="N2"
        )
    )
    selected.extend(
        _deterministic_select(
            buckets["both"], both_count, seed=seed, partition=partition, source="both"
        )
    )
    selected.sort(key=lambda candidate: candidate.key)
    counts = Counter(candidate.source for candidate in selected)
    statistics = {
        "selected_n1_only": counts["N1"],
        "selected_n2_only": counts["N2"],
        "selected_both": counts["both"],
        "n1_source_memberships": counts["N1"] + counts["both"],
        "n2_source_memberships": counts["N2"] + counts["both"],
        "selected_negative_pairs": len(selected),
    }
    if len(selected) != target:
        raise ValueError(f"mixed negative target was not met in {partition}")
    if statistics["selected_n1_only"] != statistics["selected_n2_only"]:
        raise ValueError(f"exclusive N1:N2 ratio is not 1:1 in {partition}")
    if statistics["n1_source_memberships"] != statistics["n2_source_memberships"]:
        raise ValueError(f"source-membership N1:N2 ratio is not 1:1 in {partition}")
    return selected, statistics


def _mixed_record(candidate: MixedCandidate, *, seed: int) -> dict[str, Any]:
    tcr, epitope, hla = candidate.key
    payload = "\0".join((candidate.partition, *candidate.key)).encode("utf-8")
    return {
        "negative_id": f"mixed-negative:{hashlib.sha256(payload).hexdigest()[:20]}",
        "tcr_cdr3_beta": tcr,
        "epitope": epitope,
        "hla": hla,
        "pmhc_id": f"{epitope}|{hla}",
        "label": 0,
        "source": candidate.source,
        "tier": "development",
        "positive_split_strategy": "unseen_epitope",
        "partition": candidate.partition,
        "sampling_seed": seed,
        "source_n1_negative_id": candidate.n1_negative_id,
        "source_n2_negative_id": candidate.n2_negative_id,
    }


def candidate_selection_sha256(candidates: Sequence[MixedCandidate]) -> str:
    digest = hashlib.sha256()
    for candidate in sorted(candidates, key=lambda item: item.key):
        digest.update(
            "\t".join(
                (
                    candidate.partition,
                    candidate.source,
                    *candidate.key,
                    candidate.n1_negative_id,
                    candidate.n2_negative_id,
                )
            ).encode("utf-8")
            + b"\n"
        )
    return digest.hexdigest()


def validate_mixed_invariants(
    selected: Mapping[str, Sequence[MixedCandidate]],
    positive_registry: set[PairKey],
    positive_peptides: Mapping[str, set[str]],
) -> dict[str, Any]:
    key_sets = {
        partition: {candidate.key for candidate in selected[partition]}
        for partition in PARTITIONS
    }
    duplicate_counts = {
        partition: len(selected[partition]) - len(key_sets[partition])
        for partition in PARTITIONS
    }
    pairwise_intersections = {}
    for index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[index + 1 :]:
            pairwise_intersections[f"{left}_{right}"] = len(
                key_sets[left] & key_sets[right]
            )
    registry_intersections = {
        partition: len(key_sets[partition] & positive_registry)
        for partition in PARTITIONS
    }
    peptide_leakage = {
        partition: sum(
            candidate.key[1] not in positive_peptides[partition]
            for candidate in selected[partition]
        )
        for partition in PARTITIONS
    }
    positive_peptide_intersections = {}
    for index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[index + 1 :]:
            positive_peptide_intersections[f"{left}_{right}"] = len(
                positive_peptides[left] & positive_peptides[right]
            )
    invariants = {
        "duplicate_keys_within_partition": duplicate_counts,
        "cross_partition_key_intersections": pairwise_intersections,
        "positive_registry_intersections": registry_intersections,
        "negative_peptides_outside_partition": peptide_leakage,
        "positive_peptide_partition_intersections": positive_peptide_intersections,
    }
    if any(
        value
        for group in invariants.values()
        for value in group.values()
    ):
        raise ValueError(f"mixed negative invariant failure: {invariants}")
    return invariants


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MIXED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _validate_config(config: Mapping[str, Any]) -> None:
    expected_experiment = {
        "tier": "development",
        "positive_split_strategy": "unseen_epitope",
    }
    if config.get("schema_version") != 1 or not isinstance(config.get("seed"), int):
        raise ValueError("mixed negative config schema or seed is invalid")
    if config.get("experiment") != expected_experiment:
        raise ValueError("mixed negative config targets the wrong experiment")
    if config.get("interaction_key") != list(PAIR_KEY_COLUMNS):
        raise ValueError("mixed interaction key is invalid")
    if config.get("partition_priority") != list(PARTITION_PRIORITY):
        raise ValueError("mixed partition priority must be test, validation, train")
    mixture = config.get("mixture", {})
    if mixture.get("negative_to_positive_ratio") != 1 or mixture.get("n1_to_n2_ratio") != 1:
        raise ValueError("mixed negative ratios must both be one")
    if mixture.get("both_policy") != "maximize_feasible":
        raise ValueError("mixed both policy must maximize feasible records")
    if mixture.get("both_counts_toward_each_source_membership") is not True:
        raise ValueError("source=both must count toward both source memberships")


def render_mixed_report(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Development unseen_epitope mixed N1+N2 negatives",
        "",
        "## Definition",
        "",
        f"- Seed: `{manifest['seed']}`",
        "- Inputs: existing repaired N1 and N2 CSVs only; no negative candidates are regenerated",
        "- Interaction key: `(tcr_cdr3_beta, epitope, hla)`",
        "- Global owner priority: test > validation > train",
        "- Shared N1/N2 interactions are retained once with `source=both`",
        "- N1-only and N2-only counts are equal; `both` counts toward both provenance memberships",
        "",
        "## Statistics",
        "",
        "| Partition | Positives | Mixed negatives | N1-only | N2-only | Both | N1 memberships | N2 memberships | Duplicate rows removed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for partition in PARTITIONS:
        entry = manifest["partitions"][partition]
        selection = entry["selection"]
        deduplication = entry["deduplication"]
        lines.append(
            f"| {partition} | {entry['positive_input']['positive_pairs']:,} | "
            f"{selection['selected_negative_pairs']:,} | "
            f"{selection['selected_n1_only']:,} | {selection['selected_n2_only']:,} | "
            f"{selection['selected_both']:,} | "
            f"{selection['n1_source_memberships']:,} | "
            f"{selection['n2_source_memberships']:,} | "
            f"{deduplication['total_duplicate_rows_removed']:,} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            "- Cross-partition interaction intersections: 0",
            "- Complete positive registry intersections: 0",
            "- Unseen peptide leakage: 0",
            "- Duplicate mixed negatives: 0",
            "",
            "This report is generated by `scripts/build_mixed_negatives.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_mixed_negatives(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    project_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _validate_config(config)
    seed = int(config["seed"])
    inputs = config["inputs"]
    negative_dir = _resolve(project_dir, inputs["negative_directory"])
    positive_dir = _resolve(project_dir, inputs["positive_split_directory"])
    negative_manifest_path = _resolve(
        project_dir, inputs["negative_sampling_manifest"]
    )
    positive_manifest_path = _resolve(project_dir, inputs["positive_splits_manifest"])
    registry_path = _resolve(project_dir, inputs["positive_registry"])
    negative_manifest = json.loads(negative_manifest_path.read_text(encoding="utf-8"))
    positive_manifest = json.loads(positive_manifest_path.read_text(encoding="utf-8"))
    if negative_manifest.get("seed") != seed or positive_manifest.get("seed") != seed:
        raise ValueError("input manifest seed mismatch")
    if negative_manifest["positive_registry"]["sha256"] != file_sha256(registry_path):
        raise ValueError("positive registry SHA-256 mismatch")
    if negative_manifest["positive_splits_manifest"]["sha256"] != file_sha256(
        positive_manifest_path
    ):
        raise ValueError("negative manifest references another positive split manifest")

    negative_entries = negative_manifest["tiers"]["development"]["unseen_epitope"]
    positive_entries = positive_manifest["tiers"]["development"]["unseen_epitope"][
        "splits"
    ]
    partition_rows: dict[str, dict[str, list[dict[str, str]]]] = {}
    positive_peptides: dict[str, set[str]] = {}
    input_audit: dict[str, Any] = {}
    positive_audit: dict[str, Any] = {}
    for partition in PARTITIONS:
        partition_rows[partition] = {}
        for strategy in SOURCE_STRATEGIES:
            entry = negative_entries[partition]["negatives"][strategy]
            path = negative_dir / entry["file"]
            rows = _verify_source_rows(
                path, entry, strategy=strategy, partition=partition
            )
            partition_rows[partition][strategy] = rows
            input_audit[f"{partition}_{strategy.lower()}"] = {
                "path": _relative(project_dir, path),
                "sha256": file_sha256(path),
                "rows": len(rows),
            }
        positive_entry = positive_entries[partition]
        positive_path = positive_dir / positive_entry["file"]
        if file_sha256(positive_path) != positive_entry["sha256"]:
            raise ValueError(f"positive split SHA-256 mismatch for {partition}")
        positive_rows = _read_csv(positive_path)
        if len(positive_rows) != int(positive_entry["positive_pairs"]):
            raise ValueError(f"positive split row count mismatch for {partition}")
        positive_peptides[partition] = {
            row["epitope"].strip().upper() for row in positive_rows
        }
        positive_audit[partition] = {
            "path": _relative(project_dir, positive_path),
            "sha256": file_sha256(positive_path),
            "positive_pairs": len(positive_rows),
        }

    retained, deduplication = merge_and_deduplicate_candidates(partition_rows)
    selected: dict[str, list[MixedCandidate]] = {}
    selection_stats: dict[str, dict[str, int]] = {}
    for partition in PARTITIONS:
        target = positive_audit[partition]["positive_pairs"]
        selected[partition], selection_stats[partition] = select_mixed_candidates(
            retained[partition], target, seed=seed, partition=partition
        )
    positive_registry = load_positive_registry(registry_path)
    invariants = validate_mixed_invariants(
        selected, positive_registry, positive_peptides
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    partition_manifest: dict[str, Any] = {}
    for partition in PARTITIONS:
        path = output_dir / f"development_unseen_epitope_{partition}_mixed.csv"
        records = [_mixed_record(candidate, seed=seed) for candidate in selected[partition]]
        _write_csv(path, records)
        partition_manifest[partition] = {
            "positive_input": positive_audit[partition],
            "negative_inputs": {
                strategy: input_audit[f"{partition}_{strategy.lower()}"]
                for strategy in SOURCE_STRATEGIES
            },
            "deduplication": deduplication[partition],
            "selection": {
                **selection_stats[partition],
                "selection_sha256": candidate_selection_sha256(
                    selected[partition]
                ),
                "target_negative_pairs": positive_audit[partition]["positive_pairs"],
                "positive_fraction": 0.5,
                "available_candidates_by_source": dict(
                    sorted(Counter(candidate.source for candidate in retained[partition]).items())
                ),
            },
            "output": {
                "file": path.name,
                "sha256": file_sha256(path),
                "rows": len(records),
            },
        }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "experiment": config["experiment"],
        "interaction_key": config["interaction_key"],
        "partition_priority": config["partition_priority"],
        "mixture": config["mixture"],
        "config": {
            "path": _relative(project_dir, config_path),
            "sha256": file_sha256(config_path),
        },
        "input_manifests": {
            "negative_sampling": {
                "path": _relative(project_dir, negative_manifest_path),
                "sha256": file_sha256(negative_manifest_path),
            },
            "positive_splits": {
                "path": _relative(project_dir, positive_manifest_path),
                "sha256": file_sha256(positive_manifest_path),
            },
            "positive_registry": {
                "path": _relative(project_dir, registry_path),
                "sha256": file_sha256(registry_path),
                "positive_pairs": len(positive_registry),
            },
        },
        "partitions": partition_manifest,
        "invariants": invariants,
        "summary": {
            partition: {
                "positive_pairs": positive_audit[partition]["positive_pairs"],
                **selection_stats[partition],
                "duplicate_rows_removed": deduplication[partition][
                    "total_duplicate_rows_removed"
                ],
            }
            for partition in PARTITIONS
        },
        "candidate_negatives_regenerated": False,
    }
    manifest_path = output_dir / config["outputs"]["manifest_json"]
    report_path = output_dir / config["outputs"]["report_md"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(render_mixed_report(manifest), encoding="utf-8")
    return manifest
