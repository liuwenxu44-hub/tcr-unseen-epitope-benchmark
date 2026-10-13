from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data_tiers import PAIR_KEY_COLUMNS
from .vdjdb import file_sha256


NEGATIVE_STRATEGIES = ("N1", "N2")
TIERS = ("development", "strict")
POSITIVE_SPLIT_STRATEGIES = ("random", "unseen_epitope")
PARTITIONS = ("train", "validation", "test")
GLOBAL_PARTITION_PRIORITY = ("test", "validation", "train")
N1_PARTITION_PRIORITY = GLOBAL_PARTITION_PRIORITY
N2_PARTITION_PRIORITY = GLOBAL_PARTITION_PRIORITY
REQUIRED_POSITIVE_COLUMNS = {*PAIR_KEY_COLUMNS, "pmhc_id", "record_id", "label"}
NEGATIVE_COLUMNS = [
    "negative_id",
    "tcr_cdr3_beta",
    "epitope",
    "hla",
    "pmhc_id",
    "label",
    "negative_strategy",
    "tier",
    "positive_split_strategy",
    "partition",
    "source_positive_record_id",
    "source_positive_epitope",
    "source_positive_hla",
    "source_positive_pmhc_id",
    "sampling_seed",
]

PairKey = tuple[str, str, str]
Pmhc = tuple[str, str]


def _pair_key(record: dict[str, str]) -> PairKey:
    return (
        record["tcr_cdr3_beta"],
        record["epitope"],
        record["hla"],
    )


def _pmhc(record: dict[str, str]) -> Pmhc:
    return (record["epitope"], record["hla"])


def _stable_hash(seed: int, *parts: str) -> str:
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_positive_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input has no header: {path}")
        missing = sorted(REQUIRED_POSITIVE_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        records = list(reader)

    seen: set[PairKey] = set()
    for record in records:
        key = _pair_key(record)
        if key in seen:
            raise ValueError(f"Duplicate positive pair: {key}")
        seen.add(key)
        if record["label"] != "1":
            raise ValueError(f"Expected positive label, observed: {record['label']}")
        expected_pmhc = f"{record['epitope']}|{record['hla']}"
        if record["pmhc_id"] != expected_pmhc:
            raise ValueError(f"Invalid pMHC id for {key}")
    return records


def load_positive_registry(path: Path) -> set[PairKey]:
    return {_pair_key(record) for record in _read_positive_csv(path)}


def _augment_match(
    anchor_index: int,
    candidate_lists: dict[int, list[Pmhc]],
    candidate_owner: dict[Pmhc, int],
    anchor_choice: dict[int, Pmhc],
    visited: set[Pmhc],
) -> bool:
    for candidate in candidate_lists[anchor_index]:
        if candidate in visited:
            continue
        visited.add(candidate)
        previous_owner = candidate_owner.get(candidate)
        if previous_owner is None or _augment_match(
            previous_owner,
            candidate_lists,
            candidate_owner,
            anchor_choice,
            visited,
        ):
            candidate_owner[candidate] = anchor_index
            anchor_choice[anchor_index] = candidate
            return True
    return False


def _negative_id(
    context: dict[str, str], strategy: str, key: PairKey
) -> str:
    payload = "\0".join(
        (
            context["tier"],
            context["positive_split_strategy"],
            context["partition"],
            strategy,
            *key,
        )
    ).encode("utf-8")
    return f"putative-negative:{hashlib.sha256(payload).hexdigest()[:20]}"


def generate_partition_negatives(
    positives: list[dict[str, str]],
    positive_registry: set[PairKey],
    *,
    strategy: str,
    seed: int,
    context: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if strategy not in NEGATIVE_STRATEGIES:
        raise ValueError(f"Unknown negative strategy: {strategy}")

    pmhc_pool = sorted({_pmhc(record) for record in positives})
    pmhcs_by_hla: dict[str, list[Pmhc]] = defaultdict(list)
    for pmhc in pmhc_pool:
        pmhcs_by_hla[pmhc[1]].append(pmhc)
    anchors_by_tcr: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in positives:
        anchors_by_tcr[record["tcr_cdr3_beta"]].append(record)

    matched: list[tuple[dict[str, str], Pmhc]] = []
    skipped: Counter[str] = Counter()
    namespace = (
        f"{context['tier']}|{context['positive_split_strategy']}|"
        f"{context['partition']}|{strategy}"
    )

    for tcr in sorted(anchors_by_tcr):
        anchors = sorted(
            anchors_by_tcr[tcr],
            key=lambda record: (
                _stable_hash(seed, namespace, record["record_id"]),
                record["record_id"],
            ),
        )
        candidate_lists: dict[int, list[Pmhc]] = {}
        initial_failure: dict[int, str] = {}
        for index, anchor in enumerate(anchors):
            source_pmhc = _pmhc(anchor)
            if strategy == "N1":
                raw_candidates = [pmhc for pmhc in pmhc_pool if pmhc != source_pmhc]
                no_candidate_reason = "no_other_pmhc_in_partition"
            else:
                raw_candidates = [
                    pmhc
                    for pmhc in pmhcs_by_hla.get(anchor["hla"], [])
                    if pmhc[0] != anchor["epitope"]
                ]
                no_candidate_reason = "no_same_hla_different_peptide"

            if not raw_candidates:
                candidate_lists[index] = []
                initial_failure[index] = no_candidate_reason
                continue
            candidates = [
                pmhc
                for pmhc in raw_candidates
                if (tcr, pmhc[0], pmhc[1]) not in positive_registry
            ]
            if not candidates:
                candidate_lists[index] = []
                initial_failure[index] = "all_candidates_known_positive"
                continue
            candidates.sort()
            offset = int(
                _stable_hash(seed, namespace, anchor["record_id"]), 16
            ) % len(candidates)
            candidate_lists[index] = candidates[offset:] + candidates[:offset]

        candidate_owner: dict[Pmhc, int] = {}
        anchor_choice: dict[int, Pmhc] = {}
        for index in range(len(anchors)):
            if not candidate_lists[index]:
                continue
            _augment_match(
                index,
                candidate_lists,
                candidate_owner,
                anchor_choice,
                set(),
            )

        for index, anchor in enumerate(anchors):
            candidate = anchor_choice.get(index)
            if candidate is None:
                skipped[initial_failure.get(index, "candidate_capacity_exhausted")] += 1
                continue
            matched.append((anchor, candidate))

    negatives: list[dict[str, Any]] = []
    for anchor, (epitope, hla) in matched:
        key = (anchor["tcr_cdr3_beta"], epitope, hla)
        negatives.append(
            {
                "negative_id": _negative_id(context, strategy, key),
                "tcr_cdr3_beta": key[0],
                "epitope": epitope,
                "hla": hla,
                "pmhc_id": f"{epitope}|{hla}",
                "label": 0,
                "negative_strategy": strategy,
                "tier": context["tier"],
                "positive_split_strategy": context["positive_split_strategy"],
                "partition": context["partition"],
                "source_positive_record_id": anchor["record_id"],
                "source_positive_epitope": anchor["epitope"],
                "source_positive_hla": anchor["hla"],
                "source_positive_pmhc_id": anchor["pmhc_id"],
                "sampling_seed": seed,
            }
        )
    negatives.sort(
        key=lambda record: (
            record["epitope"],
            record["hla"],
            record["tcr_cdr3_beta"],
            record["source_positive_record_id"],
        )
    )
    if strategy == "N1" and len(negatives) != len(positives):
        raise ValueError(
            f"N1 failed to reach 1:1 in {namespace}: "
            f"{len(negatives)}/{len(positives)}, skipped={dict(skipped)}"
        )
    return negatives, dict(sorted(skipped.items()))


def deduplicate_negatives_across_partitions(
    partition_negatives: dict[str, list[dict[str, Any]]],
    *,
    strategy: str,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, int]],
]:
    if strategy not in NEGATIVE_STRATEGIES:
        raise ValueError(f"Unknown negative strategy: {strategy}")
    if set(partition_negatives) != set(PARTITIONS):
        raise ValueError(f"{strategy} global deduplication requires all three partitions")

    retained: dict[str, list[dict[str, Any]]] = {}
    statistics: dict[str, dict[str, int]] = {}
    seen: set[PairKey] = set()
    for partition in GLOBAL_PARTITION_PRIORITY:
        records = partition_negatives[partition]
        keys = [_pair_key(record) for record in records]
        if len(keys) != len(set(keys)):
            raise ValueError(
                f"Duplicate {strategy} pair within {partition} before global deduplication"
            )

        kept_records = []
        for record, key in zip(records, keys):
            if key in seen:
                continue
            seen.add(key)
            kept_records.append(record)
        retained[partition] = kept_records
        statistics[partition] = {
            "generated_before_global_deduplication": len(records),
            "cross_partition_duplicates_removed": len(records) - len(kept_records),
            "generated_negative_pairs": len(kept_records),
        }

    key_sets = {
        partition: {_pair_key(record) for record in records}
        for partition, records in retained.items()
    }
    for index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[index + 1 :]:
            if key_sets[left] & key_sets[right]:
                raise ValueError(
                    f"{strategy} global deduplication failed: {left} vs {right}"
                )
    return retained, statistics


def deduplicate_n1_across_partitions(
    partition_negatives: dict[str, list[dict[str, Any]]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, int]],
]:
    return deduplicate_negatives_across_partitions(
        partition_negatives, strategy="N1"
    )


def deduplicate_n2_across_partitions(
    partition_negatives: dict[str, list[dict[str, Any]]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, int]],
]:
    return deduplicate_negatives_across_partitions(
        partition_negatives, strategy="N2"
    )


def _write_negative_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NEGATIVE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _negative_invariants(
    negatives: list[dict[str, Any]],
    positive_registry: set[PairKey],
    partition_peptides: set[str],
    forbidden_unseen_peptides: set[str],
    *,
    strategy: str,
    unseen_epitope: bool,
) -> dict[str, int]:
    keys = [
        (record["tcr_cdr3_beta"], record["epitope"], record["hla"])
        for record in negatives
    ]
    invariants = {
        "duplicate_negative_pair_keys": len(keys) - len(set(keys)),
        "known_positive_registry_intersections": len(set(keys) & positive_registry),
        "negative_peptides_outside_partition": sum(
            record["epitope"] not in partition_peptides for record in negatives
        ),
        "unseen_forbidden_peptide_uses": sum(
            record["epitope"] in forbidden_unseen_peptides for record in negatives
        )
        if unseen_epitope
        else 0,
        "source_pmhc_not_changed": sum(
            record["pmhc_id"] == record["source_positive_pmhc_id"]
            for record in negatives
        ),
        "n2_hla_mismatches": sum(
            record["hla"] != record["source_positive_hla"]
            for record in negatives
        )
        if strategy == "N2"
        else 0,
        "n2_same_peptide_violations": sum(
            record["epitope"] == record["source_positive_epitope"]
            for record in negatives
        )
        if strategy == "N2"
        else 0,
    }
    if any(invariants.values()):
        raise ValueError(f"Negative invariant failure: {invariants}")
    return invariants


def _negative_stats(
    path: Path,
    negatives: list[dict[str, Any]],
    target: int,
    skipped: dict[str, int],
    invariants: dict[str, int],
) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": file_sha256(path),
        "target_negative_pairs": target,
        "generated_negative_pairs": len(negatives),
        "coverage": len(negatives) / target if target else 0.0,
        "skipped": skipped,
        "invariants": invariants,
    }


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("negative sampling config schema_version must be 1")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    if config.get("negative_to_positive_ratio") != 1:
        raise ValueError("negative_to_positive_ratio must be 1")
    if config.get("tiers") != list(TIERS):
        raise ValueError("tiers must be development, strict")
    if config.get("positive_split_strategies") != list(POSITIVE_SPLIT_STRATEGIES):
        raise ValueError("positive_split_strategies must be random, unseen_epitope")
    if config.get("partitions") != list(PARTITIONS):
        raise ValueError("partitions must be train, validation, test")
    strategies = config.get("negative_strategies", {})
    n1 = strategies.get("N1", {})
    if n1.get("definition") != "random_other_pmhc":
        raise ValueError("N1 definition must be random_other_pmhc")
    n2 = strategies.get("N2", {})
    if n2.get("definition") != "exact_hla_different_peptide":
        raise ValueError("N2 definition must be exact_hla_different_peptide")
    if n2.get("on_no_candidate") != "skip" or n2.get("allow_n1_fallback") is not False:
        raise ValueError("N2 must skip without N1 fallback")
    for strategy, strategy_config in (("N1", n1), ("N2", n2)):
        global_deduplication = strategy_config.get("global_deduplication", {})
        if global_deduplication.get("interaction_key") != list(PAIR_KEY_COLUMNS):
            raise ValueError(
                f"{strategy} global deduplication must use the complete interaction key"
            )
        if global_deduplication.get("partition_priority") != list(
            GLOBAL_PARTITION_PRIORITY
        ):
            raise ValueError(
                f"{strategy} partition priority must be test, validation, train"
            )


def render_negative_sampling_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Putative negative sampling",
        "",
        "## Definition",
        "",
        f"- Seed: `{manifest['seed']}`",
        "- Target negative-to-positive ratio: 1:1 within every partition",
        "- N1: pair each source TCR with another pMHC from the same partition",
        "- N2: exact source HLA with a different peptide from the same partition",
        "- Every candidate is excluded if present in the complete positive registry",
        "- N2 failures are skipped and never downgraded to N1",
        "- N1 and N2 interaction keys are independently deduplicated with priority: test > validation > train",
        "",
        "## Statistics",
        "",
        "| Tier | Positive split | Partition | Positives | N1 | N1 removed | N1 positive fraction | N2 | N2 removed | N2 positive fraction |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in TIERS:
        for positive_strategy in POSITIVE_SPLIT_STRATEGIES:
            for partition in PARTITIONS:
                entry = manifest["tiers"][tier][positive_strategy][partition]
                n1 = entry["negatives"]["N1"]
                n2 = entry["negatives"]["N2"]
                n1_deduplication = n1["global_deduplication"]
                n2_deduplication = n2["global_deduplication"]
                lines.append(
                    f"| {tier.title()} | {positive_strategy} | {partition} | "
                    f"{entry['positive_input']['positive_pairs']:,} | "
                    f"{n1['generated_negative_pairs']:,} | "
                    f"{n1_deduplication['cross_partition_duplicates_removed']:,} | "
                    f"{n1['class_balance']['positive_fraction']:.4f} | "
                    f"{n2['generated_negative_pairs']:,} | "
                    f"{n2_deduplication['cross_partition_duplicates_removed']:,} | "
                    f"{n2['class_balance']['positive_fraction']:.4f} |"
                )
    summary = manifest["summary"]
    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- N1: {summary['N1']['generated_negative_pairs']:,} / {summary['N1']['target_negative_pairs']:,} ({summary['N1']['coverage']:.4f})",
            f"- N1 cross-partition duplicates removed: {summary['N1']['cross_partition_duplicates_removed']:,}",
            f"- N2: {summary['N2']['generated_negative_pairs']:,} / {summary['N2']['target_negative_pairs']:,} ({summary['N2']['coverage']:.4f})",
            f"- N2 cross-partition duplicates removed: {summary['N2']['cross_partition_duplicates_removed']:,}",
            "- Duplicate negatives: 0",
            "- Known-positive intersections: 0",
            "- Unseen-epitope forbidden peptide uses: 0",
            "",
            "This report is generated by `scripts/build_negative_samples.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_negative_samples(
    config_path: Path,
    config: dict[str, Any],
    positive_registry_path: Path,
    positive_splits_manifest_path: Path,
    positive_split_directory: Path,
    output_dir: Path,
    manifest_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    _validate_config(config)
    seed = int(config["seed"])
    positive_registry = load_positive_registry(positive_registry_path)
    positive_splits_manifest = json.loads(
        positive_splits_manifest_path.read_text(encoding="utf-8")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "negative_to_positive_ratio": 1,
        "config": {"file": config_path.name, "sha256": file_sha256(config_path)},
        "positive_registry": {
            "file": positive_registry_path.name,
            "sha256": file_sha256(positive_registry_path),
            "positive_pairs": len(positive_registry),
        },
        "positive_splits_manifest": {
            "file": positive_splits_manifest_path.name,
            "sha256": file_sha256(positive_splits_manifest_path),
        },
        "n1_global_deduplication": {
            "interaction_key": list(PAIR_KEY_COLUMNS),
            "partition_priority": list(N1_PARTITION_PRIORITY),
        },
        "n2_global_deduplication": {
            "interaction_key": list(PAIR_KEY_COLUMNS),
            "partition_priority": list(N2_PARTITION_PRIORITY),
        },
        "tiers": {},
        "artifacts": {
            "manifest_json": manifest_path.name,
            "report_md": report_path.name,
        },
    }
    totals = {
        strategy: {
            "target_negative_pairs": 0,
            "generated_negative_pairs": 0,
            "cross_partition_duplicates_removed": 0,
        }
        for strategy in NEGATIVE_STRATEGIES
    }

    for tier in TIERS:
        tier_manifest: dict[str, Any] = {}
        for positive_strategy in POSITIVE_SPLIT_STRATEGIES:
            partition_records: dict[str, list[dict[str, str]]] = {}
            for partition in PARTITIONS:
                source_info = positive_splits_manifest["tiers"][tier][positive_strategy][
                    "splits"
                ][partition]
                source_path = positive_split_directory / source_info["file"]
                if file_sha256(source_path) != source_info["sha256"]:
                    raise ValueError(f"Positive split checksum mismatch: {source_path}")
                partition_records[partition] = _read_positive_csv(source_path)

            partition_peptides = {
                partition: {record["epitope"] for record in records}
                for partition, records in partition_records.items()
            }
            generated: dict[str, dict[str, list[dict[str, Any]]]] = {}
            skipped_by_partition: dict[str, dict[str, dict[str, int]]] = {}
            for partition in PARTITIONS:
                positives = partition_records[partition]
                context = {
                    "tier": tier,
                    "positive_split_strategy": positive_strategy,
                    "partition": partition,
                }
                generated[partition] = {}
                skipped_by_partition[partition] = {}
                for negative_strategy in NEGATIVE_STRATEGIES:
                    negatives, skipped = generate_partition_negatives(
                        positives,
                        positive_registry,
                        strategy=negative_strategy,
                        seed=seed,
                        context=context,
                    )
                    generated[partition][negative_strategy] = negatives
                    skipped_by_partition[partition][negative_strategy] = skipped

            deduplication_by_strategy: dict[
                str, dict[str, dict[str, int]]
            ] = {}
            for negative_strategy in NEGATIVE_STRATEGIES:
                deduplicated, deduplication = (
                    deduplicate_negatives_across_partitions(
                        {
                            partition: generated[partition][negative_strategy]
                            for partition in PARTITIONS
                        },
                        strategy=negative_strategy,
                    )
                )
                deduplication_by_strategy[negative_strategy] = deduplication
                for partition in PARTITIONS:
                    generated[partition][negative_strategy] = deduplicated[partition]

            strategy_manifest: dict[str, Any] = {}
            for partition in PARTITIONS:
                positives = partition_records[partition]
                source_info = positive_splits_manifest["tiers"][tier][positive_strategy][
                    "splits"
                ][partition]
                forbidden_peptides = set().union(
                    *(
                        peptides
                        for other_partition, peptides in partition_peptides.items()
                        if other_partition != partition
                    )
                )
                negative_manifest: dict[str, Any] = {}
                for negative_strategy in NEGATIVE_STRATEGIES:
                    negatives = generated[partition][negative_strategy]
                    output_path = output_dir / (
                        f"{tier}_{positive_strategy}_{partition}_"
                        f"{negative_strategy.lower()}.csv"
                    )
                    _write_negative_csv(output_path, negatives)
                    invariants = _negative_invariants(
                        negatives,
                        positive_registry,
                        partition_peptides[partition],
                        forbidden_peptides,
                        strategy=negative_strategy,
                        unseen_epitope=positive_strategy == "unseen_epitope",
                    )
                    negative_manifest[negative_strategy] = _negative_stats(
                        output_path,
                        negatives,
                        len(positives),
                        skipped_by_partition[partition][negative_strategy],
                        invariants,
                    )
                    deduplication = deduplication_by_strategy[negative_strategy][
                        partition
                    ]
                    negative_manifest[negative_strategy]["global_deduplication"] = {
                        "partition_priority": list(GLOBAL_PARTITION_PRIORITY),
                        **deduplication,
                    }
                    negative_manifest[negative_strategy]["class_balance"] = {
                        "positive_pairs": len(positives),
                        "negative_pairs": len(negatives),
                        "positive_fraction": len(positives)
                        / (len(positives) + len(negatives)),
                    }
                    totals[negative_strategy][
                        "cross_partition_duplicates_removed"
                    ] += deduplication["cross_partition_duplicates_removed"]
                    totals[negative_strategy]["target_negative_pairs"] += len(positives)
                    totals[negative_strategy]["generated_negative_pairs"] += len(negatives)
                strategy_manifest[partition] = {
                    "positive_input": {
                        "file": source_info["file"],
                        "sha256": source_info["sha256"],
                        "positive_pairs": len(positives),
                    },
                    "negatives": negative_manifest,
                }
            tier_manifest[positive_strategy] = strategy_manifest
        manifest["tiers"][tier] = tier_manifest

    manifest["summary"] = {
        strategy: {
            **values,
            "coverage": values["generated_negative_pairs"]
            / values["target_negative_pairs"],
        }
        for strategy, values in totals.items()
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_negative_sampling_report(manifest), encoding="utf-8")
    return manifest
