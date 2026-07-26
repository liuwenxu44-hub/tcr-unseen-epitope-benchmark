from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from .data_tiers import PAIR_KEY_COLUMNS
from .vdjdb import file_sha256


SPLIT_NAMES = ("train", "validation", "test")
STRATEGY_NAMES = ("random", "unseen_epitope")
REQUIRED_COLUMNS = {*PAIR_KEY_COLUMNS, "epitope", "pmhc_id", "label"}


def _pair_key(record: dict[str, str]) -> tuple[str, str, str]:
    return (
        record["tcr_cdr3_beta"],
        record["epitope"],
        record["hla"],
    )


def _stable_hash(seed: int, *parts: str) -> str:
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_positive_tier(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input has no header: {path}")
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        records = list(reader)

    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = _pair_key(record)
        if key in seen:
            raise ValueError(f"Duplicate positive pair in source tier: {key}")
        seen.add(key)
        if record["label"] != "1":
            raise ValueError(f"Non-positive label in source tier: {record['label']}")
        expected_pmhc = f"{record['epitope']}|{record['hla']}"
        if record["pmhc_id"] != expected_pmhc:
            raise ValueError(
                f"Invalid pMHC id for {key}: {record['pmhc_id']} != {expected_pmhc}"
            )
    return list(reader.fieldnames), records


def _validate_ratios(ratios: dict[str, Any]) -> dict[str, Decimal]:
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"ratios must define exactly: {', '.join(SPLIT_NAMES)}")
    decimals = {name: Decimal(str(ratios[name])) for name in SPLIT_NAMES}
    if any(value <= 0 for value in decimals.values()):
        raise ValueError("all split ratios must be positive")
    if sum(decimals.values()) != Decimal("1"):
        raise ValueError("split ratios must sum to 1")
    return decimals


def _target_counts(total: int, ratios: dict[str, Decimal]) -> dict[str, int]:
    raw = {name: ratios[name] * total for name in SPLIT_NAMES}
    counts = {name: int(raw[name]) for name in SPLIT_NAMES}
    remainder = total - sum(counts.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(raw[name] - counts[name]), SPLIT_NAMES.index(name)),
    )
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _sort_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        records,
        key=lambda record: (
            record["epitope"],
            record["hla"],
            record["tcr_cdr3_beta"],
        ),
    )


def random_positive_split(
    records: list[dict[str, str]], ratios: dict[str, Decimal], seed: int
) -> dict[str, list[dict[str, str]]]:
    ordered = sorted(
        records,
        key=lambda record: (
            _stable_hash(seed, *_pair_key(record)),
            _pair_key(record),
        ),
    )
    targets = _target_counts(len(ordered), ratios)
    output: dict[str, list[dict[str, str]]] = {}
    start = 0
    for name in SPLIT_NAMES:
        end = start + targets[name]
        output[name] = _sort_records(ordered[start:end])
        start = end
    return output


def unseen_epitope_positive_split(
    records: list[dict[str, str]], ratios: dict[str, Decimal], seed: int
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record["epitope"]].append(record)
    if len(grouped) < len(SPLIT_NAMES):
        raise ValueError("unseen_epitope split requires at least three peptide groups")

    targets = _target_counts(len(records), ratios)
    counts = {name: 0 for name in SPLIT_NAMES}
    assignments: dict[str, list[dict[str, str]]] = {name: [] for name in SPLIT_NAMES}
    groups = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), _stable_hash(seed, item[0]), item[0]),
    )

    for peptide, group_records in groups:
        group_size = len(group_records)
        candidates: list[tuple[int, int, str, str]] = []
        for name in SPLIT_NAMES:
            proposed = dict(counts)
            proposed[name] += group_size
            squared_error = sum(
                (proposed[split] - targets[split]) ** 2 for split in SPLIT_NAMES
            )
            overshoot = max(proposed[name] - targets[name], 0)
            candidates.append(
                (
                    squared_error,
                    overshoot,
                    _stable_hash(seed, peptide, name),
                    name,
                )
            )
        selected = min(candidates)[-1]
        assignments[selected].extend(group_records)
        counts[selected] += group_size

    return {name: _sort_records(assignments[name]) for name in SPLIT_NAMES}


def _write_csv(path: Path, fieldnames: list[str], records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _partition_invariants(
    source_records: list[dict[str, str]],
    partitions: dict[str, list[dict[str, str]]],
    *,
    require_peptide_disjoint: bool,
) -> dict[str, Any]:
    source_keys = {_pair_key(record) for record in source_records}
    combined_keys: list[tuple[str, str, str]] = []
    split_keys: dict[str, set[tuple[str, str, str]]] = {}
    split_peptides: dict[str, set[str]] = {}
    for name in SPLIT_NAMES:
        keys = [_pair_key(record) for record in partitions[name]]
        split_keys[name] = set(keys)
        split_peptides[name] = {record["epitope"] for record in partitions[name]}
        combined_keys.extend(keys)

    overlap_pairs = sum(
        len(split_keys[left] & split_keys[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    )
    peptide_overlap = sum(
        len(split_peptides[left] & split_peptides[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    )
    combined_set = set(combined_keys)
    invariants = {
        "source_positive_pairs": len(source_records),
        "combined_positive_pairs": len(combined_keys),
        "combined_duplicate_pair_keys": len(combined_keys) - len(combined_set),
        "cross_split_pair_overlaps": overlap_pairs,
        "missing_source_pair_keys": len(source_keys - combined_set),
        "extra_pair_keys": len(combined_set - source_keys),
        "peptide_overlap_across_splits": peptide_overlap,
        "peptide_disjoint_required": require_peptide_disjoint,
    }
    if invariants["combined_duplicate_pair_keys"] != 0 or overlap_pairs != 0:
        raise ValueError("split partition contains duplicate or overlapping pair keys")
    if invariants["missing_source_pair_keys"] != 0 or invariants["extra_pair_keys"] != 0:
        raise ValueError("split partition does not equal its source tier")
    if require_peptide_disjoint and peptide_overlap != 0:
        raise ValueError("unseen_epitope peptide sets are not disjoint")
    return invariants


def _split_stats(
    path: Path, records: list[dict[str, str]], source_total: int
) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": file_sha256(path),
        "positive_pairs": len(records),
        "fraction_of_tier": len(records) / source_total if source_total else 0.0,
        "unique_tcrs": len({record["tcr_cdr3_beta"] for record in records}),
        "unique_epitopes": len({record["epitope"] for record in records}),
        "unique_hlas": len({record["hla"] for record in records}),
        "unique_pmhcs": len({record["pmhc_id"] for record in records}),
    }


def _validate_config(config: dict[str, Any]) -> dict[str, Decimal]:
    if config.get("schema_version") != 1:
        raise ValueError("positive split config schema_version must be 1")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    if set(config.get("tiers", {})) != {"development", "strict"}:
        raise ValueError("tiers must define development and strict")
    strategies = config.get("strategies", {})
    if set(strategies) != set(STRATEGY_NAMES):
        raise ValueError("strategies must define random and unseen_epitope")
    if strategies["random"].get("grouping") != "pair":
        raise ValueError("random strategy must group by pair")
    if strategies["unseen_epitope"].get("grouping") != "epitope":
        raise ValueError("unseen_epitope strategy must group by epitope")
    return _validate_ratios(config.get("ratios", {}))


def render_positive_split_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Positive data splits",
        "",
        "## Definition",
        "",
        f"- Seed: `{manifest['seed']}`",
        "- Ratios: train 70%, validation 15%, test 15%",
        "- Random: deterministic pair-level allocation",
        "- Unseen epitope: peptide-sequence grouped allocation; peptide sets are disjoint",
        "- Positive samples only",
        "",
        "## Statistics",
        "",
        "| Tier | Strategy | Split | Positive pairs | Fraction | Peptides | pMHCs |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for tier_name in ("development", "strict"):
        for strategy_name in STRATEGY_NAMES:
            strategy = manifest["tiers"][tier_name][strategy_name]
            for split_name in SPLIT_NAMES:
                split = strategy["splits"][split_name]
                lines.append(
                    f"| {tier_name.title()} | {strategy_name} | {split_name} | "
                    f"{split['positive_pairs']:,} | {split['fraction_of_tier']:.4f} | "
                    f"{split['unique_epitopes']:,} | {split['unique_pmhcs']:,} |"
                )
    lines.extend(
        [
            "",
            "## Invariants",
            "",
            "For every tier and strategy:",
            "",
            "- train + validation + test equals the source tier",
            "- duplicate, missing, extra, and cross-split pair-key counts are zero",
            "- unseen_epitope peptide overlap across splits is zero",
            "- output rows are sorted by epitope, HLA, and CDR3beta",
            "",
            "This report is generated by `scripts/build_positive_splits.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_positive_splits(
    config_path: Path,
    config: dict[str, Any],
    tier_paths: dict[str, Path],
    output_dir: Path,
    manifest_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    ratios = _validate_config(config)
    seed = int(config["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "ratios": {name: float(ratios[name]) for name in SPLIT_NAMES},
        "config": {"file": config_path.name, "sha256": file_sha256(config_path)},
        "tiers": {},
        "artifacts": {
            "manifest_json": manifest_path.name,
            "report_md": report_path.name,
        },
    }

    for tier_name in ("development", "strict"):
        source_path = tier_paths[tier_name]
        fieldnames, records = _load_positive_tier(source_path)
        tier_manifest: dict[str, Any] = {
            "source": {
                "file": source_path.name,
                "sha256": file_sha256(source_path),
                "positive_pairs": len(records),
            }
        }
        strategy_partitions = {
            "random": random_positive_split(records, ratios, seed),
            "unseen_epitope": unseen_epitope_positive_split(records, ratios, seed),
        }
        for strategy_name in STRATEGY_NAMES:
            partitions = strategy_partitions[strategy_name]
            split_manifest: dict[str, Any] = {}
            for split_name in SPLIT_NAMES:
                path = output_dir / f"{tier_name}_{strategy_name}_{split_name}.csv"
                _write_csv(path, fieldnames, partitions[split_name])
                split_manifest[split_name] = _split_stats(
                    path, partitions[split_name], len(records)
                )
            tier_manifest[strategy_name] = {
                "splits": split_manifest,
                "invariants": _partition_invariants(
                    records,
                    partitions,
                    require_peptide_disjoint=strategy_name == "unseen_epitope",
                ),
            }
        manifest["tiers"][tier_name] = tier_manifest

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_positive_split_report(manifest), encoding="utf-8")
    return manifest
