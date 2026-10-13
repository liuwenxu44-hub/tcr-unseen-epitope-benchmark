from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from rapidfuzz.distance import Levenshtein

from .data_tiers import PAIR_KEY_COLUMNS
from .vdjdb import file_sha256


PARTITIONS = ("train", "validation", "test")
REQUIRED_COLUMNS = {*PAIR_KEY_COLUMNS, "epitope", "pmhc_id", "label"}


@dataclass(frozen=True)
class SimilarityGraph:
    components: tuple[tuple[str, ...], ...]
    edge_count: int
    pairwise_comparison_count: int


def normalized_levenshtein_similarity(left: str, right: str) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0
    return 1.0 - Levenshtein.distance(left, right) / denominator


def _stable_hash(seed: int, *parts: str) -> str:
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pair_key(record: Mapping[str, str]) -> tuple[str, str, str]:
    return tuple(record[column] for column in PAIR_KEY_COLUMNS)  # type: ignore[return-value]


def _sort_records(records: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return sorted(
        (dict(record) for record in records),
        key=lambda record: (
            record["epitope"],
            record["hla"],
            record["tcr_cdr3_beta"],
        ),
    )


def _validate_ratios(ratios: Mapping[str, Any]) -> dict[str, Decimal]:
    if set(ratios) != set(PARTITIONS):
        raise ValueError("ratios must define train, validation, and test")
    values = {name: Decimal(str(ratios[name])) for name in PARTITIONS}
    if any(value <= 0 for value in values.values()) or sum(values.values()) != Decimal(
        "1"
    ):
        raise ValueError("split ratios must be positive and sum to 1")
    return values


def _target_counts(total: int, ratios: Mapping[str, Decimal]) -> dict[str, int]:
    raw = {name: ratios[name] * total for name in PARTITIONS}
    counts = {name: int(raw[name]) for name in PARTITIONS}
    remainder = total - sum(counts.values())
    order = sorted(
        PARTITIONS,
        key=lambda name: (-(raw[name] - counts[name]), PARTITIONS.index(name)),
    )
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def build_similarity_graph(
    peptides: Sequence[str], threshold: float
) -> SimilarityGraph:
    nodes = sorted(set(peptides))
    if len(nodes) < len(PARTITIONS):
        raise ValueError("distance-aware split requires at least three unique peptides")
    parent = {peptide: peptide for peptide in nodes}

    def find(peptide: str) -> str:
        while parent[peptide] != peptide:
            parent[peptide] = parent[parent[peptide]]
            peptide = parent[peptide]
        return peptide

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            lower, upper = sorted((left_root, right_root))
            parent[upper] = lower

    edge_count = 0
    comparison_count = 0
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            comparison_count += 1
            if normalized_levenshtein_similarity(left, right) > threshold:
                union(left, right)
                edge_count += 1

    grouped: dict[str, list[str]] = defaultdict(list)
    for peptide in nodes:
        grouped[find(peptide)].append(peptide)
    components = tuple(sorted(tuple(sorted(values)) for values in grouped.values()))
    return SimilarityGraph(components, edge_count, comparison_count)


def assign_components(
    records: Sequence[Mapping[str, str]],
    graph: SimilarityGraph,
    ratios: Mapping[str, Decimal],
    seed: int,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[tuple[str, ...]]]]:
    targets = _target_counts(len(records), ratios)
    return _assign_components_to_targets(
        records, graph, PARTITIONS, targets, seed
    )


def _assign_components_to_targets(
    records: Sequence[Mapping[str, str]],
    graph: SimilarityGraph,
    names: Sequence[str],
    targets: Mapping[str, int],
    seed: int,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[tuple[str, ...]]]]:
    names = tuple(names)
    if not names or len(set(names)) != len(names):
        raise ValueError("allocation names must be non-empty and unique")
    if set(targets) != set(names) or sum(targets.values()) != len(records):
        raise ValueError("allocation targets must match names and source record count")
    records_by_peptide: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for record in records:
        records_by_peptide[record["epitope"]].append(record)
    component_groups = []
    for component in graph.components:
        component_records = [
            record
            for peptide in component
            for record in records_by_peptide[peptide]
        ]
        component_groups.append((component, component_records))
    component_groups.sort(
        key=lambda item: (
            -len(item[1]),
            _stable_hash(seed, "|".join(item[0])),
            item[0],
        )
    )
    if len(component_groups) < len(names):
        raise ValueError(
            "distance-aware graph has fewer connected components than allocations"
        )

    counts = {name: 0 for name in names}
    assigned_records: dict[str, list[dict[str, str]]] = {
        name: [] for name in names
    }
    assigned_components: dict[str, list[tuple[str, ...]]] = {
        name: [] for name in names
    }
    for index, (component, component_records) in enumerate(component_groups):
        empty = [name for name in names if not assigned_components[name]]
        remaining_after = len(component_groups) - index - 1
        allowed = empty if len(empty) > remaining_after else list(names)
        candidates = []
        for name in allowed:
            proposed = dict(counts)
            proposed[name] += len(component_records)
            squared_error = sum(
                (proposed[partition] - targets[partition]) ** 2
                for partition in names
            )
            overshoot = max(proposed[name] - targets[name], 0)
            candidates.append(
                (
                    squared_error,
                    overshoot,
                    _stable_hash(seed, "|".join(component), name),
                    name,
                )
            )
        selected = min(candidates)[-1]
        assigned_records[selected].extend(dict(record) for record in component_records)
        assigned_components[selected].append(component)
        counts[selected] += len(component_records)

    if any(not assigned_records[name] for name in names):
        raise ValueError("distance-aware allocation produced an empty group")
    return (
        {name: _sort_records(assigned_records[name]) for name in names},
        assigned_components,
    )


def assign_components_to_buckets(
    records: Sequence[Mapping[str, str]],
    graph: SimilarityGraph,
    bucket_count: int,
    seed: int,
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[tuple[str, ...]]],
    dict[str, int],
]:
    if bucket_count < 2:
        raise ValueError("distance-aware allocation requires at least two buckets")
    names = tuple(f"bucket_{index}" for index in range(bucket_count))
    base, remainder = divmod(len(records), bucket_count)
    targets = {
        name: base + (index < remainder) for index, name in enumerate(names)
    }
    assigned_records, assigned_components = _assign_components_to_targets(
        records, graph, names, targets, seed
    )
    return assigned_records, assigned_components, targets


def _csv_bytes(fieldnames: Sequence[str], records: Sequence[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode("utf-8")


def canonical_partition_sha256(
    fieldnames: Sequence[str], partitions: Mapping[str, Sequence[Mapping[str, str]]]
) -> dict[str, str]:
    return {
        name: hashlib.sha256(_csv_bytes(fieldnames, partitions[name])).hexdigest()
        for name in PARTITIONS
    }


def _load_source(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("source positive tier has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"source positive tier is missing columns: {sorted(missing)}")
        records = list(reader)
        fieldnames = list(reader.fieldnames)
    keys = [_pair_key(record) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("source positive tier contains duplicate interaction keys")
    for record in records:
        if record["label"] != "1":
            raise ValueError("source positive tier contains a non-positive label")
        if record["pmhc_id"] != f"{record['epitope']}|{record['hla']}":
            raise ValueError("source positive tier contains an invalid pMHC ID")
    return fieldnames, records


def _read_partition(path: Path, expected_fieldnames: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(expected_fieldnames):
            raise ValueError(f"partition CSV header mismatch: {path}")
        return list(reader)


def cross_partition_maximum_similarity(
    partitions: Mapping[str, Sequence[Mapping[str, str]]]
) -> dict[str, Any]:
    peptide_sets = {
        name: sorted({record["epitope"] for record in partitions[name]})
        for name in PARTITIONS
    }
    pairwise = {}
    for index, left_name in enumerate(PARTITIONS):
        for right_name in PARTITIONS[index + 1 :]:
            candidates = [
                (normalized_levenshtein_similarity(left, right), left, right)
                for left in peptide_sets[left_name]
                for right in peptide_sets[right_name]
            ]
            similarity, left, right = min(
                candidates, key=lambda item: (-item[0], item[1], item[2])
            )
            pairwise[f"{left_name}_vs_{right_name}"] = {
                "maximum_similarity": similarity,
                "left_peptide": left,
                "right_peptide": right,
            }
    maximum = max(item["maximum_similarity"] for item in pairwise.values())
    return {"maximum_similarity": maximum, "pairwise": pairwise}


def _record_counter(
    records: Sequence[Mapping[str, str]], fieldnames: Sequence[str]
) -> Counter[tuple[str, ...]]:
    return Counter(tuple(record[field] for field in fieldnames) for record in records)


def validate_partitions(
    source_records: Sequence[Mapping[str, str]],
    partitions: Mapping[str, Sequence[Mapping[str, str]]],
    assigned_components: Mapping[str, Sequence[tuple[str, ...]]],
    fieldnames: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    combined = [
        record for name in PARTITIONS for record in partitions[name]
    ]
    source_counter = _record_counter(source_records, fieldnames)
    combined_counter = _record_counter(combined, fieldnames)
    missing_records = sum((source_counter - combined_counter).values())
    extra_records = sum((combined_counter - source_counter).values())
    pair_keys = [_pair_key(record) for record in combined]
    peptide_sets = {
        name: {record["epitope"] for record in partitions[name]}
        for name in PARTITIONS
    }
    peptide_overlap = sum(
        len(peptide_sets[left] & peptide_sets[right])
        for index, left in enumerate(PARTITIONS)
        for right in PARTITIONS[index + 1 :]
    )
    all_components = [
        component
        for partition in PARTITIONS
        for component in assigned_components[partition]
    ]
    split_components = sum(
        len(
            {
                partition
                for partition in PARTITIONS
                if set(component) & peptide_sets[partition]
            }
        )
        > 1
        for component in all_components
    )
    cross_similarity = cross_partition_maximum_similarity(partitions)
    invariants = {
        "source_positive_pairs": len(source_records),
        "combined_positive_pairs": len(combined),
        "missing_full_records": missing_records,
        "extra_or_modified_full_records": extra_records,
        "combined_duplicate_interaction_keys": len(pair_keys) - len(set(pair_keys)),
        "peptide_overlap_across_partitions": peptide_overlap,
        "split_component_count": split_components,
        "nonempty_partition_count": sum(bool(partitions[name]) for name in PARTITIONS),
        "cross_partition_maximum_similarity": cross_similarity["maximum_similarity"],
        "cross_partition_threshold_satisfied": (
            cross_similarity["maximum_similarity"] <= threshold + 1e-15
        ),
    }
    if missing_records or extra_records:
        raise ValueError("partition records do not exactly equal the source tier")
    if invariants["combined_duplicate_interaction_keys"]:
        raise ValueError("partition records contain duplicate interaction keys")
    if peptide_overlap or split_components:
        raise ValueError("peptides or connected components cross partitions")
    if invariants["nonempty_partition_count"] != len(PARTITIONS):
        raise ValueError("distance-aware split contains an empty partition")
    if not invariants["cross_partition_threshold_satisfied"]:
        raise ValueError("cross-partition peptide similarity exceeds threshold")
    return {"invariants": invariants, "cross_similarity": cross_similarity}


def _partition_stats(
    path: Path,
    records: Sequence[Mapping[str, str]],
    components: Sequence[tuple[str, ...]],
    source_total: int,
) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": file_sha256(path),
        "positive_pairs": len(records),
        "fraction_of_tier": len(records) / source_total,
        "unique_epitopes": len({record["epitope"] for record in records}),
        "unique_pmhcs": len({record["pmhc_id"] for record in records}),
        "unique_hlas": len({record["hla"] for record in records}),
        "unique_tcrs": len({record["tcr_cdr3_beta"] for record in records}),
        "component_count": len(components),
    }


def render_report(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Development distance-aware unseen positive split",
        "",
        "## Definition",
        "",
        f"- Seed: `{manifest['seed']}`",
        f"- Normalized Levenshtein similarity: `{manifest['similarity']['definition']}`",
        f"- Threshold: `{manifest['similarity']['threshold']}`",
        "- Graph edge: peptide similarity strictly greater than the threshold",
        "- Connected components are indivisible allocation units",
        "- Positive samples only",
        "",
        "## Partition statistics",
        "",
        "| Partition | Records | Fraction | Peptides | pMHCs | HLAs | Components |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in PARTITIONS:
        values = manifest["partitions"][name]
        lines.append(
            f"| {name} | {values['positive_pairs']:,} | "
            f"{values['fraction_of_tier']:.6f} | {values['unique_epitopes']} | "
            f"{values['unique_pmhcs']} | {values['unique_hlas']} | "
            f"{values['component_count']} |"
        )
    graph = manifest["graph"]
    lines.extend(
        [
            "",
            "## Similarity graph",
            "",
            f"- Unique peptides: {graph['node_count']}",
            f"- Similarity edges: {graph['edge_count']}",
            f"- Connected components: {graph['component_count']}",
            f"- Largest component: {graph['largest_component_peptide_count']} peptides, "
            f"{graph['largest_component_record_count']:,} records",
            f"- Cross-partition maximum similarity: "
            f"{manifest['cross_partition_similarity']['maximum_similarity']:.6f}",
            "",
            "## Invariants",
            "",
            "- Every source record appears exactly once without modification.",
            "- Peptide sets are pairwise disjoint.",
            "- Connected components do not cross partitions.",
            "- Every partition is nonempty.",
            "- Cross-partition maximum similarity is at most 0.5.",
            "",
            "This report is generated by `scripts/build_distance_aware_split.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_distance_aware_split(
    config_path: Path, project_dir: Path, *, force: bool = False
) -> dict[str, Any]:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("distance-aware split config schema_version must be 1")
    if config["source"].get("tier") != "development":
        raise ValueError("distance-aware split source must be Development")
    seed = int(config["seed"])
    threshold = float(config["similarity"]["threshold"])
    if threshold != 0.5 or config["similarity"].get("edge_rule") != "similarity > threshold":
        raise ValueError("distance-aware split requires the fixed strict 0.5 threshold")
    ratios = _validate_ratios(config["ratios"])

    source_path = project_dir / config["source"]["csv"]
    if file_sha256(source_path) != config["source"]["sha256"]:
        raise ValueError("Development source SHA-256 mismatch")
    fieldnames, records = _load_source(source_path)
    graph = build_similarity_graph(
        [record["epitope"] for record in records], threshold
    )
    partitions, assigned_components = assign_components(
        records, graph, ratios, seed
    )
    validation = validate_partitions(
        records,
        partitions,
        assigned_components,
        fieldnames,
        threshold,
    )

    output_dir = project_dir / config["outputs"]["directory"]
    csv_paths = {
        name: output_dir / config["outputs"][f"{name}_csv"] for name in PARTITIONS
    }
    manifest_path = output_dir / config["outputs"]["manifest_json"]
    report_path = project_dir / config["outputs"]["report_md"]
    output_paths = [*csv_paths.values(), manifest_path, report_path]
    if not force and any(path.exists() for path in output_paths):
        raise FileExistsError("refusing to overwrite distance-aware split outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, path in csv_paths.items():
        path.write_bytes(_csv_bytes(fieldnames, partitions[name]))
    written_partitions = {
        name: _read_partition(path, fieldnames) for name, path in csv_paths.items()
    }
    validation = validate_partitions(
        records,
        written_partitions,
        assigned_components,
        fieldnames,
        threshold,
    )
    partitions = written_partitions

    peptide_record_counts = Counter(record["epitope"] for record in records)
    component_record_counts = [
        sum(peptide_record_counts[peptide] for peptide in component)
        for component in graph.components
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "tier": "development",
        "ratios": {name: float(ratios[name]) for name in PARTITIONS},
        "similarity": {
            "definition": config["similarity"]["definition"],
            "threshold": threshold,
            "edge_rule": config["similarity"]["edge_rule"],
        },
        "source": {
            "path": config["source"]["csv"],
            "sha256": file_sha256(source_path),
            "positive_pairs": len(records),
            "unique_epitopes": len(peptide_record_counts),
            "unique_pmhcs": len({record["pmhc_id"] for record in records}),
            "unique_hlas": len({record["hla"] for record in records}),
        },
        "config": {
            "path": config_path.relative_to(project_dir).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "graph": {
            "node_count": len(peptide_record_counts),
            "edge_count": graph.edge_count,
            "pairwise_comparison_count": graph.pairwise_comparison_count,
            "component_count": len(graph.components),
            "largest_component_peptide_count": max(map(len, graph.components)),
            "largest_component_record_count": max(component_record_counts),
        },
        "partitions": {
            name: _partition_stats(
                csv_paths[name],
                partitions[name],
                assigned_components[name],
                len(records),
            )
            for name in PARTITIONS
        },
        "cross_partition_similarity": validation["cross_similarity"],
        "invariants": validation["invariants"],
        "partition_csv_sha256": canonical_partition_sha256(
            fieldnames, partitions
        ),
        "artifacts": {
            "manifest_json": manifest_path.relative_to(project_dir).as_posix(),
            "report_md": report_path.relative_to(project_dir).as_posix(),
        },
        "negative_samples_generated": False,
        "models_trained_or_evaluated": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(manifest), encoding="utf-8")
    return manifest
