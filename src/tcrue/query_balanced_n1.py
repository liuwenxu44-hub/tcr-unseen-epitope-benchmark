from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .negative_sampling import (
    NEGATIVE_COLUMNS,
    _read_positive_csv,
    load_positive_registry,
)
from .vdjdb import file_sha256


PARTITIONS = ("train", "validation", "test")
FOLD_IDS = tuple(f"fold_{index}" for index in range(1, 6))
QUERY_KEY_COLUMNS = ("epitope", "hla")
INTERACTION_KEY_COLUMNS = ("tcr_cdr3_beta", "epitope", "hla")
SEED_MODULUS = 2_147_483_647

PairKey = tuple[str, str, str]
QueryKey = tuple[str, str]


def _normalize(value: str) -> str:
    return value.strip().upper()


def _query_key(row: Mapping[str, Any]) -> QueryKey:
    return _normalize(str(row["epitope"])), _normalize(str(row["hla"]))


def _pair_key(row: Mapping[str, Any]) -> PairKey:
    return (
        _normalize(str(row["tcr_cdr3_beta"])),
        _normalize(str(row["epitope"])),
        _normalize(str(row["hla"])),
    )


def _query_id(query: QueryKey) -> str:
    return f"{query[0]}|{query[1]}"


def derive_query_seed(
    base_seed: int,
    fold_id: str,
    partition: str,
    epitope: str,
    hla: str,
) -> int:
    if fold_id not in FOLD_IDS:
        raise ValueError(f"unknown fold ID: {fold_id}")
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition: {partition}")
    payload = "\0".join(
        (
            str(base_seed),
            fold_id,
            partition,
            _normalize(epitope),
            _normalize(hla),
            "query_balanced_n1_cap100",
        )
    ).encode("utf-8")
    derived = int(hashlib.sha256(payload).hexdigest()[:16], 16) % SEED_MODULUS
    return derived or 1


def _stable_hash(seed: int, namespace: str, *parts: str) -> str:
    payload = "\0".join((str(seed), namespace, *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _negative_id(fold_id: str, partition: str, key: PairKey) -> str:
    payload = "\0".join(
        ("query_balanced_n1_cap100", fold_id, partition, *key)
    ).encode("utf-8")
    return f"putative-negative:{hashlib.sha256(payload).hexdigest()[:20]}"


def _registry_by_query(positive_registry: set[PairKey]) -> dict[QueryKey, set[str]]:
    grouped: dict[QueryKey, set[str]] = defaultdict(set)
    for tcr, epitope, hla in positive_registry:
        grouped[(epitope, hla)].add(tcr)
    return dict(grouped)


def generate_query_balanced_partition(
    positives: Sequence[Mapping[str, str]],
    positive_registry: set[PairKey],
    *,
    base_seed: int,
    fold_id: str,
    partition: str,
    cap: int = 100,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, Any]]]:
    if cap != 100:
        raise ValueError("query-balanced N1 construction requires cap=100")
    normalized_positives = []
    positive_keys = []
    for source in positives:
        row = dict(source)
        for column in INTERACTION_KEY_COLUMNS:
            row[column] = _normalize(str(row[column]))
        normalized_positives.append(row)
        positive_keys.append(_pair_key(row))
    if len(positive_keys) != len(set(positive_keys)):
        raise ValueError(f"duplicate positive interactions in {fold_id}/{partition}")

    positives_by_query: dict[QueryKey, list[dict[str, str]]] = defaultdict(list)
    sources_by_tcr: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in normalized_positives:
        positives_by_query[_query_key(row)].append(row)
        sources_by_tcr[row["tcr_cdr3_beta"]].append(row)
    partition_tcrs = set(sources_by_tcr)
    known_by_query = _registry_by_query(positive_registry)

    selected_positives: list[dict[str, str]] = []
    selected_negatives: list[dict[str, Any]] = []
    query_statistics = []
    for query in sorted(positives_by_query):
        epitope, hla = query
        seed = derive_query_seed(base_seed, fold_id, partition, epitope, hla)
        query_positives = positives_by_query[query]
        legal_tcrs = partition_tcrs - known_by_query.get(query, set())
        retained = min(len(query_positives), len(legal_tcrs), cap)
        if retained == 0:
            raise ValueError(
                f"query has m_q=0 in {fold_id}/{partition}: {_query_id(query)}"
            )

        ordered_positives = sorted(
            query_positives,
            key=lambda row: (
                _stable_hash(seed, "positive", row["record_id"]),
                row["tcr_cdr3_beta"],
                row["record_id"],
            ),
        )
        chosen_positives = ordered_positives[:retained]
        selected_positives.extend(chosen_positives)

        ordered_tcrs = sorted(
            legal_tcrs,
            key=lambda tcr: (_stable_hash(seed, "negative", tcr), tcr),
        )
        chosen_tcrs = ordered_tcrs[:retained]
        if len(chosen_tcrs) != len(set(chosen_tcrs)):
            raise ValueError(f"duplicate negative TCR within query: {_query_id(query)}")
        for tcr in chosen_tcrs:
            source = min(
                sources_by_tcr[tcr],
                key=lambda row: (
                    _stable_hash(seed, "source", row["record_id"]),
                    row["record_id"],
                ),
            )
            key = (tcr, epitope, hla)
            if key in positive_registry:
                raise ValueError(f"selected known positive as N1: {key}")
            if _query_key(source) == query:
                raise ValueError(f"N1 source pMHC was not changed: {key}")
            selected_negatives.append(
                {
                    "negative_id": _negative_id(fold_id, partition, key),
                    "tcr_cdr3_beta": tcr,
                    "epitope": epitope,
                    "hla": hla,
                    "pmhc_id": _query_id(query),
                    "label": 0,
                    "negative_strategy": "N1",
                    "tier": "development",
                    "positive_split_strategy": (
                        "distance_aware_5fold_rebalanced_query_balanced_cap100"
                    ),
                    "partition": partition,
                    "source_positive_record_id": source["record_id"],
                    "source_positive_epitope": source["epitope"],
                    "source_positive_hla": source["hla"],
                    "source_positive_pmhc_id": source["pmhc_id"],
                    "sampling_seed": seed,
                }
            )
        query_statistics.append(
            {
                "query_id": _query_id(query),
                "epitope": epitope,
                "hla": hla,
                "derived_seed": seed,
                "original_positive_count": len(query_positives),
                "legal_negative_candidate_count": len(legal_tcrs),
                "retained_positive_count": retained,
                "retained_negative_count": retained,
            }
        )

    selected_positives.sort(
        key=lambda row: (
            row["epitope"],
            row["hla"],
            row["tcr_cdr3_beta"],
            row["record_id"],
        )
    )
    selected_negatives.sort(
        key=lambda row: (
            row["epitope"],
            row["hla"],
            row["tcr_cdr3_beta"],
            row["negative_id"],
        )
    )
    audit_query_balanced_partition(
        normalized_positives,
        selected_positives,
        selected_negatives,
        positive_registry,
        cap=cap,
    )
    return selected_positives, selected_negatives, query_statistics


def audit_query_balanced_partition(
    source_positives: Sequence[Mapping[str, Any]],
    selected_positives: Sequence[Mapping[str, Any]],
    selected_negatives: Sequence[Mapping[str, Any]],
    positive_registry: set[PairKey],
    *,
    cap: int,
) -> dict[str, Any]:
    source_queries = {_query_key(row) for row in source_positives}
    positive_counts: dict[QueryKey, int] = defaultdict(int)
    negative_counts: dict[QueryKey, int] = defaultdict(int)
    positive_keys = [_pair_key(row) for row in selected_positives]
    negative_keys = [_pair_key(row) for row in selected_negatives]
    for row in selected_positives:
        positive_counts[_query_key(row)] += 1
    for row in selected_negatives:
        negative_counts[_query_key(row)] += 1
    selected_queries = set(positive_counts) | set(negative_counts)
    if selected_queries != source_queries:
        raise ValueError("query-balanced output changed the query set")
    if len(positive_keys) != len(set(positive_keys)):
        raise ValueError("query-balanced positives contain duplicate interactions")
    if len(negative_keys) != len(set(negative_keys)):
        raise ValueError("query-balanced negatives contain duplicate interactions")
    if set(positive_keys) & set(negative_keys):
        raise ValueError("query-balanced positive and negative interactions overlap")
    if set(negative_keys) & positive_registry:
        raise ValueError("query-balanced negatives intersect positive registry")
    for query in source_queries:
        if (
            positive_counts[query] == 0
            or positive_counts[query] != negative_counts[query]
            or positive_counts[query] > cap
        ):
            raise ValueError(f"query balance failed: {_query_id(query)}")
    source_tcrs = {_pair_key(row)[0] for row in source_positives}
    negative_tcrs = {_pair_key(row)[0] for row in selected_negatives}
    if not negative_tcrs <= source_tcrs:
        raise ValueError("query-balanced negative uses TCR outside partition")
    retained = [positive_counts[query] for query in sorted(source_queries)]
    return {
        "query_count": len(source_queries),
        "positive_count": len(selected_positives),
        "negative_count": len(selected_negatives),
        "retained_per_class_minimum": min(retained),
        "retained_per_class_median": statistics.median(retained),
        "retained_per_class_maximum": max(retained),
        "total_records_per_query_minimum": 2 * min(retained),
        "total_records_per_query_median": 2 * statistics.median(retained),
        "total_records_per_query_maximum": 2 * max(retained),
        "maximum_query_fraction_of_all_records": max(retained) / sum(retained),
        "known_positive_registry_intersections": 0,
        "duplicate_positive_interactions": 0,
        "duplicate_negative_interactions": 0,
        "negative_tcrs_outside_partition": 0,
    }


def _csv_text(
    records: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_query_audit(path: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    indexed = {
        (row["fold_id"], row["partition"], row["epitope"], row["hla"]): row
        for row in rows
    }
    if len(indexed) != len(rows):
        raise ValueError("query-balance audit contains duplicate rows")
    return indexed


def _validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "base_seed": 13,
        "tier": "development",
        "fold_ids": list(FOLD_IDS),
        "partitions": list(PARTITIONS),
        "query_key": list(QUERY_KEY_COLUMNS),
        "interaction_key": list(INTERACTION_KEY_COLUMNS),
        "per_query_cap": 100,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"query-balanced N1 config requires {key}={value!r}")
    if config.get("seed_derivation") != {
        "algorithm": "sha256",
        "namespace": (
            "base_seed|fold_id|partition|epitope|hla|query_balanced_n1_cap100"
        ),
        "modulus": SEED_MODULUS,
    }:
        raise ValueError("query-balanced N1 seed derivation mismatch")


def render_report(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Distance-aware five-fold query-balanced N1 cap100",
        "",
        "## Construction",
        "",
        "- Query key: `(epitope, hla)`.",
        "- Per query: `m_q = min(positive count, legal N1 capacity, 100)`.",
        "- Positives and candidate TCRs are selected by query-derived deterministic seeds.",
        "- Candidate TCRs come only from the current fold partition.",
        "- Complete positive-registry interactions are excluded.",
        "- Existing rebalanced positives and global-balanced N1 files are unchanged.",
        "",
        "## Partition statistics",
        "",
        "| Fold | Partition | Queries | Positives | Negatives | Per-query total records min / median / max | Largest query fraction | Positive SHA-256 | Negative SHA-256 |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for fold_id in FOLD_IDS:
        for partition in PARTITIONS:
            values = manifest["folds"][fold_id]["partitions"][partition]
            audit = values["audit"]
            lines.append(
                f"| {fold_id} | {partition} | {audit['query_count']} | "
                f"{audit['positive_count']:,} | {audit['negative_count']:,} | "
                f"{audit['total_records_per_query_minimum']} / "
                f"{audit['total_records_per_query_median']} / "
                f"{audit['total_records_per_query_maximum']} | "
                f"{audit['maximum_query_fraction_of_all_records']:.6%} | "
                f"`{values['positive_output']['sha256']}` | "
                f"`{values['negative_output']['sha256']}` |"
            )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            "- Every original query is retained with nonzero `m_q`.",
            "- Every query has equal positive and negative counts, each at most 100.",
            "- Negative interactions have zero overlap with the complete positive registry.",
            "- No duplicate interactions or within-fold cross-partition leakage were found.",
            "- Two complete deterministic rebuilds produced identical CSV SHA-256 values.",
            "- No model was trained.",
            "",
            "This report is generated by `scripts/build_query_balanced_n1.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_query_balanced_n1(
    config_path: Path, project_dir: Path, *, force: bool = False
) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_dir = project_dir.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("query-balanced N1 config must be a mapping")
    _validate_config(config)
    inputs = config["inputs"]
    source_manifest_path = project_dir / inputs["positive_manifest"]["path"]
    registry_path = project_dir / inputs["positive_registry"]["path"]
    query_audit_path = project_dir / inputs["query_balance_audit"]["path"]
    frozen_paths = {
        "positive_manifest": source_manifest_path,
        "positive_registry": registry_path,
        "query_balance_audit": query_audit_path,
    }
    for name, path in frozen_paths.items():
        if file_sha256(path) != inputs[name]["sha256"]:
            raise ValueError(f"{name} SHA-256 mismatch")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("status") != "active"
        or source_manifest.get("negative_samples_generated") is not False
        or source_manifest.get("fold_count") != 5
    ):
        raise ValueError("source is not the active positive-only rebalanced split")

    outputs = config["outputs"]
    positive_directory = project_dir / outputs["positive_directory"]
    negative_directory = project_dir / outputs["negative_directory"]
    positive_paths = {
        fold_id: {
            partition: positive_directory
            / outputs["positive_csv_pattern"].format(
                fold=fold_id, partition=partition
            )
            for partition in PARTITIONS
        }
        for fold_id in FOLD_IDS
    }
    negative_paths = {
        fold_id: {
            partition: negative_directory
            / outputs["negative_csv_pattern"].format(
                fold=fold_id, partition=partition
            )
            for partition in PARTITIONS
        }
        for fold_id in FOLD_IDS
    }
    manifest_path = project_dir / outputs["manifest_json"]
    report_path = project_dir / outputs["report_md"]
    all_outputs = [
        path
        for paths in (*positive_paths.values(), *negative_paths.values())
        for path in paths.values()
    ] + [manifest_path, report_path]
    if not force and any(path.exists() for path in all_outputs):
        raise FileExistsError("refusing to overwrite query-balanced N1 outputs")

    source_directory = project_dir / inputs["positive_directory"]
    positive_registry = load_positive_registry(registry_path)
    query_audit = _load_query_audit(query_audit_path)
    base_seed = int(config["base_seed"])
    source_hashes_before = {}
    first_rebuild_hashes = {}
    second_rebuild_hashes = {}
    folds_manifest = {}
    for fold_id in FOLD_IDS:
        fold_manifest = {"partitions": {}, "invariants": {}}
        combined_key_sets: dict[str, set[PairKey]] = {}
        peptide_sets: dict[str, set[str]] = {}
        first_rebuild_hashes[fold_id] = {}
        second_rebuild_hashes[fold_id] = {}
        for partition in PARTITIONS:
            source_entry = source_manifest["folds"][fold_id]["partitions"][partition]
            source_path = source_directory / source_entry["file"]
            source_sha = file_sha256(source_path)
            if source_sha != source_entry["sha256"]:
                raise ValueError(f"source positive SHA-256 mismatch: {fold_id}/{partition}")
            source_hashes_before[(fold_id, partition)] = source_sha
            positives = _read_positive_csv(source_path)
            if len(positives) != int(source_entry["positive_pairs"]):
                raise ValueError(f"source positive count mismatch: {fold_id}/{partition}")
            first_positive, first_negative, first_statistics = (
                generate_query_balanced_partition(
                    positives,
                    positive_registry,
                    base_seed=base_seed,
                    fold_id=fold_id,
                    partition=partition,
                    cap=100,
                )
            )
            second_positive, second_negative, second_statistics = (
                generate_query_balanced_partition(
                    positives,
                    positive_registry,
                    base_seed=base_seed,
                    fold_id=fold_id,
                    partition=partition,
                    cap=100,
                )
            )
            positive_fieldnames = list(positives[0])
            first_positive_text = _csv_text(first_positive, positive_fieldnames)
            first_negative_text = _csv_text(first_negative, NEGATIVE_COLUMNS)
            second_positive_text = _csv_text(second_positive, positive_fieldnames)
            second_negative_text = _csv_text(second_negative, NEGATIVE_COLUMNS)
            first_hashes = {
                "positive": _text_sha256(first_positive_text),
                "negative": _text_sha256(first_negative_text),
            }
            second_hashes = {
                "positive": _text_sha256(second_positive_text),
                "negative": _text_sha256(second_negative_text),
            }
            if (
                first_hashes != second_hashes
                or first_positive != second_positive
                or first_negative != second_negative
                or first_statistics != second_statistics
            ):
                raise ValueError(f"rebuild mismatch: {fold_id}/{partition}")
            first_rebuild_hashes[fold_id][partition] = first_hashes
            second_rebuild_hashes[fold_id][partition] = second_hashes

            for values in first_statistics:
                audit_key = (fold_id, partition, values["epitope"], values["hla"])
                expected = query_audit.get(audit_key)
                if expected is None:
                    raise ValueError(f"query missing from prior audit: {audit_key}")
                if (
                    int(expected["positive_count"])
                    != values["original_positive_count"]
                    or int(expected["legal_negative_candidate_count"])
                    != values["legal_negative_candidate_count"]
                    or int(expected["retained_per_class_cap_100"])
                    != values["retained_positive_count"]
                ):
                    raise ValueError(f"query capacity disagrees with audit: {audit_key}")

            positive_path = positive_paths[fold_id][partition]
            negative_path = negative_paths[fold_id][partition]
            positive_path.parent.mkdir(parents=True, exist_ok=True)
            negative_path.parent.mkdir(parents=True, exist_ok=True)
            positive_path.write_text(first_positive_text, encoding="utf-8")
            negative_path.write_text(first_negative_text, encoding="utf-8")
            if (
                file_sha256(positive_path) != first_hashes["positive"]
                or file_sha256(negative_path) != first_hashes["negative"]
            ):
                raise ValueError(f"written CSV SHA-256 mismatch: {fold_id}/{partition}")
            audit = audit_query_balanced_partition(
                positives,
                first_positive,
                first_negative,
                positive_registry,
                cap=100,
            )
            combined_keys = {
                _pair_key(row) for row in (*first_positive, *first_negative)
            }
            if len(combined_keys) != len(first_positive) + len(first_negative):
                raise ValueError(f"combined duplicate interactions: {fold_id}/{partition}")
            combined_key_sets[partition] = combined_keys
            peptide_sets[partition] = {row["epitope"] for row in first_positive}
            fold_manifest["partitions"][partition] = {
                "source_positive": {
                    "path": source_path.relative_to(project_dir).as_posix(),
                    "sha256": source_sha,
                    "positive_pairs": len(positives),
                },
                "positive_output": {
                    "path": positive_path.relative_to(project_dir).as_posix(),
                    "sha256": file_sha256(positive_path),
                    "rows": len(first_positive),
                },
                "negative_output": {
                    "path": negative_path.relative_to(project_dir).as_posix(),
                    "sha256": file_sha256(negative_path),
                    "rows": len(first_negative),
                },
                "audit": audit,
                "query_statistics": first_statistics,
            }

        interaction_intersections = {}
        peptide_intersections = {}
        for index, left in enumerate(PARTITIONS):
            for right in PARTITIONS[index + 1 :]:
                name = f"{left}_vs_{right}"
                interaction_intersections[name] = len(
                    combined_key_sets[left] & combined_key_sets[right]
                )
                peptide_intersections[name] = len(
                    peptide_sets[left] & peptide_sets[right]
                )
        if any(interaction_intersections.values()) or any(peptide_intersections.values()):
            raise ValueError(f"cross-partition leakage in {fold_id}")
        fold_manifest["invariants"] = {
            "partition_interaction_intersections": interaction_intersections,
            "partition_peptide_intersections": peptide_intersections,
            "all_queries_retained": True,
            "all_queries_balanced": True,
        }
        folds_manifest[fold_id] = fold_manifest

    source_hashes_after = {
        key: file_sha256(
            source_directory
            / source_manifest["folds"][key[0]]["partitions"][key[1]]["file"]
        )
        for key in source_hashes_before
    }
    if source_hashes_after != source_hashes_before:
        raise ValueError("source positive CSVs changed during construction")
    if first_rebuild_hashes != second_rebuild_hashes:
        raise ValueError("complete rebuild SHA-256 mismatch")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "tier": "development",
        "fold_count": 5,
        "base_seed": base_seed,
        "per_query_cap": 100,
        "query_key": list(QUERY_KEY_COLUMNS),
        "interaction_key": list(INTERACTION_KEY_COLUMNS),
        "seed_derivation": config["seed_derivation"],
        "formula": "m_q = min(positive_count, legal_negative_candidate_count, 100)",
        "config": {
            "path": config_path.relative_to(project_dir).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "inputs": {
            name: {
                "path": path.relative_to(project_dir).as_posix(),
                "sha256": inputs[name]["sha256"],
            }
            for name, path in frozen_paths.items()
        },
        "folds": folds_manifest,
        "reproducibility": {
            "complete_rebuild_count": 2,
            "sha256_identical": True,
            "first_rebuild_csv_sha256": first_rebuild_hashes,
            "second_rebuild_csv_sha256": second_rebuild_hashes,
        },
        "invariants": {
            "all_original_queries_retained": True,
            "all_queries_positive_negative_one_to_one": True,
            "maximum_per_query_per_class": 100,
            "known_positive_registry_intersections": 0,
            "duplicate_interactions": 0,
            "within_fold_cross_partition_interactions": 0,
            "within_fold_cross_partition_peptides": 0,
            "source_positive_csvs_modified": False,
            "global_balanced_n1_modified": False,
            "models_trained": False,
        },
        "artifacts": {
            "positive_directory": outputs["positive_directory"],
            "negative_directory": outputs["negative_directory"],
            "manifest_json": outputs["manifest_json"],
            "report_md": outputs["report_md"],
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(manifest), encoding="utf-8")
    return manifest
