from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data_tiers import PAIR_KEY_COLUMNS
from .distance_aware_split import (
    PARTITIONS,
    SimilarityGraph,
    _csv_bytes,
    _load_source,
    _partition_stats,
    _read_partition,
    _record_counter,
    _sort_records,
    assign_components_to_buckets,
    build_similarity_graph,
    validate_partitions,
)
from .vdjdb import file_sha256


FOLD_COUNT = 5
BUCKET_NAMES = tuple(f"bucket_{index}" for index in range(FOLD_COUNT))
FOLD_NAMES = tuple(f"fold_{index + 1}" for index in range(FOLD_COUNT))


def materialize_profile_assignments(
    records: Sequence[Mapping[str, str]],
    profile_assignments: Mapping[str, Sequence[Any]],
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[tuple[str, ...]]],
]:
    if set(profile_assignments) != set(BUCKET_NAMES):
        raise ValueError("profile assignments must define all five buckets")
    records_by_peptide: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for record in records:
        records_by_peptide[record["epitope"]].append(record)
    bucket_components = {
        name: [profile.peptides for profile in profile_assignments[name]]
        for name in BUCKET_NAMES
    }
    buckets = {
        name: _sort_records(
            [
                record
                for profile in profile_assignments[name]
                for peptide in profile.peptides
                for record in records_by_peptide[peptide]
            ]
        )
        for name in BUCKET_NAMES
    }
    return buckets, bucket_components


def build_fold_partitions(
    buckets: Mapping[str, Sequence[Mapping[str, str]]],
    bucket_components: Mapping[str, Sequence[tuple[str, ...]]],
) -> dict[str, dict[str, Any]]:
    if set(buckets) != set(BUCKET_NAMES) or set(bucket_components) != set(
        BUCKET_NAMES
    ):
        raise ValueError("five-fold construction requires bucket_0 through bucket_4")
    folds: dict[str, dict[str, Any]] = {}
    for index, fold_name in enumerate(FOLD_NAMES):
        test_bucket = BUCKET_NAMES[index]
        validation_bucket = BUCKET_NAMES[(index + 1) % FOLD_COUNT]
        train_buckets = tuple(
            name
            for name in BUCKET_NAMES
            if name not in {test_bucket, validation_bucket}
        )
        partitions = {
            "train": _sort_records(
                [record for name in train_buckets for record in buckets[name]]
            ),
            "validation": _sort_records(buckets[validation_bucket]),
            "test": _sort_records(buckets[test_bucket]),
        }
        components = {
            "train": [
                component
                for name in train_buckets
                for component in bucket_components[name]
            ],
            "validation": list(bucket_components[validation_bucket]),
            "test": list(bucket_components[test_bucket]),
        }
        folds[fold_name] = {
            "index": index,
            "test_bucket": test_bucket,
            "validation_bucket": validation_bucket,
            "train_buckets": train_buckets,
            "partitions": partitions,
            "components": components,
        }
    return folds


def canonical_fold_sha256(
    fieldnames: Sequence[str], folds: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, str]]:
    return {
        fold_name: {
            partition: hashlib.sha256(
                _csv_bytes(
                    fieldnames,
                    folds[fold_name]["partitions"][partition],
                )
            ).hexdigest()
            for partition in PARTITIONS
        }
        for fold_name in FOLD_NAMES
    }


def _pair_key(record: Mapping[str, str]) -> tuple[str, str, str]:
    return tuple(record[column] for column in PAIR_KEY_COLUMNS)  # type: ignore[return-value]


def validate_five_folds(
    source_records: Sequence[Mapping[str, str]],
    graph: SimilarityGraph,
    buckets: Mapping[str, Sequence[Mapping[str, str]]],
    bucket_components: Mapping[str, Sequence[tuple[str, ...]]],
    folds: Mapping[str, Mapping[str, Any]],
    fieldnames: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    if set(folds) != set(FOLD_NAMES):
        raise ValueError("five-fold result must contain exactly five folds")
    source_counter = _record_counter(source_records, fieldnames)
    bucket_records = [
        record for name in BUCKET_NAMES for record in buckets[name]
    ]
    bucket_counter = _record_counter(bucket_records, fieldnames)
    if bucket_counter != source_counter:
        raise ValueError("five-fold buckets do not exactly equal the source tier")
    bucket_pair_keys = [_pair_key(record) for record in bucket_records]
    if len(bucket_pair_keys) != len(set(bucket_pair_keys)):
        raise ValueError("five-fold buckets contain duplicate interaction keys")

    bucket_peptides = {
        name: {record["epitope"] for record in buckets[name]}
        for name in BUCKET_NAMES
    }
    for index, left in enumerate(BUCKET_NAMES):
        for right in BUCKET_NAMES[index + 1 :]:
            if bucket_peptides[left] & bucket_peptides[right]:
                raise ValueError("peptides cross five-fold buckets")
    assigned_components = [
        component
        for name in BUCKET_NAMES
        for component in bucket_components[name]
    ]
    if Counter(assigned_components) != Counter(graph.components):
        raise ValueError("connected components are missing, duplicated, or split")

    fold_audits: dict[str, Any] = {}
    test_records: list[Mapping[str, str]] = []
    test_peptide_counts: Counter[str] = Counter()
    test_component_counts: Counter[tuple[str, ...]] = Counter()
    for fold_name in FOLD_NAMES:
        fold = folds[fold_name]
        audit = validate_partitions(
            source_records,
            fold["partitions"],
            fold["components"],
            fieldnames,
            threshold,
        )
        fold_audits[fold_name] = audit
        test_partition = fold["partitions"]["test"]
        test_records.extend(test_partition)
        test_peptide_counts.update(
            {record["epitope"] for record in test_partition}
        )
        test_component_counts.update(fold["components"]["test"])

    if _record_counter(test_records, fieldnames) != source_counter:
        raise ValueError("each source record must be test exactly once")
    source_peptides = {record["epitope"] for record in source_records}
    if set(test_peptide_counts) != source_peptides or any(
        count != 1 for count in test_peptide_counts.values()
    ):
        raise ValueError("each peptide must be test exactly once")
    if Counter(test_component_counts) != Counter(
        {component: 1 for component in graph.components}
    ):
        raise ValueError("each connected component must be test exactly once")

    maximum_similarity = max(
        audit["cross_similarity"]["maximum_similarity"]
        for audit in fold_audits.values()
    )
    return {
        "folds": fold_audits,
        "invariants": {
            "source_positive_pairs": len(source_records),
            "bucket_positive_pairs": len(bucket_records),
            "bucket_duplicate_interaction_keys": len(bucket_pair_keys)
            - len(set(bucket_pair_keys)),
            "fold_count": len(folds),
            "records_present_once_per_fold": True,
            "records_tested_exactly_once": True,
            "peptides_tested_exactly_once": True,
            "components_tested_exactly_once": True,
            "cross_partition_maximum_similarity": maximum_similarity,
            "cross_partition_threshold_satisfied": maximum_similarity
            <= threshold + 1e-15,
        },
    }


def _test_reference_stats(
    train_records: Sequence[Mapping[str, str]],
    test_records: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    train_hlas = {record["hla"] for record in train_records}
    test_queries = {
        (record["epitope"], record["hla"]) for record in test_records
    }
    referenced = sum(hla in train_hlas for _, hla in test_queries)
    return {
        "test_query_count": len(test_queries),
        "test_queries_with_same_hla_train_reference": referenced,
        "same_hla_train_reference_coverage": referenced / len(test_queries),
    }


def _component_sha256(components: Sequence[tuple[str, ...]]) -> str:
    payload = json.dumps(
        sorted(components), ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def render_report(manifest: Mapping[str, Any]) -> str:
    allocation = manifest["bucket_allocation"]
    lines = [
        "# Development distance-aware five-fold positive splits",
        "",
        "## Definition",
        "",
        f"- Seed: `{manifest['seed']}`",
        f"- Fold count: `{manifest['fold_count']}`",
        f"- Similarity threshold: `{manifest['similarity']['threshold']}`",
        f"- Allocation objective: `{allocation['objective']}`",
        "- Edges connect peptide pairs with normalized Levenshtein similarity greater than 0.5.",
        "- Connected components are assigned intact to five deterministic buckets.",
        "- Fold i uses bucket i as test, bucket (i+1) mod 5 as validation, and the other buckets as train.",
        "- Positive samples only.",
        "",
        "## Bucket allocation",
        "",
        "| Bucket | Target queries | Actual queries | Target records | Actual records | Peptides | Components |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if "weights" in allocation:
        lines.insert(
            7,
            f"- Allocation weights: query={allocation['weights']['query_count']}, "
            f"records={allocation['weights']['positive_record_count']}",
        )
    if "supersedes" in manifest:
        lines.insert(
            8 if "weights" in allocation else 7,
            f"- Supersedes: `{manifest['supersedes']['manifest']['path']}`",
        )
    for name in BUCKET_NAMES:
        values = manifest["buckets"][name]
        lines.append(
            f"| {name} | {values['target_query_count']:.1f} | "
            f"{values['query_count']} | {values['target_positive_pairs']:,} | "
            f"{values['positive_pairs']:,} | "
            f"{values['unique_epitopes']} | {values['component_count']} |"
        )
    lines.extend(
        [
            "",
            "## Fold partition statistics",
            "",
            "| Fold | Partition | Records | Peptides | pMHCs | HLAs |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for fold_name in FOLD_NAMES:
        for partition in PARTITIONS:
            values = manifest["folds"][fold_name]["partitions"][partition]
            lines.append(
                f"| {fold_name} | {partition} | {values['positive_pairs']:,} | "
                f"{values['unique_epitopes']} | {values['unique_pmhcs']} | "
                f"{values['unique_hlas']} |"
            )
    lines.extend(
        [
            "",
            "## Test-query coverage",
            "",
            "| Fold | Test queries | Same-HLA train reference | Coverage | Maximum cross-partition similarity |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for fold_name in FOLD_NAMES:
        values = manifest["folds"][fold_name]
        lines.append(
            f"| {fold_name} | {values['test_query_count']} | "
            f"{values['test_queries_with_same_hla_train_reference']} | "
            f"{values['same_hla_train_reference_coverage']:.6f} | "
            f"{values['cross_partition_similarity']['maximum_similarity']:.6f} |"
        )
    graph = manifest["graph"]
    lines.extend(
        [
            "",
            "## Audit",
            "",
            f"- Source records: {manifest['source']['positive_pairs']:,}",
            f"- Unique peptides: {graph['node_count']}",
            f"- Connected components: {graph['component_count']}",
            f"- Largest component: {graph['largest_component_peptide_count']} peptides, "
            f"{graph['largest_component_record_count']:,} records",
            f"- Maximum cross-partition similarity across all folds: "
            f"{manifest['invariants']['cross_partition_maximum_similarity']:.6f}",
            "- Every source record appears exactly once in each fold.",
            "- Every peptide and connected component appears as test exactly once.",
            "",
            f"This report is generated by `{manifest['artifacts']['generator_script']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_distance_aware_five_folds(
    config_path: Path, project_dir: Path, *, force: bool = False
) -> dict[str, Any]:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("distance-aware five-fold config schema_version must be 1")
    if config["source"].get("tier") != "development":
        raise ValueError("distance-aware five-fold source must be Development")
    if int(config.get("fold_count", 0)) != FOLD_COUNT:
        raise ValueError("distance-aware five-fold construction requires five folds")
    if config["rotation"] != {
        "test_bucket_offset": 0,
        "validation_bucket_offset": 1,
    }:
        raise ValueError("distance-aware five-fold rotation must use the next bucket")
    seed = int(config["seed"])
    threshold = float(config["similarity"]["threshold"])
    if threshold != 0.5 or config["similarity"].get(
        "edge_rule"
    ) != "similarity > threshold":
        raise ValueError("distance-aware five-fold requires the strict 0.5 threshold")

    source_path = project_dir / config["source"]["csv"]
    if file_sha256(source_path) != config["source"]["sha256"]:
        raise ValueError("Development source SHA-256 mismatch")
    fieldnames, records = _load_source(source_path)
    graph = build_similarity_graph(
        [record["epitope"] for record in records], threshold
    )
    allocation = config.get("allocation", {"objective": "record_priority"})
    objective = allocation.get("objective")
    reference_audit: dict[str, Any] | None = None
    if objective == "record_priority":
        buckets, bucket_components, targets = assign_components_to_buckets(
            records, graph, FOLD_COUNT, seed
        )
        allocation_manifest = {
            "objective": "record_priority",
            "algorithm": "greedy minimum squared record-count error",
            "component_order": "descending record count, seeded SHA-256 tie break",
            "bucket_tie_break": "seeded SHA-256",
        }
    elif objective == "joint_weighted":
        from .distance_aware_5fold_audit import (
            build_component_profiles,
            simulate_component_assignment,
        )

        query_weight = float(allocation["weights"]["query_count"])
        record_weight = float(allocation["weights"]["positive_record_count"])
        if not math.isclose(query_weight + record_weight, 1.0):
            raise ValueError("joint allocation weights must sum to one")
        profiles = build_component_profiles(records, graph)
        profile_assignments = simulate_component_assignment(
            profiles,
            objective="joint_weighted",
            seed=seed,
            bucket_count=FOLD_COUNT,
            query_weight=query_weight,
            record_weight=record_weight,
        )
        buckets, bucket_components = materialize_profile_assignments(
            records, profile_assignments
        )
        base, remainder = divmod(len(records), FOLD_COUNT)
        targets = {
            name: base + (index < remainder)
            for index, name in enumerate(BUCKET_NAMES)
        }
        allocation_manifest = {
            "objective": "joint_weighted",
            "algorithm": "deterministic greedy normalized squared error",
            "weights": {
                "query_count": query_weight,
                "positive_record_count": record_weight,
            },
            "component_order": "descending weighted normalized component magnitude, seeded SHA-256 tie break",
            "bucket_score": "weighted normalized query-count SSE plus record-count SSE",
            "bucket_tie_break": "seeded SHA-256",
        }
        audit_specification = config["reference_audit"]
        audit_path = project_dir / audit_specification["path"]
        if file_sha256(audit_path) != audit_specification["sha256"]:
            raise ValueError("reference five-fold audit SHA-256 mismatch")
        with audit_path.open("r", encoding="utf-8") as handle:
            reference_audit = json.load(handle)
        selected = audit_specification["selected_simulation"]
        expected = reference_audit["simulations"][selected]["buckets"]
        observed = {
            name: {
                "query_count": len(
                    {record["pmhc_id"] for record in buckets[name]}
                ),
                "positive_record_count": len(buckets[name]),
            }
            for name in BUCKET_NAMES
        }
        expected_counts = {
            name: {
                "query_count": int(values["query_count"]),
                "positive_record_count": int(values["positive_record_count"]),
            }
            for name, values in expected.items()
        }
        if observed != expected_counts:
            raise ValueError("joint allocation does not reproduce the audited simulation")
        allocation_manifest["reference_audit"] = {
            "path": audit_specification["path"],
            "sha256": file_sha256(audit_path),
            "selected_simulation": selected,
        }
    else:
        raise ValueError("unsupported distance-aware five-fold allocation objective")
    folds = build_fold_partitions(buckets, bucket_components)
    validation = validate_five_folds(
        records,
        graph,
        buckets,
        bucket_components,
        folds,
        fieldnames,
        threshold,
    )

    output_dir = project_dir / config["outputs"]["directory"]
    pattern = config["outputs"]["csv_pattern"]
    csv_paths = {
        fold_name: {
            partition: output_dir
            / pattern.format(fold=fold_name, partition=partition)
            for partition in PARTITIONS
        }
        for fold_name in FOLD_NAMES
    }
    manifest_path = output_dir / config["outputs"]["manifest_json"]
    report_path = project_dir / config["outputs"]["report_md"]
    supersedes = config.get("supersedes")
    superseded_marker_path: Path | None = None
    superseded_marker_bytes: bytes | None = None
    if supersedes is not None:
        old_manifest_path = project_dir / supersedes["manifest"]
        if file_sha256(old_manifest_path) != supersedes["sha256"]:
            raise ValueError("superseded five-fold manifest SHA-256 mismatch")
        superseded_marker_path = project_dir / supersedes["marker"]
        superseded_marker = {
            "schema_version": 1,
            "status": "superseded",
            "reason": "replaced by deterministic joint query-count and record-count allocation",
            "manifest": {
                "path": supersedes["manifest"],
                "sha256": file_sha256(old_manifest_path),
            },
            "superseded_by": {
                "manifest": manifest_path.relative_to(project_dir).as_posix(),
                "allocation_objective": objective,
            },
            "existing_artifacts_preserved": True,
        }
        superseded_marker_bytes = (
            json.dumps(superseded_marker, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    output_paths = [
        path for fold_paths in csv_paths.values() for path in fold_paths.values()
    ] + [manifest_path, report_path]
    if superseded_marker_path is not None:
        output_paths.append(superseded_marker_path)
    if not force and any(path.exists() for path in output_paths):
        raise FileExistsError("refusing to overwrite distance-aware five-fold outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold_name in FOLD_NAMES:
        for partition in PARTITIONS:
            csv_paths[fold_name][partition].write_bytes(
                _csv_bytes(
                    fieldnames,
                    folds[fold_name]["partitions"][partition],
                )
            )

    written_folds = build_fold_partitions(buckets, bucket_components)
    for fold_name in FOLD_NAMES:
        written_folds[fold_name]["partitions"] = {
            partition: _read_partition(
                csv_paths[fold_name][partition], fieldnames
            )
            for partition in PARTITIONS
        }
    validation = validate_five_folds(
        records,
        graph,
        buckets,
        bucket_components,
        written_folds,
        fieldnames,
        threshold,
    )
    folds = written_folds

    peptide_record_counts = Counter(record["epitope"] for record in records)
    component_record_counts = {
        component: sum(peptide_record_counts[peptide] for peptide in component)
        for component in graph.components
    }
    fold_hashes = canonical_fold_sha256(fieldnames, folds)
    generation_sha = hashlib.sha256(
        json.dumps(
            fold_hashes, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
    ).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "fold_count": FOLD_COUNT,
        "tier": "development",
        "similarity": {
            "definition": config["similarity"]["definition"],
            "threshold": threshold,
            "edge_rule": config["similarity"]["edge_rule"],
        },
        "rotation": config["rotation"],
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
            "largest_component_record_count": max(component_record_counts.values()),
        },
        "status": "active",
        "bucket_allocation": allocation_manifest,
        "buckets": {
            name: {
                "target_positive_pairs": targets[name],
                "positive_pairs": len(buckets[name]),
                "target_deviation": len(buckets[name]) - targets[name],
                "target_query_count": len(
                    {record["pmhc_id"] for record in records}
                )
                / FOLD_COUNT,
                "query_count": len(
                    {record["pmhc_id"] for record in buckets[name]}
                ),
                "fraction_of_tier": len(buckets[name]) / len(records),
                "unique_epitopes": len(
                    {record["epitope"] for record in buckets[name]}
                ),
                "unique_pmhcs": len(
                    {record["pmhc_id"] for record in buckets[name]}
                ),
                "unique_hlas": len({record["hla"] for record in buckets[name]}),
                "component_count": len(bucket_components[name]),
                "component_sha256": _component_sha256(bucket_components[name]),
            }
            for name in BUCKET_NAMES
        },
        "folds": {},
        "fold_csv_sha256": fold_hashes,
        "generation_sha256": generation_sha,
        "invariants": validation["invariants"],
        "artifacts": {
            "manifest_json": manifest_path.relative_to(project_dir).as_posix(),
            "report_md": report_path.relative_to(project_dir).as_posix(),
            "generator_script": config["outputs"].get(
                "generator_script", "scripts/build_distance_aware_5fold.py"
            ),
        },
        "negative_samples_generated": False,
        "models_trained_or_evaluated": False,
    }
    if supersedes is not None:
        assert superseded_marker_path is not None
        assert superseded_marker_bytes is not None
        manifest["supersedes"] = {
            "manifest": {
                "path": supersedes["manifest"],
                "sha256": supersedes["sha256"],
            },
            "marker": {
                "path": superseded_marker_path.relative_to(project_dir).as_posix(),
                "sha256": hashlib.sha256(superseded_marker_bytes).hexdigest(),
            },
            "existing_artifacts_preserved": True,
        }
    for fold_name in FOLD_NAMES:
        fold = folds[fold_name]
        reference_stats = _test_reference_stats(
            fold["partitions"]["train"], fold["partitions"]["test"]
        )
        manifest["folds"][fold_name] = {
            "index": fold["index"],
            "test_bucket": fold["test_bucket"],
            "validation_bucket": fold["validation_bucket"],
            "train_buckets": list(fold["train_buckets"]),
            "partitions": {
                partition: _partition_stats(
                    csv_paths[fold_name][partition],
                    fold["partitions"][partition],
                    fold["components"][partition],
                    len(records),
                )
                for partition in PARTITIONS
            },
            **reference_stats,
            "cross_partition_similarity": validation["folds"][fold_name][
                "cross_similarity"
            ],
            "invariants": validation["folds"][fold_name]["invariants"],
        }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(manifest), encoding="utf-8")
    if superseded_marker_path is not None:
        assert superseded_marker_bytes is not None
        superseded_marker_path.parent.mkdir(parents=True, exist_ok=True)
        superseded_marker_path.write_bytes(superseded_marker_bytes)
        if file_sha256(superseded_marker_path) != manifest["supersedes"]["marker"][
            "sha256"
        ]:
            raise ValueError("superseded marker SHA-256 mismatch")
        old_manifest_path = project_dir / supersedes["manifest"]
        if file_sha256(old_manifest_path) != supersedes["sha256"]:
            raise ValueError("superseded five-fold manifest changed during generation")
    return manifest
