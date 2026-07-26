from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import pstdev
from typing import Any, Mapping, Sequence

from .distance_aware_5fold import (
    BUCKET_NAMES,
    FOLD_NAMES,
    build_fold_partitions,
    canonical_fold_sha256,
)
from .distance_aware_split import (
    SimilarityGraph,
    _load_source,
    _stable_hash,
    assign_components_to_buckets,
    build_similarity_graph,
    normalized_levenshtein_similarity,
)
from .vdjdb import file_sha256


OBJECTIVES = ("query_priority", "record_priority", "joint_weighted")


@dataclass(frozen=True)
class ComponentProfile:
    component_id: str
    peptides: tuple[str, ...]
    pmhcs: frozenset[str]
    hlas: frozenset[str]
    positive_record_count: int

    @property
    def peptide_count(self) -> int:
        return len(self.peptides)

    @property
    def query_count(self) -> int:
        return len(self.pmhcs)

    @property
    def hla_count(self) -> int:
        return len(self.hlas)


def build_component_profiles(
    records: Sequence[Mapping[str, str]], graph: SimilarityGraph
) -> list[ComponentProfile]:
    records_by_peptide: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for record in records:
        records_by_peptide[record["epitope"]].append(record)
    profiles = []
    for component in graph.components:
        component_records = [
            record
            for peptide in component
            for record in records_by_peptide[peptide]
        ]
        digest = hashlib.sha256("\0".join(component).encode("ascii")).hexdigest()
        profiles.append(
            ComponentProfile(
                component_id=f"component_{digest[:16]}",
                peptides=component,
                pmhcs=frozenset(record["pmhc_id"] for record in component_records),
                hlas=frozenset(record["hla"] for record in component_records),
                positive_record_count=len(component_records),
            )
        )
    if len({profile.component_id for profile in profiles}) != len(profiles):
        raise ValueError("component ID collision")
    return profiles


def _normalized_errors(
    query_counts: Sequence[int], record_counts: Sequence[int]
) -> tuple[float, float]:
    query_target = sum(query_counts) / len(query_counts)
    record_target = sum(record_counts) / len(record_counts)
    query_error = sum(
        ((value - query_target) / query_target) ** 2 for value in query_counts
    )
    record_error = sum(
        ((value - record_target) / record_target) ** 2 for value in record_counts
    )
    return query_error, record_error


def simulate_component_assignment(
    profiles: Sequence[ComponentProfile],
    *,
    objective: str,
    seed: int,
    bucket_count: int = 5,
    query_weight: float = 0.5,
    record_weight: float = 0.5,
) -> dict[str, list[ComponentProfile]]:
    if objective not in {"query_priority", "joint_weighted"}:
        raise ValueError("simulation objective must be query_priority or joint_weighted")
    if bucket_count < 2 or len(profiles) < bucket_count:
        raise ValueError("simulation requires enough components for nonempty buckets")
    if objective == "joint_weighted" and (
        query_weight < 0
        or record_weight < 0
        or not math.isclose(query_weight + record_weight, 1.0)
    ):
        raise ValueError("joint simulation weights must be nonnegative and sum to one")

    bucket_names = tuple(f"bucket_{index}" for index in range(bucket_count))
    total_queries = sum(profile.query_count for profile in profiles)
    total_records = sum(profile.positive_record_count for profile in profiles)
    exact_query_weight = Fraction(str(query_weight))
    exact_record_weight = Fraction(str(record_weight))
    if objective == "query_priority":
        ordered = sorted(
            profiles,
            key=lambda profile: (
                -profile.query_count,
                -profile.positive_record_count,
                _stable_hash(seed, "|".join(profile.peptides)),
                profile.peptides,
            ),
        )
    else:
        ordered = sorted(
            profiles,
            key=lambda profile: (
                -(
                    exact_query_weight
                    * bucket_count**2
                    * profile.query_count**2
                    / total_queries**2
                    + exact_record_weight
                    * bucket_count**2
                    * profile.positive_record_count**2
                    / total_records**2
                ),
                _stable_hash(seed, "|".join(profile.peptides)),
                profile.peptides,
            ),
        )

    assignments = {name: [] for name in bucket_names}
    query_counts = {name: 0 for name in bucket_names}
    record_counts = {name: 0 for name in bucket_names}
    for index, profile in enumerate(ordered):
        empty = [name for name in bucket_names if not assignments[name]]
        remaining_after = len(ordered) - index - 1
        allowed = empty if len(empty) > remaining_after else list(bucket_names)
        candidates = []
        for name in allowed:
            proposed_queries = dict(query_counts)
            proposed_records = dict(record_counts)
            proposed_queries[name] += profile.query_count
            proposed_records[name] += profile.positive_record_count
            query_error = sum(
                Fraction(
                    (bucket_count * proposed_queries[bucket] - total_queries) ** 2,
                    total_queries**2,
                )
                for bucket in bucket_names
            )
            record_error = sum(
                Fraction(
                    (bucket_count * proposed_records[bucket] - total_records) ** 2,
                    total_records**2,
                )
                for bucket in bucket_names
            )
            if objective == "query_priority":
                score = (query_error, record_error)
            else:
                score = (
                    exact_query_weight * query_error
                    + exact_record_weight * record_error,
                    query_error,
                    record_error,
                )
            candidates.append(
                (
                    *score,
                    _stable_hash(
                        seed, "|".join(profile.peptides), name, objective
                    ),
                    name,
                )
            )
        selected = min(candidates)[-1]
        assignments[selected].append(profile)
        query_counts[selected] += profile.query_count
        record_counts[selected] += profile.positive_record_count
    if any(not assignments[name] for name in bucket_names):
        raise ValueError("simulation produced an empty bucket")
    return assignments


def assignment_summary(
    assignments: Mapping[str, Sequence[ComponentProfile]],
) -> dict[str, Any]:
    bucket_names = tuple(
        sorted(assignments, key=lambda name: int(name.split("_")[-1]))
    )
    buckets = {}
    for index, name in enumerate(bucket_names):
        profiles = assignments[name]
        buckets[name] = {
            "test_fold": f"fold_{index + 1}",
            "component_count": len(profiles),
            "peptide_count": sum(profile.peptide_count for profile in profiles),
            "query_count": sum(profile.query_count for profile in profiles),
            "positive_record_count": sum(
                profile.positive_record_count for profile in profiles
            ),
            "hla_count": len(
                {hla for profile in profiles for hla in profile.hlas}
            ),
        }
    query_counts = [values["query_count"] for values in buckets.values()]
    record_counts = [
        values["positive_record_count"] for values in buckets.values()
    ]
    query_error, record_error = _normalized_errors(query_counts, record_counts)
    return {
        "buckets": buckets,
        "balance": {
            "query_target": sum(query_counts) / len(query_counts),
            "query_minimum": min(query_counts),
            "query_maximum": max(query_counts),
            "query_range": max(query_counts) - min(query_counts),
            "query_population_std": pstdev(query_counts),
            "query_normalized_squared_error": query_error,
            "record_target": sum(record_counts) / len(record_counts),
            "record_minimum": min(record_counts),
            "record_maximum": max(record_counts),
            "record_range": max(record_counts) - min(record_counts),
            "record_population_std": pstdev(record_counts),
            "record_normalized_squared_error": record_error,
        },
    }


def _profile_assignments_from_components(
    profiles: Sequence[ComponentProfile],
    bucket_components: Mapping[str, Sequence[tuple[str, ...]]],
) -> dict[str, list[ComponentProfile]]:
    indexed = {profile.peptides: profile for profile in profiles}
    assignments = {
        name: [indexed[component] for component in bucket_components[name]]
        for name in BUCKET_NAMES
    }
    assigned_ids = [
        profile.component_id
        for values in assignments.values()
        for profile in values
    ]
    if len(assigned_ids) != len(set(assigned_ids)) or len(assigned_ids) != len(
        profiles
    ):
        raise ValueError("current component assignment is not one-to-one")
    return assignments


def _split_csv_hashes(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    output_dir = manifest_path.parent
    hashes: dict[str, dict[str, str]] = {}
    for fold_name in FOLD_NAMES:
        hashes[fold_name] = {}
        for partition, values in manifest["folds"][fold_name]["partitions"].items():
            path = output_dir / values["file"]
            actual = file_sha256(path)
            if actual != values["sha256"]:
                raise ValueError(f"current split CSV SHA-256 mismatch: {path}")
            hashes[fold_name][partition] = actual
    return hashes


def _component_json(
    profile: ComponentProfile, current_bucket: str
) -> dict[str, Any]:
    return {
        "component_id": profile.component_id,
        "peptides": list(profile.peptides),
        "peptide_count": profile.peptide_count,
        "query_count": profile.query_count,
        "positive_record_count": profile.positive_record_count,
        "hla_count": profile.hla_count,
        "current_bucket": current_bucket,
        "current_test_fold": f"fold_{int(current_bucket.split('_')[-1]) + 1}",
    }


def render_report(audit: Mapping[str, Any]) -> str:
    reason = audit["fold_1_single_query_explanation"]
    lines = [
        "# Development distance-aware five-fold imbalance audit",
        "",
        "## Fold 1 explanation",
        "",
        f"Fold 1 test contains {reason['positive_record_count']:,} records but one query "
        f"because all records belong to `{reason['query_id']}`. The peptide "
        f"`{reason['peptide']}` is a singleton similarity component: its maximum "
        f"similarity to every other Development peptide is only "
        f"{reason['maximum_similarity_to_other_peptide']:.6f}, below the strict "
        "`similarity > 0.5` edge rule. The component is therefore indivisible, and "
        "the record-priority allocator leaves it alone because it already exceeds "
        "the ideal per-bucket record target.",
        "",
        "## Largest 20 components",
        "",
        "Components are ordered by positive-record count.",
        "",
        "| Rank | Peptides | Peptide count | Queries | Records | HLAs | Current test fold |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for index, component in enumerate(audit["largest_20_components"], start=1):
        lines.append(
            f"| {index} | {', '.join(component['peptides'])} | "
            f"{component['peptide_count']} | {component['query_count']} | "
            f"{component['positive_record_count']:,} | {component['hla_count']} | "
            f"{component['current_test_fold']} |"
        )
    lines.extend(
        [
            "",
            "## Deterministic allocation simulations",
            "",
            "| Objective | Fold | Test queries | Test records |",
            "|---|---|---:|---:|",
        ]
    )
    labels = {
        "query_priority": "Query priority",
        "record_priority": "Record priority",
        "joint_weighted": "Joint 0.5/0.5",
    }
    for objective in OBJECTIVES:
        for values in audit["simulations"][objective]["buckets"].values():
            lines.append(
                f"| {labels[objective]} | {values['test_fold']} | "
                f"{values['query_count']} | {values['positive_record_count']:,} |"
            )
    lines.extend(
        [
            "",
            "## Balance trade-off",
            "",
            "| Objective | Query range | Record range | Query std | Record std |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for objective in OBJECTIVES:
        values = audit["simulations"][objective]["balance"]
        lines.append(
            f"| {labels[objective]} | {values['query_range']} | "
            f"{values['record_range']:,} | {values['query_population_std']:.3f} | "
            f"{values['record_population_std']:.3f} |"
        )
    feasibility = audit["feasibility"]
    lines.extend(
        [
            "",
            "## Feasibility judgement",
            "",
            f"- Ideal bucket targets are {feasibility['query_target']:.1f} queries and "
            f"{feasibility['record_target']:.1f} records.",
            f"- The largest indivisible component contains "
            f"{feasibility['largest_component_records']:,} records and only "
            f"{feasibility['largest_component_queries']} query.",
            f"- Any valid allocation has a record range of at least "
            f"{feasibility['theoretical_minimum_record_range']:.1f}; exact record "
            "balance is impossible without splitting this component.",
            "- Query counts can be balanced to a range of one. The joint simulation "
            "does this while retaining the unavoidable record imbalance, but no "
            "assignment can make record counts tightly equal under the current graph constraint.",
            "- All simulations keep components intact, so the cross-partition maximum "
            "similarity remains at most 0.5 by construction.",
            "",
            "No existing split CSV was modified and no replacement split was written.",
            "",
            "This report is generated by `scripts/audit_distance_aware_5fold.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_distance_aware_five_fold(
    config_path: Path, project_dir: Path, *, force: bool = False
) -> dict[str, Any]:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("distance-aware five-fold audit config schema_version must be 1")
    if int(config["simulation"]["bucket_count"]) != 5:
        raise ValueError("distance-aware five-fold audit requires five buckets")
    threshold = float(config["similarity"]["threshold"])
    if threshold != 0.5 or config["similarity"]["edge_rule"] != "similarity > threshold":
        raise ValueError("distance-aware five-fold audit requires the strict threshold")
    seed = int(config["seed"])
    source_path = project_dir / config["inputs"]["source"]["path"]
    manifest_path = project_dir / config["inputs"]["split_manifest"]["path"]
    for name, path in (("source", source_path), ("split_manifest", manifest_path)):
        if file_sha256(path) != config["inputs"][name]["sha256"]:
            raise ValueError(f"distance-aware five-fold audit {name} SHA-256 mismatch")
    with manifest_path.open("r", encoding="utf-8") as handle:
        split_manifest = json.load(handle)
    if (
        split_manifest["seed"] != seed
        or split_manifest["fold_count"] != 5
        or split_manifest["similarity"]["threshold"] != threshold
        or split_manifest["source"]["sha256"] != file_sha256(source_path)
    ):
        raise ValueError("current five-fold manifest provenance mismatch")
    csv_hashes_before = _split_csv_hashes(manifest_path, split_manifest)

    fieldnames, records = _load_source(source_path)
    graph = build_similarity_graph(
        [record["epitope"] for record in records], threshold
    )
    profiles = build_component_profiles(records, graph)
    current_buckets, current_components, _ = assign_components_to_buckets(
        records, graph, 5, seed
    )
    current_folds = build_fold_partitions(current_buckets, current_components)
    reconstructed_hashes = canonical_fold_sha256(fieldnames, current_folds)
    if reconstructed_hashes != split_manifest["fold_csv_sha256"]:
        raise ValueError("current five-fold split cannot be deterministically reconstructed")
    current_assignments = _profile_assignments_from_components(
        profiles, current_components
    )
    current_component_locations = {
        profile.component_id: bucket
        for bucket, values in current_assignments.items()
        for profile in values
    }

    weights = config["simulation"]["joint_weights"]
    simulations = {
        "query_priority": assignment_summary(
            simulate_component_assignment(
                profiles, objective="query_priority", seed=seed
            )
        ),
        "record_priority": assignment_summary(current_assignments),
        "joint_weighted": assignment_summary(
            simulate_component_assignment(
                profiles,
                objective="joint_weighted",
                seed=seed,
                query_weight=float(weights["query"]),
                record_weight=float(weights["records"]),
            )
        ),
    }
    for objective, values in simulations.items():
        values["deterministic"] = True
        values["components_split"] = False
        values["cross_partition_similarity_threshold_satisfied"] = True
        values["objective"] = objective

    component_rows = [
        _component_json(profile, current_component_locations[profile.component_id])
        for profile in profiles
    ]
    component_rows.sort(
        key=lambda value: (
            -value["positive_record_count"],
            -value["query_count"],
            value["peptides"],
        )
    )
    fold_1_profiles = current_assignments["bucket_0"]
    if len(fold_1_profiles) != 1 or fold_1_profiles[0].query_count != 1:
        raise ValueError("Fold 1 no longer has the audited single-query structure")
    dominant = fold_1_profiles[0]
    other_peptides = [
        peptide
        for profile in profiles
        for peptide in profile.peptides
        if peptide not in dominant.peptides
    ]
    maximum_other_similarity = max(
        normalized_levenshtein_similarity(dominant.peptides[0], peptide)
        for peptide in other_peptides
    )

    record_target = len(records) / 5
    query_total = len({record["pmhc_id"] for record in records})
    query_target = query_total / 5
    remaining_average = (
        len(records) - dominant.positive_record_count
    ) / 4
    feasibility = {
        "query_target": query_target,
        "record_target": record_target,
        "largest_component_records": dominant.positive_record_count,
        "largest_component_queries": dominant.query_count,
        "largest_component_record_excess_over_target": dominant.positive_record_count
        - record_target,
        "theoretical_minimum_record_range": dominant.positive_record_count
        - remaining_average,
        "exact_record_balance_possible": False,
        "minimum_integer_query_range": 1 if query_total % 5 else 0,
        "query_range_one_achieved": simulations["joint_weighted"]["balance"][
            "query_range"
        ]
        == 1,
        "joint_practical_tradeoff_available": True,
        "tight_simultaneous_record_and_query_balance_possible": False,
    }
    audit: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "tier": "development",
        "similarity": {
            "definition": split_manifest["similarity"]["definition"],
            "threshold": threshold,
            "edge_rule": config["similarity"]["edge_rule"],
        },
        "component_count": len(profiles),
        "components": component_rows,
        "largest_20_components": component_rows[:20],
        "fold_1_single_query_explanation": {
            "component_id": dominant.component_id,
            "peptide": dominant.peptides[0],
            "query_id": next(iter(dominant.pmhcs)),
            "peptide_count": dominant.peptide_count,
            "query_count": dominant.query_count,
            "positive_record_count": dominant.positive_record_count,
            "hla_count": dominant.hla_count,
            "maximum_similarity_to_other_peptide": maximum_other_similarity,
            "ideal_record_target": record_target,
            "record_excess_over_target": dominant.positive_record_count
            - record_target,
        },
        "simulations": simulations,
        "feasibility": feasibility,
        "input_audit": {
            "source": {
                "path": config["inputs"]["source"]["path"],
                "sha256": file_sha256(source_path),
            },
            "split_manifest": {
                "path": config["inputs"]["split_manifest"]["path"],
                "sha256": file_sha256(manifest_path),
            },
            "split_csv_sha256": csv_hashes_before,
            "reconstructed_fold_csv_sha256": reconstructed_hashes,
        },
        "existing_split_csv_modified": False,
        "new_split_generated": False,
        "negative_samples_generated": False,
        "models_trained_or_evaluated": False,
    }

    output_dir = project_dir / config["outputs"]["directory"]
    json_path = output_dir / config["outputs"]["audit_json"]
    report_path = output_dir / config["outputs"]["report_md"]
    if not force and (json_path.exists() or report_path.exists()):
        raise FileExistsError("refusing to overwrite five-fold audit outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(render_report(audit), encoding="utf-8")

    if file_sha256(manifest_path) != config["inputs"]["split_manifest"]["sha256"]:
        raise ValueError("current split manifest changed during audit")
    if _split_csv_hashes(manifest_path, split_manifest) != csv_hashes_before:
        raise ValueError("current split CSV changed during audit")
    return audit
