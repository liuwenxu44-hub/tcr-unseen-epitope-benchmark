from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .vdjdb import file_sha256


PARTITIONS = ("train", "validation", "test")
QUERY_COLUMNS = ("epitope", "hla")
INTERACTION_COLUMNS = ("tcr_cdr3_beta", "epitope", "hla")
BASE_AUDIT_COLUMNS = (
    "fold_id",
    "partition",
    "query_id",
    "epitope",
    "hla",
    "positive_count",
    "current_n1_negative_count",
    "current_positive_fraction",
    "unique_positive_tcr_count",
    "partition_unique_tcr_count",
    "known_positive_tcr_exclusion_count",
    "legal_negative_candidate_count",
)


def query_id(epitope: str, hla: str) -> str:
    return f"{epitope.strip().upper()}|{hla.strip().upper()}"


def _query_key(row: Mapping[str, str]) -> tuple[str, str]:
    return row["epitope"].strip().upper(), row["hla"].strip().upper()


def _interaction_key(row: Mapping[str, str]) -> tuple[str, str, str]:
    return (
        row["tcr_cdr3_beta"].strip().upper(),
        row["epitope"].strip().upper(),
        row["hla"].strip().upper(),
    )


def _read_csv(path: Path, *, expected_label: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {*INTERACTION_COLUMNS, "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"audit input is missing required columns: {path}")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if row["label"].strip() != expected_label:
                raise ValueError(f"unexpected label at {path}:{row_number}")
            normalized = dict(row)
            for column in INTERACTION_COLUMNS:
                normalized[column] = row[column].strip().upper()
            rows.append(normalized)
    if not rows:
        raise ValueError(f"audit input is empty: {path}")
    return rows


def load_positive_registry_by_query(
    path: Path,
) -> tuple[dict[tuple[str, str], set[str]], int]:
    rows = _read_csv(path, expected_label="1")
    registry: dict[tuple[str, str], set[str]] = defaultdict(set)
    interactions = set()
    for row in rows:
        key = _interaction_key(row)
        interactions.add(key)
        registry[_query_key(row)].add(key[0])
    if len(interactions) != len(rows):
        raise ValueError("complete positive registry contains duplicate interactions")
    return dict(registry), len(interactions)


def compute_partition_query_audit(
    positives: Sequence[Mapping[str, str]],
    negatives: Sequence[Mapping[str, str]],
    registry_by_query: Mapping[tuple[str, str], set[str]],
    *,
    fold_id: str,
    partition: str,
    caps: Sequence[int],
) -> list[dict[str, Any]]:
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition: {partition}")
    if not caps or any(int(cap) <= 0 for cap in caps) or len(set(caps)) != len(caps):
        raise ValueError("query caps must be unique positive integers")
    positive_keys = [_interaction_key(row) for row in positives]
    negative_keys = [_interaction_key(row) for row in negatives]
    if len(positive_keys) != len(set(positive_keys)):
        raise ValueError(f"duplicate positives in {fold_id}/{partition}")
    if len(negative_keys) != len(set(negative_keys)):
        raise ValueError(f"duplicate N1 negatives in {fold_id}/{partition}")

    positives_by_query: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    negatives_by_query: Counter[tuple[str, str]] = Counter()
    for row in positives:
        positives_by_query[_query_key(row)].append(row)
    for row in negatives:
        negatives_by_query[_query_key(row)] += 1
    unknown_negative_queries = set(negatives_by_query) - set(positives_by_query)
    if unknown_negative_queries:
        raise ValueError(
            f"N1 negatives use queries outside positives in {fold_id}/{partition}"
        )

    partition_tcrs = {key[0] for key in positive_keys}
    audit_rows = []
    for epitope, hla in sorted(positives_by_query):
        query = (epitope, hla)
        query_positives = positives_by_query[query]
        positive_count = len(query_positives)
        negative_count = int(negatives_by_query[query])
        known_positive_tcrs = partition_tcrs & registry_by_query.get(query, set())
        legal_candidates = partition_tcrs - known_positive_tcrs
        row: dict[str, Any] = {
            "fold_id": fold_id,
            "partition": partition,
            "query_id": query_id(epitope, hla),
            "epitope": epitope,
            "hla": hla,
            "positive_count": positive_count,
            "current_n1_negative_count": negative_count,
            "current_positive_fraction": positive_count
            / (positive_count + negative_count),
            "unique_positive_tcr_count": len(
                {record["tcr_cdr3_beta"] for record in query_positives}
            ),
            "partition_unique_tcr_count": len(partition_tcrs),
            "known_positive_tcr_exclusion_count": len(known_positive_tcrs),
            "legal_negative_candidate_count": len(legal_candidates),
        }
        for cap in caps:
            retained = min(positive_count, len(legal_candidates), int(cap))
            row[f"retained_per_class_cap_{cap}"] = retained
            row[f"retained_total_records_cap_{cap}"] = 2 * retained
            row[f"exclusion_reason_cap_{cap}"] = (
                "no_legal_negative_candidates" if retained == 0 else ""
            )
        audit_rows.append(row)
    return audit_rows


def canonical_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        [dict(sorted(row.items())) for row in rows],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nearest_rank(values: Sequence[int], fraction: float) -> int:
    if not values:
        raise ValueError("distribution values are empty")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(fraction * len(ordered) + 0.999999) - 1))
    return ordered[index]


def summarize_cap(
    rows: Sequence[Mapping[str, Any]], cap: int
) -> dict[str, Any]:
    retained_column = f"retained_per_class_cap_{cap}"
    reason_column = f"exclusion_reason_cap_{cap}"
    retained_rows = [row for row in rows if int(row[retained_column]) > 0]
    excluded_rows = [row for row in rows if int(row[retained_column]) == 0]
    retained_values = [int(row[retained_column]) for row in retained_rows]
    reason_counts = Counter(str(row[reason_column]) for row in excluded_rows)
    total_per_class = sum(retained_values)
    frequency = Counter(retained_values)
    distribution = {
        "minimum": min(retained_values) if retained_values else None,
        "p25": _nearest_rank(retained_values, 0.25) if retained_values else None,
        "median": _nearest_rank(retained_values, 0.5) if retained_values else None,
        "p75": _nearest_rank(retained_values, 0.75) if retained_values else None,
        "maximum": max(retained_values) if retained_values else None,
        "mean": total_per_class / len(retained_values) if retained_values else None,
        "per_class_count_frequency": {
            str(value): frequency[value] for value in sorted(frequency)
        },
    }
    return {
        "cap": cap,
        "retained_query_count": len(retained_rows),
        "excluded_query_count": len(excluded_rows),
        "excluded_query_reasons": {
            reason: reason_counts[reason] for reason in sorted(reason_counts)
        },
        "retained_positive_count": total_per_class,
        "retained_negative_count": total_per_class,
        "retained_total_record_count": 2 * total_per_class,
        "per_query_retained_per_class_distribution": distribution,
        "maximum_query_fraction_of_all_records": (
            max(retained_values) / total_per_class if retained_values else None
        ),
    }


def build_simulations(
    rows: Sequence[Mapping[str, Any]],
    fold_ids: Sequence[str],
    partitions: Sequence[str],
    caps: Sequence[int],
) -> dict[str, Any]:
    simulations = {}
    for cap in caps:
        folds = {}
        for fold_id in fold_ids:
            fold_partitions = {}
            for partition in partitions:
                selected = [
                    row
                    for row in rows
                    if row["fold_id"] == fold_id and row["partition"] == partition
                ]
                if not selected:
                    raise ValueError(f"missing audit rows for {fold_id}/{partition}")
                fold_partitions[partition] = summarize_cap(selected, int(cap))
            folds[fold_id] = {"partitions": fold_partitions}
        simulations[f"cap_{cap}"] = {"cap": int(cap), "folds": folds}
    return simulations


def audit_csv_text(rows: Sequence[Mapping[str, Any]], caps: Sequence[int]) -> str:
    columns = list(BASE_AUDIT_COLUMNS)
    for cap in caps:
        columns.extend(
            (
                f"retained_per_class_cap_{cap}",
                f"retained_total_records_cap_{cap}",
                f"exclusion_reason_cap_{cap}",
            )
        )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Distance-aware five-fold query-balance audit",
        "",
        "## Candidate definition",
        "",
        "- Query key: `(epitope, hla)`.",
        "- Candidate TCRs are unique TCRs from the current fold partition positives.",
        "- TCR-query interactions in the complete positive registry are excluded.",
        "- Simulation only computes counts; it does not select or write balanced samples.",
        "",
        "## Target query capacity",
        "",
        "| Fold | Partition | Positives | Current N1 | Legal N1 capacity | Partition TCRs |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["target_query_occurrences"]:
        lines.append(
            f"| {row['fold_id']} | {row['partition']} | {row['positive_count']:,} | "
            f"{row['current_n1_negative_count']:,} | "
            f"{row['legal_negative_candidate_count']:,} | "
            f"{row['partition_unique_tcr_count']:,} |"
        )
    lines.extend(
        [
            "",
            "## Cap simulations",
            "",
            "| Cap | Fold | Partition | Retained queries | Excluded queries | Positives | Negatives | Median per class | Maximum per class | Largest query fraction |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cap_key in sorted(
        summary["simulations"], key=lambda value: int(value.split("_")[1])
    ):
        cap_entry = summary["simulations"][cap_key]
        for fold_id in summary["fold_ids"]:
            for partition in summary["partitions"]:
                values = cap_entry["folds"][fold_id]["partitions"][partition]
                distribution = values["per_query_retained_per_class_distribution"]
                lines.append(
                    f"| {cap_entry['cap']} | {fold_id} | {partition} | "
                    f"{values['retained_query_count']} | {values['excluded_query_count']} | "
                    f"{values['retained_positive_count']:,} | "
                    f"{values['retained_negative_count']:,} | "
                    f"{distribution['median']} | {distribution['maximum']} | "
                    f"{values['maximum_query_fraction_of_all_records']:.6%} |"
                )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            "- All 30 source CSV SHA-256 values match their manifests.",
            "- Positive registry SHA-256 matches the frozen configuration.",
            "- Each partition calculation was rebuilt twice with identical canonical SHA-256.",
            "- Existing CSV files were not modified; no negatives were resampled and no model was trained.",
            "",
            "This report is generated by `scripts/audit_query_balance.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_query_balance(
    config_path: Path, project_dir: Path, *, force: bool = False
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("query-balance audit config schema mismatch")
    fold_ids = tuple(config["fold_ids"])
    partitions = tuple(config["partitions"])
    caps = tuple(int(cap) for cap in config["per_query_caps"])
    if fold_ids != tuple(f"fold_{index}" for index in range(1, 6)):
        raise ValueError("query-balance audit requires fold_1 through fold_5")
    if partitions != PARTITIONS or caps != (50, 100, 200, 500, 1000):
        raise ValueError("query-balance audit partitions or caps are not frozen")
    if config["query_key"] != list(QUERY_COLUMNS):
        raise ValueError("query key mismatch")
    if config["interaction_key"] != list(INTERACTION_COLUMNS):
        raise ValueError("interaction key mismatch")

    inputs = config["inputs"]
    input_paths = {
        name: project_dir / inputs[name]["path"]
        for name in ("positive_manifest", "negative_manifest", "positive_registry")
    }
    for name, path in input_paths.items():
        if file_sha256(path) != inputs[name]["sha256"]:
            raise ValueError(f"{name} SHA-256 mismatch")
    positive_manifest = json.loads(
        input_paths["positive_manifest"].read_text(encoding="utf-8")
    )
    negative_manifest = json.loads(
        input_paths["negative_manifest"].read_text(encoding="utf-8")
    )
    if (
        positive_manifest.get("status") != "active"
        or positive_manifest.get("negative_samples_generated") is not False
        or positive_manifest.get("fold_count") != 5
    ):
        raise ValueError("positive manifest is not the active rebalanced five-fold split")
    if (
        negative_manifest.get("negative_strategy") != "N1"
        or negative_manifest.get("positive_split_strategy")
        != "distance_aware_5fold_rebalanced"
        or negative_manifest.get("fold_count") != 5
    ):
        raise ValueError("negative manifest is not rebalanced five-fold N1")
    if (
        negative_manifest["positive_splits_manifest"]["sha256"]
        != inputs["positive_manifest"]["sha256"]
    ):
        raise ValueError("positive and negative manifests are not linked")

    registry_by_query, registry_size = load_positive_registry_by_query(
        input_paths["positive_registry"]
    )
    positive_directory = project_dir / inputs["positive_directory"]
    negative_directory = project_dir / inputs["negative_directory"]
    audit_rows = []
    input_file_audit = {}
    reproducibility_hashes = {}
    for fold_id in fold_ids:
        fold_files = {}
        fold_hashes = {}
        for partition in partitions:
            positive_entry = positive_manifest["folds"][fold_id]["partitions"][partition]
            negative_entry = negative_manifest["folds"][fold_id]["partitions"][partition]
            positive_path = positive_directory / positive_entry["file"]
            negative_path = negative_directory / negative_entry["negative_output"]["file"]
            positive_sha = file_sha256(positive_path)
            negative_sha = file_sha256(negative_path)
            if positive_sha != positive_entry["sha256"]:
                raise ValueError(f"positive CSV SHA-256 mismatch: {fold_id}/{partition}")
            if negative_sha != negative_entry["negative_output"]["sha256"]:
                raise ValueError(f"N1 CSV SHA-256 mismatch: {fold_id}/{partition}")
            positives = _read_csv(positive_path, expected_label="1")
            negatives = _read_csv(negative_path, expected_label="0")
            if len(positives) != int(positive_entry["positive_pairs"]):
                raise ValueError(f"positive row-count mismatch: {fold_id}/{partition}")
            if len(negatives) != int(negative_entry["negative_pairs"]):
                raise ValueError(f"N1 row-count mismatch: {fold_id}/{partition}")
            first = compute_partition_query_audit(
                positives,
                negatives,
                registry_by_query,
                fold_id=fold_id,
                partition=partition,
                caps=caps,
            )
            second = compute_partition_query_audit(
                positives,
                negatives,
                registry_by_query,
                fold_id=fold_id,
                partition=partition,
                caps=caps,
            )
            first_sha = canonical_rows_sha256(first)
            second_sha = canonical_rows_sha256(second)
            if first_sha != second_sha or first != second:
                raise ValueError(f"query audit is not reproducible: {fold_id}/{partition}")
            audit_rows.extend(first)
            fold_hashes[partition] = {
                "first_sha256": first_sha,
                "second_sha256": second_sha,
                "identical": True,
            }
            fold_files[partition] = {
                "positive": {
                    "path": positive_path.relative_to(project_dir).as_posix(),
                    "sha256": positive_sha,
                    "rows": len(positives),
                },
                "negative": {
                    "path": negative_path.relative_to(project_dir).as_posix(),
                    "sha256": negative_sha,
                    "rows": len(negatives),
                },
            }
        input_file_audit[fold_id] = fold_files
        reproducibility_hashes[fold_id] = fold_hashes

    audit_rows.sort(
        key=lambda row: (
            fold_ids.index(row["fold_id"]),
            partitions.index(row["partition"]),
            row["query_id"],
        )
    )
    simulations = build_simulations(audit_rows, fold_ids, partitions, caps)
    target = str(config["target_query"]).strip().upper()
    target_rows = [row for row in audit_rows if row["query_id"] == target]
    if len(target_rows) != len(fold_ids):
        raise ValueError("target query must occur exactly once per fold")

    output_config = config["outputs"]
    output_directory = project_dir / output_config["directory"]
    output_paths = {
        "audit_csv": output_directory / output_config["audit_csv"],
        "simulation_json": output_directory / output_config["simulation_json"],
        "report_md": output_directory / output_config["report_md"],
    }
    if not force and any(path.exists() for path in output_paths.values()):
        raise FileExistsError("refusing to overwrite query-balance audit outputs")
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_text = audit_csv_text(audit_rows, caps)
    output_paths["audit_csv"].write_text(csv_text, encoding="utf-8")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "tier": "development",
        "fold_ids": list(fold_ids),
        "partitions": list(partitions),
        "query_key": list(QUERY_COLUMNS),
        "interaction_key": list(INTERACTION_COLUMNS),
        "per_query_caps": list(caps),
        "candidate_definition": {
            "query_fixed": True,
            "tcr_source": "unique_positive_TCRs_from_current_fold_partition",
            "known_positive_exclusion": "complete_positive_registry",
            "duplicate_interactions_allowed": False,
        },
        "inputs": {
            name: {
                "path": input_paths[name].relative_to(project_dir).as_posix(),
                "sha256": inputs[name]["sha256"],
            }
            for name in input_paths
        },
        "positive_registry_interaction_count": registry_size,
        "input_file_audit": input_file_audit,
        "query_audit_row_count": len(audit_rows),
        "query_audit_csv": {
            "path": output_paths["audit_csv"].relative_to(project_dir).as_posix(),
            "sha256": file_sha256(output_paths["audit_csv"]),
        },
        "target_query": target,
        "target_query_occurrences": target_rows,
        "simulations": simulations,
        "reproducibility": {
            "calculation_rebuild_count": 2,
            "all_partition_sha256_identical": True,
            "partition_sha256": reproducibility_hashes,
            "complete_rows_sha256": canonical_rows_sha256(audit_rows),
        },
        "invariants": {
            "source_csvs_modified": False,
            "negative_samples_resampled": False,
            "balanced_csvs_written": False,
            "models_trained": False,
        },
    }
    output_paths["simulation_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_paths["report_md"].write_text(render_report(summary), encoding="utf-8")
    return summary
