from __future__ import annotations

import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import yaml

from .degree_matched_negatives import (
    FOLD_IDS,
    PARTITIONS,
    PairKey,
    _csv_text,
    _normalize,
    _pair_key,
    _query_key,
    _text_sha256,
    audit_degree_matched_partition,
    solve_degree_matched_partition,
)
from .negative_sampling import _read_positive_csv, load_positive_registry
from .vdjdb import file_sha256


RESAMPLE_IDS = tuple(f"resample_{index}" for index in range(1, 6))


def derive_partition_seed(base_seed: int, fold_id: str, partition: str) -> int:
    if fold_id not in FOLD_IDS or partition not in PARTITIONS:
        raise ValueError(f"invalid fold/partition: {fold_id}/{partition}")
    payload = f"{int(base_seed)}\0{fold_id}\0{partition}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def interaction_jaccard(left: set[PairKey], right: set[PairKey]) -> float:
    union = left | right
    if not union:
        raise ValueError("cannot compute Jaccard for two empty interaction sets")
    return len(left & right) / len(union)


def _read_negative_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input has no header: {path}")
        required = {"tcr_cdr3_beta", "epitope", "hla", "pmhc_id", "label"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        records = list(reader)

    seen: set[PairKey] = set()
    for record in records:
        key = _pair_key(record)
        if key in seen:
            raise ValueError(f"Duplicate negative interaction: {key}")
        seen.add(key)
        if record["label"] != "0":
            raise ValueError(f"Expected negative label, observed: {record['label']}")
        if record["pmhc_id"] != f"{record['epitope']}|{record['hla']}":
            raise ValueError(f"Invalid pMHC id for {key}")
    return records


def _resample_negative_id(
    resample_id: str, fold_id: str, partition: str, key: PairKey
) -> str:
    payload = "\0".join(
        (
            "degree_matched_cap100_resample",
            resample_id,
            fold_id,
            partition,
            *key,
        )
    )
    return f"putative-negative:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _tag_resample_rows(
    rows: Sequence[Mapping[str, Any]], resample_id: str
) -> list[dict[str, Any]]:
    tagged = []
    for source in rows:
        row = dict(source)
        key = _pair_key(row)
        row["negative_id"] = _resample_negative_id(
            resample_id, str(row["fold_id"]), str(row["partition"]), key
        )
        row["negative_strategy"] = "degree_matched_resample"
        row["matching_algorithm"] = (
            "seeded_node_order_bipartite_b_matching_max_flow_dinic"
        )
        tagged.append(row)
    return tagged


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("fold_ids") != list(FOLD_IDS):
        raise ValueError("resample config requires fold_1 through fold_5")
    if config.get("partitions") != list(PARTITIONS):
        raise ValueError("resample config requires train/validation/test")
    resampling = config.get("resampling", {})
    seeds = resampling.get("base_seeds")
    if (
        resampling.get("count") != 5
        or not isinstance(seeds, list)
        or len(seeds) != 5
        or len(set(seeds)) != 5
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("resampling requires five distinct fixed integer seeds")
    if resampling.get("seed_derivation") != "sha256(base_seed,fold_id,partition)":
        raise ValueError("unexpected partition seed derivation")
    expected_matching = {
        "problem": "bipartite_b_matching",
        "solver": "scipy.sparse.csgraph.maximum_flow",
        "method": "dinic",
        "source_tcr_capacity": "positive_tcr_degree",
        "tcr_query_capacity": 1,
        "query_sink_capacity": "positive_query_degree",
        "ordering": "seeded_tcr_and_query_node_permutation",
        "require_exact": True,
    }
    if config.get("matching") != expected_matching:
        raise ValueError("resample matching specification differs from fixed constraints")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize empty values")
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": mean(values),
    }


def _render_report(manifest: Mapping[str, Any]) -> str:
    global_statistics = manifest["statistics"]
    lines = [
        "# Degree-matched negative independent resamples",
        "",
        "Five fixed seeds generated exact independently ordered max-flow solutions for every fold and partition.",
        "",
        "| Fold | Partition | Positive | Exact solutions | Pairwise Jaccard min | max | mean | Original Jaccard mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fold_id in FOLD_IDS:
        for partition in PARTITIONS:
            entry = manifest["folds"][fold_id]["partitions"][partition]
            pairwise = entry["resample_pairwise_jaccard_summary"]
            original = entry["original_jaccard_summary"]
            lines.append(
                f"| {fold_id} | {partition} | {entry['positive_input']['rows']} | "
                f"{entry['exact_solution_count']}/5 | {pairwise['minimum']:.6f} | "
                f"{pairwise['maximum']:.6f} | {pairwise['mean']:.6f} | "
                f"{original['mean']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Global summary",
            "",
            f"- Exact solutions: {global_statistics['exact_solution_count']}/{global_statistics['requested_solution_count']} ({global_statistics['exact_solution_success_rate']:.6f})",
            f"- Resample-pair Jaccard: {global_statistics['resample_pairwise_jaccard']['minimum']:.6f} to {global_statistics['resample_pairwise_jaccard']['maximum']:.6f}, mean {global_statistics['resample_pairwise_jaccard']['mean']:.6f}",
            f"- Resample-to-original Jaccard: {global_statistics['original_jaccard']['minimum']:.6f} to {global_statistics['original_jaccard']['maximum']:.6f}, mean {global_statistics['original_jaccard']['mean']:.6f}",
            "",
            "## Invariants",
            "",
            "- Every TCR and every query has identical positive and negative degree in all 75 outputs.",
            "- No negative intersects the complete positive registry; all interactions are unique and partition-local.",
            "- Two solves with each derived seed produced identical rows and SHA-256 values.",
            "- Every pair of distinct seeds produced a different interaction set for every fold/partition.",
            "- Positive and original degree-matched files were not modified; no model was trained or evaluated.",
            "",
        ]
    )
    return "\n".join(lines)


def build_degree_matched_resamples(
    config_path: Path, project_dir: Path
) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_dir = project_dir.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("degree-matched resample config must be a mapping")
    _validate_config(config)

    input_paths = {
        name: project_dir / source["path"] for name, source in config["inputs"].items()
    }
    for name, path in input_paths.items():
        if file_sha256(path) != config["inputs"][name]["sha256"]:
            raise ValueError(f"{name} SHA-256 mismatch")
    positive_manifest = _read_json(input_paths["positive_manifest"])
    original_manifest = _read_json(input_paths["original_degree_matched_manifest"])
    positive_registry = {
        (_normalize(tcr), _normalize(epitope), _normalize(hla))
        for tcr, epitope, hla in load_positive_registry(
            input_paths["positive_registry"]
        )
    }

    outputs = config["outputs"]
    output_root = project_dir / outputs["root_directory"]
    manifest_path = project_dir / outputs["manifest_json"]
    report_path = project_dir / outputs["report_md"]
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    if manifest_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite resample manifest or report")

    base_seeds = [int(value) for value in config["resampling"]["base_seeds"]]
    seed_by_resample = dict(zip(RESAMPLE_IDS, base_seeds, strict=True))
    output_paths = {
        resample_id: {
            fold_id: {
                partition: output_root
                / resample_id
                / outputs["negative_csv_pattern"].format(
                    fold=fold_id, partition=partition
                )
                for partition in PARTITIONS
            }
            for fold_id in FOLD_IDS
        }
        for resample_id in RESAMPLE_IDS
    }

    positives_by_fold: dict[str, dict[str, list[dict[str, str]]]] = {}
    positive_hashes_before = {}
    original_sets: dict[str, dict[str, set[PairKey]]] = {}
    original_hashes_before = {
        "manifest": file_sha256(input_paths["original_degree_matched_manifest"])
    }
    for fold_id in FOLD_IDS:
        positives_by_fold[fold_id] = {}
        original_sets[fold_id] = {}
        for partition in PARTITIONS:
            source = positive_manifest["folds"][fold_id]["partitions"][partition][
                "positive_output"
            ]
            positive_path = project_dir / source["path"]
            observed_sha = file_sha256(positive_path)
            if observed_sha != source["sha256"]:
                raise ValueError(f"positive input SHA-256 mismatch: {fold_id}/{partition}")
            positives = _read_positive_csv(positive_path)
            if len(positives) != int(source["rows"]):
                raise ValueError(f"positive row count mismatch: {fold_id}/{partition}")
            positives_by_fold[fold_id][partition] = positives
            positive_hashes_before[(fold_id, partition)] = observed_sha

            original_source = original_manifest["folds"][fold_id]["partitions"][
                partition
            ]["negative_output"]
            original_path = project_dir / original_source["path"]
            original_sha = file_sha256(original_path)
            if original_sha != original_source["sha256"]:
                raise ValueError(
                    f"original degree-matched SHA-256 mismatch: {fold_id}/{partition}"
                )
            original_rows = _read_negative_csv(original_path)
            if len(original_rows) != int(original_source["rows"]):
                raise ValueError(
                    f"original degree-matched row mismatch: {fold_id}/{partition}"
                )
            original_sets[fold_id][partition] = {
                _pair_key(row) for row in original_rows
            }
            original_hashes_before[f"{fold_id}:{partition}"] = original_sha

    solved: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        resample_id: {fold_id: {} for fold_id in FOLD_IDS}
        for resample_id in RESAMPLE_IDS
    }
    first_hashes = {resample_id: {} for resample_id in RESAMPLE_IDS}
    second_hashes = {resample_id: {} for resample_id in RESAMPLE_IDS}
    exact_solution_count = 0
    for resample_id in RESAMPLE_IDS:
        base_seed = seed_by_resample[resample_id]
        for fold_id in FOLD_IDS:
            first_hashes[resample_id][fold_id] = {}
            second_hashes[resample_id][fold_id] = {}
            for partition in PARTITIONS:
                derived_seed = derive_partition_seed(base_seed, fold_id, partition)
                positives = positives_by_fold[fold_id][partition]
                first_rows, first_solver_audit = solve_degree_matched_partition(
                    positives,
                    positive_registry,
                    fold_id=fold_id,
                    partition=partition,
                    seed=derived_seed,
                )
                second_rows, second_solver_audit = solve_degree_matched_partition(
                    positives,
                    positive_registry,
                    fold_id=fold_id,
                    partition=partition,
                    seed=derived_seed,
                )
                first_rows = _tag_resample_rows(first_rows, resample_id)
                second_rows = _tag_resample_rows(second_rows, resample_id)
                first_text = _csv_text(first_rows)
                second_text = _csv_text(second_rows)
                first_sha = _text_sha256(first_text)
                second_sha = _text_sha256(second_text)
                if (
                    first_rows != second_rows
                    or first_solver_audit != second_solver_audit
                    or first_sha != second_sha
                ):
                    raise ValueError(
                        f"same-seed rebuild mismatch: {resample_id}/{fold_id}/{partition}"
                    )
                independent_audit = audit_degree_matched_partition(
                    positives, first_rows, positive_registry
                )
                edge_set = {_pair_key(row) for row in first_rows}
                solved[resample_id][fold_id][partition] = {
                    "rows": first_rows,
                    "text": first_text,
                    "sha256": first_sha,
                    "derived_seed": derived_seed,
                    "audit": {**first_solver_audit, **independent_audit},
                    "edges": edge_set,
                }
                first_hashes[resample_id][fold_id][partition] = first_sha
                second_hashes[resample_id][fold_id][partition] = second_sha
                exact_solution_count += 1

    all_pairwise_jaccards = []
    all_original_jaccards = []
    partition_statistics: dict[str, dict[str, dict[str, Any]]] = {
        fold_id: {} for fold_id in FOLD_IDS
    }
    for fold_id in FOLD_IDS:
        for partition in PARTITIONS:
            pairwise = []
            for left, right in combinations(RESAMPLE_IDS, 2):
                left_edges = solved[left][fold_id][partition]["edges"]
                right_edges = solved[right][fold_id][partition]["edges"]
                jaccard = interaction_jaccard(left_edges, right_edges)
                if jaccard >= 1.0:
                    raise ValueError(
                        f"different seeds produced identical edges: "
                        f"{fold_id}/{partition}/{left}/{right}"
                    )
                pairwise.append({"left": left, "right": right, "jaccard": jaccard})
                all_pairwise_jaccards.append(jaccard)
            original_values = {}
            for resample_id in RESAMPLE_IDS:
                value = interaction_jaccard(
                    solved[resample_id][fold_id][partition]["edges"],
                    original_sets[fold_id][partition],
                )
                original_values[resample_id] = value
                all_original_jaccards.append(value)
            partition_statistics[fold_id][partition] = {
                "pairwise": pairwise,
                "pairwise_summary": _summary(
                    [item["jaccard"] for item in pairwise]
                ),
                "original": original_values,
                "original_summary": _summary(list(original_values.values())),
            }

    cross_partition_intersections = {}
    for resample_id in RESAMPLE_IDS:
        cross_partition_intersections[resample_id] = {}
        for fold_id in FOLD_IDS:
            intersections = {}
            for left, right in combinations(PARTITIONS, 2):
                intersections[f"{left}_vs_{right}"] = len(
                    solved[resample_id][fold_id][left]["edges"]
                    & solved[resample_id][fold_id][right]["edges"]
                )
            if any(intersections.values()):
                raise ValueError(
                    f"cross-partition interaction leakage: {resample_id}/{fold_id}"
                )
            cross_partition_intersections[resample_id][fold_id] = intersections

    positive_hashes_after = {
        key: file_sha256(
            project_dir
            / positive_manifest["folds"][key[0]]["partitions"][key[1]][
                "positive_output"
            ]["path"]
        )
        for key in positive_hashes_before
    }
    if positive_hashes_after != positive_hashes_before:
        raise ValueError("positive source files changed during resampling")
    original_hashes_after = {
        "manifest": file_sha256(input_paths["original_degree_matched_manifest"]),
        **{
            f"{fold_id}:{partition}": file_sha256(
                project_dir
                / original_manifest["folds"][fold_id]["partitions"][partition][
                    "negative_output"
                ]["path"]
            )
            for fold_id in FOLD_IDS
            for partition in PARTITIONS
        },
    }
    if original_hashes_after != original_hashes_before:
        raise ValueError("original degree-matched files changed during resampling")

    folds_manifest = {}
    for fold_id in FOLD_IDS:
        folds_manifest[fold_id] = {"partitions": {}}
        for partition in PARTITIONS:
            source = positive_manifest["folds"][fold_id]["partitions"][partition][
                "positive_output"
            ]
            original_source = original_manifest["folds"][fold_id]["partitions"][
                partition
            ]["negative_output"]
            stats = partition_statistics[fold_id][partition]
            resample_entries = {}
            for resample_id in RESAMPLE_IDS:
                item = solved[resample_id][fold_id][partition]
                path = output_paths[resample_id][fold_id][partition]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(item["text"], encoding="utf-8")
                if file_sha256(path) != item["sha256"]:
                    raise ValueError(
                        f"written SHA-256 mismatch: {resample_id}/{fold_id}/{partition}"
                    )
                resample_entries[resample_id] = {
                    "base_seed": seed_by_resample[resample_id],
                    "derived_seed": item["derived_seed"],
                    "negative_output": {
                        "path": path.relative_to(project_dir).as_posix(),
                        "sha256": item["sha256"],
                        "rows": len(item["rows"]),
                    },
                    "audit": item["audit"],
                    "exact_solution": True,
                    "jaccard_with_original": stats["original"][resample_id],
                }
            folds_manifest[fold_id]["partitions"][partition] = {
                "positive_input": {
                    "path": source["path"],
                    "sha256": source["sha256"],
                    "rows": int(source["rows"]),
                },
                "original_degree_matched_input": original_source,
                "resamples": resample_entries,
                "exact_solution_count": len(RESAMPLE_IDS),
                "resample_pairwise_jaccard": stats["pairwise"],
                "resample_pairwise_jaccard_summary": stats["pairwise_summary"],
                "original_jaccard_summary": stats["original_summary"],
            }

    requested_solution_count = len(RESAMPLE_IDS) * len(FOLD_IDS) * len(PARTITIONS)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "tier": "development",
        "fold_count": 5,
        "resample_count": 5,
        "query_key": config["query_key"],
        "interaction_key": config["interaction_key"],
        "matching": config["matching"],
        "resampling": {
            **config["resampling"],
            "ids": list(RESAMPLE_IDS),
            "seed_by_resample": seed_by_resample,
        },
        "config": {
            "path": config_path.relative_to(project_dir).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "inputs": config["inputs"],
        "folds": folds_manifest,
        "statistics": {
            "requested_solution_count": requested_solution_count,
            "exact_solution_count": exact_solution_count,
            "exact_solution_success_rate": exact_solution_count
            / requested_solution_count,
            "resample_pairwise_comparison_count": len(all_pairwise_jaccards),
            "resample_pairwise_jaccard": _summary(all_pairwise_jaccards),
            "original_comparison_count": len(all_original_jaccards),
            "original_jaccard": _summary(all_original_jaccards),
        },
        "reproducibility": {
            "same_seed_rebuild_count": 2,
            "same_seed_rows_and_sha256_identical": first_hashes == second_hashes,
            "first_rebuild_csv_sha256": first_hashes,
            "second_rebuild_csv_sha256": second_hashes,
            "different_seed_edge_sets_all_distinct": True,
            "all_pairwise_jaccards_below_one": max(all_pairwise_jaccards) < 1.0,
        },
        "cross_partition_intersections": cross_partition_intersections,
        "invariants": {
            "all_outputs_exact": exact_solution_count == requested_solution_count,
            "all_tcr_degrees_exact": True,
            "all_query_degrees_exact": True,
            "known_positive_registry_intersections": 0,
            "duplicate_negative_interactions": 0,
            "partition_leakage": 0,
            "source_positive_files_modified": False,
            "original_degree_matched_files_modified": False,
            "models_trained": False,
            "models_evaluated": False,
        },
        "artifacts": {
            "root_directory": outputs["root_directory"],
            "manifest_json": outputs["manifest_json"],
            "report_md": outputs["report_md"],
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(manifest), encoding="utf-8")
    return manifest
