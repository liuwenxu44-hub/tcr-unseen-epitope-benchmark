from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

from .negative_sampling import _read_positive_csv, load_positive_registry
from .vdjdb import file_sha256


PARTITIONS = ("train", "validation", "test")
FOLD_IDS = tuple(f"fold_{index}" for index in range(1, 6))
QueryKey = tuple[str, str]
PairKey = tuple[str, str, str]
NEGATIVE_COLUMNS = (
    "negative_id",
    "tcr_cdr3_beta",
    "epitope",
    "hla",
    "pmhc_id",
    "label",
    "negative_strategy",
    "tier",
    "positive_split_strategy",
    "fold_id",
    "partition",
    "tcr_positive_degree",
    "query_positive_degree",
    "matching_algorithm",
)


class ExactDegreeMatchError(RuntimeError):
    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        super().__init__(json.dumps(self.report, sort_keys=True))


def _normalize(value: Any) -> str:
    return str(value).strip().upper()


def _query_key(row: Mapping[str, Any]) -> QueryKey:
    return _normalize(row["epitope"]), _normalize(row["hla"])


def _pair_key(row: Mapping[str, Any]) -> PairKey:
    return _normalize(row["tcr_cdr3_beta"]), *_query_key(row)


def _query_id(query: QueryKey) -> str:
    return f"{query[0]}|{query[1]}"


def _negative_id(fold_id: str, partition: str, key: PairKey) -> str:
    payload = "\0".join(
        ("degree_matched_cap100", fold_id, partition, *key)
    ).encode("utf-8")
    return f"putative-negative:{hashlib.sha256(payload).hexdigest()[:20]}"


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=NEGATIVE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _residual_cut_diagnostic(
    capacity: csr_matrix,
    flow: csr_matrix,
    *,
    source: int,
    tcrs: Sequence[str],
    queries: Sequence[QueryKey],
    tcr_start: int,
    query_start: int,
) -> dict[str, Any]:
    residual = (capacity - flow).tocsr()
    residual.eliminate_zeros()
    reachable = set(
        int(value)
        for value in breadth_first_order(
            residual, source, directed=True, return_predecessors=False
        )
    )
    reachable_tcrs = [
        tcr for index, tcr in enumerate(tcrs) if tcr_start + index in reachable
    ]
    reachable_queries = [
        _query_id(query)
        for index, query in enumerate(queries)
        if query_start + index in reachable
    ]
    coo = capacity.tocoo()
    cut_capacity = int(
        sum(
            int(value)
            for row, column, value in zip(coo.row, coo.col, coo.data, strict=True)
            if int(row) in reachable and int(column) not in reachable
        )
    )
    return {
        "reachable_tcr_count": len(reachable_tcrs),
        "reachable_query_count": len(reachable_queries),
        "reachable_tcrs": reachable_tcrs,
        "reachable_queries": reachable_queries,
        "minimum_cut_capacity": cut_capacity,
    }


def solve_degree_matched_partition(
    positives: Sequence[Mapping[str, Any]],
    positive_registry: set[PairKey],
    *,
    fold_id: str,
    partition: str,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if fold_id not in FOLD_IDS or partition not in PARTITIONS:
        raise ValueError(f"invalid fold/partition: {fold_id}/{partition}")
    normalized = []
    positive_keys = []
    for source in positives:
        row = dict(source)
        row["tcr_cdr3_beta"] = _normalize(row["tcr_cdr3_beta"])
        row["epitope"] = _normalize(row["epitope"])
        row["hla"] = _normalize(row["hla"])
        if str(row["label"]) != "1":
            raise ValueError(f"non-positive source row in {fold_id}/{partition}")
        normalized.append(row)
        positive_keys.append(_pair_key(row))
    if not normalized or len(positive_keys) != len(set(positive_keys)):
        raise ValueError(f"empty or duplicate positive interactions: {fold_id}/{partition}")

    tcr_degree = Counter(row["tcr_cdr3_beta"] for row in normalized)
    query_degree = Counter(_query_key(row) for row in normalized)
    tcrs = sorted(tcr_degree)
    queries = sorted(query_degree)
    if seed is not None:
        if isinstance(seed, bool) or not 0 <= int(seed) < 2**64:
            raise ValueError("seed must be an integer in [0, 2**64)")
        rng = np.random.default_rng(int(seed))
        tcrs = [tcrs[index] for index in rng.permutation(len(tcrs))]
        queries = [queries[index] for index in rng.permutation(len(queries))]
    total_required = len(normalized)
    if sum(tcr_degree.values()) != total_required or sum(query_degree.values()) != total_required:
        raise ValueError("positive degree totals are inconsistent")

    source_node = 0
    tcr_start = 1
    query_start = tcr_start + len(tcrs)
    sink_node = query_start + len(queries)
    node_count = sink_node + 1
    graph_rows: list[int] = []
    graph_columns: list[int] = []
    graph_capacities: list[int] = []

    for index, tcr in enumerate(tcrs):
        graph_rows.append(source_node)
        graph_columns.append(tcr_start + index)
        graph_capacities.append(int(tcr_degree[tcr]))

    allowed_candidate_count = 0
    candidate_counts_by_tcr = Counter()
    candidate_counts_by_query = Counter()
    for tcr_index, tcr in enumerate(tcrs):
        for query_index, query in enumerate(queries):
            key = (tcr, query[0], query[1])
            if key in positive_registry:
                continue
            graph_rows.append(tcr_start + tcr_index)
            graph_columns.append(query_start + query_index)
            graph_capacities.append(1)
            allowed_candidate_count += 1
            candidate_counts_by_tcr[tcr] += 1
            candidate_counts_by_query[query] += 1

    for index, query in enumerate(queries):
        graph_rows.append(query_start + index)
        graph_columns.append(sink_node)
        graph_capacities.append(int(query_degree[query]))
    capacity = csr_matrix(
        (
            np.asarray(graph_capacities, dtype=np.int64),
            (
                np.asarray(graph_rows, dtype=np.int64),
                np.asarray(graph_columns, dtype=np.int64),
            ),
        ),
        shape=(node_count, node_count),
        dtype=np.int64,
    )
    capacity.sort_indices()
    result = maximum_flow(capacity, source_node, sink_node, method="dinic")
    flow_value = int(result.flow_value)
    flow_block = result.flow[
        tcr_start:query_start, query_start:sink_node
    ].tocoo()
    matched_tcr_degree = Counter()
    matched_query_degree = Counter()
    selected_keys = []
    for tcr_index, query_index, value in zip(
        flow_block.row, flow_block.col, flow_block.data, strict=True
    ):
        if int(value) <= 0:
            continue
        if int(value) != 1:
            raise ValueError("unit TCR-query edge carried non-unit flow")
        tcr = tcrs[int(tcr_index)]
        query = queries[int(query_index)]
        selected_keys.append((tcr, query[0], query[1]))
        matched_tcr_degree[tcr] += 1
        matched_query_degree[query] += 1

    if flow_value != total_required:
        failed_tcrs = [
            {
                "tcr_cdr3_beta": tcr,
                "required_degree": int(tcr_degree[tcr]),
                "matched_degree": int(matched_tcr_degree[tcr]),
                "deficit": int(tcr_degree[tcr] - matched_tcr_degree[tcr]),
                "legal_query_candidate_count": int(candidate_counts_by_tcr[tcr]),
            }
            for tcr in tcrs
            if matched_tcr_degree[tcr] != tcr_degree[tcr]
        ]
        failed_queries = [
            {
                "query_id": _query_id(query),
                "required_degree": int(query_degree[query]),
                "matched_degree": int(matched_query_degree[query]),
                "deficit": int(query_degree[query] - matched_query_degree[query]),
                "legal_tcr_candidate_count": int(candidate_counts_by_query[query]),
            }
            for query in queries
            if matched_query_degree[query] != query_degree[query]
        ]
        report = {
            "status": "infeasible",
            "fold_id": fold_id,
            "partition": partition,
            "required_edge_count": total_required,
            "maximum_matched_edge_count": flow_value,
            "maximum_matching_ratio": flow_value / total_required,
            "failed_tcrs": failed_tcrs,
            "failed_queries": failed_queries,
            "constraint_bottleneck": _residual_cut_diagnostic(
                capacity,
                result.flow,
                source=source_node,
                tcrs=tcrs,
                queries=queries,
                tcr_start=tcr_start,
                query_start=query_start,
            ),
        }
        if seed is not None:
            report["random_seed"] = int(seed)
        raise ExactDegreeMatchError(report)

    if len(selected_keys) != total_required or len(selected_keys) != len(set(selected_keys)):
        raise ValueError("exact flow did not produce unique required interactions")
    if set(selected_keys) & positive_registry:
        raise ValueError("degree-matched negatives intersect positive registry")
    if matched_tcr_degree != tcr_degree or matched_query_degree != query_degree:
        raise ValueError("exact flow does not reproduce both positive degree sequences")

    rows = []
    for key in sorted(selected_keys, key=lambda item: (item[1], item[2], item[0])):
        tcr, epitope, hla = key
        query = (epitope, hla)
        rows.append(
            {
                "negative_id": _negative_id(fold_id, partition, key),
                "tcr_cdr3_beta": tcr,
                "epitope": epitope,
                "hla": hla,
                "pmhc_id": _query_id(query),
                "label": 0,
                "negative_strategy": "degree_matched",
                "tier": "development",
                "positive_split_strategy": (
                    "distance_aware_5fold_rebalanced_query_balanced_cap100"
                ),
                "fold_id": fold_id,
                "partition": partition,
                "tcr_positive_degree": int(tcr_degree[tcr]),
                "query_positive_degree": int(query_degree[query]),
                "matching_algorithm": "bipartite_b_matching_max_flow_dinic",
            }
        )
    audit = {
        "status": "exact",
        "positive_count": total_required,
        "negative_count": len(rows),
        "matching_rate": flow_value / total_required,
        "maximum_flow_value": flow_value,
        "unique_tcr_count": len(tcrs),
        "unique_query_count": len(queries),
        "candidate_edge_count": allowed_candidate_count,
        "registry_rejected_candidate_count": len(tcrs) * len(queries)
        - allowed_candidate_count,
        "minimum_legal_queries_per_tcr": min(candidate_counts_by_tcr.values()),
        "minimum_legal_tcrs_per_query": min(candidate_counts_by_query.values()),
        "tcr_degree_exact": True,
        "query_degree_exact": True,
        "known_positive_registry_intersections": 0,
        "duplicate_negative_interactions": 0,
        "negative_tcrs_outside_partition": 0,
        "negative_queries_outside_partition": 0,
    }
    if seed is not None:
        audit["random_seed"] = int(seed)
        audit["randomized_node_order"] = True
    return rows, audit


def audit_degree_matched_partition(
    positives: Sequence[Mapping[str, Any]],
    negatives: Sequence[Mapping[str, Any]],
    positive_registry: set[PairKey],
) -> dict[str, Any]:
    positive_tcr_degree = Counter(_normalize(row["tcr_cdr3_beta"]) for row in positives)
    negative_tcr_degree = Counter(_normalize(row["tcr_cdr3_beta"]) for row in negatives)
    positive_query_degree = Counter(_query_key(row) for row in positives)
    negative_query_degree = Counter(_query_key(row) for row in negatives)
    negative_keys = [_pair_key(row) for row in negatives]
    if positive_tcr_degree != negative_tcr_degree:
        raise ValueError("TCR positive/negative degree sequences differ")
    if positive_query_degree != negative_query_degree:
        raise ValueError("query positive/negative degree sequences differ")
    if len(negative_keys) != len(set(negative_keys)):
        raise ValueError("degree-matched negatives contain duplicate interactions")
    intersections = set(negative_keys) & positive_registry
    if intersections:
        raise ValueError("degree-matched negatives intersect positive registry")
    if set(negative_tcr_degree) - set(positive_tcr_degree):
        raise ValueError("negative uses TCR outside partition")
    if set(negative_query_degree) - set(positive_query_degree):
        raise ValueError("negative uses query outside partition")
    return {
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "matching_rate": len(negatives) / len(positives),
        "tcr_degree_exact": True,
        "query_degree_exact": True,
        "known_positive_registry_intersections": 0,
        "duplicate_negative_interactions": 0,
        "negative_tcrs_outside_partition": 0,
        "negative_queries_outside_partition": 0,
    }


def _render_report(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Distance-aware five-fold degree-matched negatives",
        "",
        "Each partition was solved independently as an exact bipartite b-matching.",
        "",
        "| Fold | Partition | Positive | Negative | TCRs | Queries | Matching rate | Exact |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for fold_id in FOLD_IDS:
        for partition in PARTITIONS:
            entry = manifest["folds"][fold_id]["partitions"][partition]
            audit = entry["audit"]
            lines.append(
                f"| {fold_id} | {partition} | {audit['positive_count']} | "
                f"{audit['negative_count']} | {audit['unique_tcr_count']} | "
                f"{audit['unique_query_count']} | {audit['matching_rate']:.6f} | yes |"
            )
    lines.extend(
        [
            "",
            "## Invariants",
            "",
            "- Every TCR has identical positive and negative degree within its fold/partition.",
            "- Every (epitope, HLA) query has identical positive and negative degree.",
            "- No negative interaction occurs in the complete positive registry.",
            "- All interactions are unique and use only partition-local TCRs and queries.",
            "- Two complete deterministic solves produced identical CSV SHA-256 values.",
            "- Existing positives and N1 data were not modified; no model was trained.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("fold_ids") != list(FOLD_IDS):
        raise ValueError("degree-matched config requires fold_1 through fold_5")
    if config.get("partitions") != list(PARTITIONS):
        raise ValueError("degree-matched config requires train/validation/test")
    matching = config.get("matching", {})
    expected = {
        "problem": "bipartite_b_matching",
        "solver": "scipy.sparse.csgraph.maximum_flow",
        "method": "dinic",
        "source_tcr_capacity": "positive_tcr_degree",
        "tcr_query_capacity": 1,
        "query_sink_capacity": "positive_query_degree",
        "ordering": "lexicographic_tcr_then_query",
        "require_exact": True,
    }
    if matching != expected:
        raise ValueError("degree-matched solver configuration differs from fixed specification")


def build_degree_matched_negatives(
    config_path: Path, project_dir: Path
) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_dir = project_dir.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("degree-matched config must be a mapping")
    _validate_config(config)
    positive_manifest_path = project_dir / config["inputs"]["positive_manifest"]["path"]
    registry_path = project_dir / config["inputs"]["positive_registry"]["path"]
    for name, path in (
        ("positive_manifest", positive_manifest_path),
        ("positive_registry", registry_path),
    ):
        if file_sha256(path) != config["inputs"][name]["sha256"]:
            raise ValueError(f"{name} SHA-256 mismatch")
    source_manifest = json.loads(positive_manifest_path.read_text(encoding="utf-8"))
    if int(source_manifest.get("fold_count", -1)) != 5:
        raise ValueError("positive source manifest does not contain five folds")
    positive_registry = {
        (_normalize(tcr), _normalize(epitope), _normalize(hla))
        for tcr, epitope, hla in load_positive_registry(registry_path)
    }

    outputs = config["outputs"]
    negative_directory = project_dir / outputs["negative_directory"]
    manifest_path = project_dir / outputs["manifest_json"]
    report_path = project_dir / outputs["report_md"]
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
    all_outputs = [
        path for fold_paths in negative_paths.values() for path in fold_paths.values()
    ] + [manifest_path, report_path]
    if any(path.exists() for path in all_outputs):
        raise FileExistsError("refusing to overwrite degree-matched outputs")

    positives_by_fold: dict[str, dict[str, list[dict[str, str]]]] = {}
    positive_hashes_before = {}
    solved_rows = {}
    solved_audits = {}
    first_hashes = {}
    second_hashes = {}
    for fold_id in FOLD_IDS:
        positives_by_fold[fold_id] = {}
        solved_rows[fold_id] = {}
        solved_audits[fold_id] = {}
        first_hashes[fold_id] = {}
        second_hashes[fold_id] = {}
        for partition in PARTITIONS:
            source = source_manifest["folds"][fold_id]["partitions"][partition][
                "positive_output"
            ]
            positive_path = project_dir / source["path"]
            observed_sha = file_sha256(positive_path)
            if observed_sha != source["sha256"]:
                raise ValueError(
                    f"positive input SHA-256 mismatch: {fold_id}/{partition}"
                )
            positives = _read_positive_csv(positive_path)
            if len(positives) != int(source["rows"]):
                raise ValueError(f"positive row count mismatch: {fold_id}/{partition}")
            positives_by_fold[fold_id][partition] = positives
            positive_hashes_before[(fold_id, partition)] = observed_sha
            first_rows, first_audit = solve_degree_matched_partition(
                positives,
                positive_registry,
                fold_id=fold_id,
                partition=partition,
            )
            second_rows, second_audit = solve_degree_matched_partition(
                positives,
                positive_registry,
                fold_id=fold_id,
                partition=partition,
            )
            first_text = _csv_text(first_rows)
            second_text = _csv_text(second_rows)
            first_sha = _text_sha256(first_text)
            second_sha = _text_sha256(second_text)
            if (
                first_rows != second_rows
                or first_audit != second_audit
                or first_sha != second_sha
            ):
                raise ValueError(f"deterministic rebuild mismatch: {fold_id}/{partition}")
            audit = {
                **first_audit,
                **audit_degree_matched_partition(
                    positives, first_rows, positive_registry
                ),
            }
            solved_rows[fold_id][partition] = (first_rows, first_text)
            solved_audits[fold_id][partition] = audit
            first_hashes[fold_id][partition] = first_sha
            second_hashes[fold_id][partition] = second_sha

    if first_hashes != second_hashes:
        raise ValueError("complete deterministic rebuild SHA-256 mismatch")
    for fold_id in FOLD_IDS:
        positive_tcr_sets = {
            partition: {
                _normalize(row["tcr_cdr3_beta"])
                for row in positives_by_fold[fold_id][partition]
            }
            for partition in PARTITIONS
        }
        positive_query_sets = {
            partition: {_query_key(row) for row in positives_by_fold[fold_id][partition]}
            for partition in PARTITIONS
        }
        for partition in PARTITIONS:
            negatives = solved_rows[fold_id][partition][0]
            if {
                _normalize(row["tcr_cdr3_beta"]) for row in negatives
            } - positive_tcr_sets[partition]:
                raise ValueError(f"TCR partition leakage: {fold_id}/{partition}")
            if {_query_key(row) for row in negatives} - positive_query_sets[partition]:
                raise ValueError(f"query partition leakage: {fold_id}/{partition}")

    folds_manifest = {}
    for fold_id in FOLD_IDS:
        fold_manifest = {"partitions": {}, "invariants": {}}
        interaction_sets = {}
        for partition in PARTITIONS:
            positives = positives_by_fold[fold_id][partition]
            rows, text = solved_rows[fold_id][partition]
            path = negative_paths[fold_id][partition]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            observed_sha = file_sha256(path)
            if observed_sha != first_hashes[fold_id][partition]:
                raise ValueError(f"written CSV SHA-256 mismatch: {fold_id}/{partition}")
            source = source_manifest["folds"][fold_id]["partitions"][partition][
                "positive_output"
            ]
            fold_manifest["partitions"][partition] = {
                "positive_input": {
                    "path": source["path"],
                    "sha256": source["sha256"],
                    "rows": len(positives),
                },
                "negative_output": {
                    "path": path.relative_to(project_dir).as_posix(),
                    "sha256": observed_sha,
                    "rows": len(rows),
                },
                "audit": solved_audits[fold_id][partition],
                "exact_solution": True,
            }
            interaction_sets[partition] = {_pair_key(row) for row in rows}
        cross_partition = {}
        for index, left in enumerate(PARTITIONS):
            for right in PARTITIONS[index + 1 :]:
                cross_partition[f"{left}_vs_{right}"] = len(
                    interaction_sets[left] & interaction_sets[right]
                )
        if any(cross_partition.values()):
            raise ValueError(f"within-fold cross-partition duplicate in {fold_id}")
        fold_manifest["invariants"] = {
            "partition_interaction_intersections": cross_partition,
            "all_partitions_exact": True,
        }
        folds_manifest[fold_id] = fold_manifest

    positive_hashes_after = {
        key: file_sha256(
            project_dir
            / source_manifest["folds"][key[0]]["partitions"][key[1]][
                "positive_output"
            ]["path"]
        )
        for key in positive_hashes_before
    }
    if positive_hashes_after != positive_hashes_before:
        raise ValueError("positive source files changed during degree matching")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "tier": "development",
        "fold_count": 5,
        "query_key": config["query_key"],
        "interaction_key": config["interaction_key"],
        "matching": config["matching"],
        "config": {
            "path": config_path.relative_to(project_dir).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "inputs": {
            name: {
                "path": value["path"],
                "sha256": value["sha256"],
            }
            for name, value in config["inputs"].items()
        },
        "folds": folds_manifest,
        "reproducibility": {
            "complete_rebuild_count": 2,
            "sha256_identical": True,
            "first_rebuild_csv_sha256": first_hashes,
            "second_rebuild_csv_sha256": second_hashes,
        },
        "invariants": {
            "all_fold_partitions_have_exact_solution": True,
            "all_tcr_degrees_exact": True,
            "all_query_degrees_exact": True,
            "known_positive_registry_intersections": 0,
            "duplicate_negative_interactions": 0,
            "partition_leakage": 0,
            "source_positive_files_modified": False,
            "existing_n1_files_modified": False,
            "models_trained": False,
        },
        "artifacts": {
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
    report_path.write_text(_render_report(manifest), encoding="utf-8")
    return manifest
