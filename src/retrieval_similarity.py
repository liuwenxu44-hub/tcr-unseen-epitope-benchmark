from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .retrieval import (
    METRIC_NAMES,
    CandidatePool,
    build_candidate_pool,
    deterministic_random_scores,
    file_sha256,
    macro_metric_delta,
    rank_labels,
    ranking_metrics,
    validate_reference_candidate_pool,
)


PER_QUERY_COLUMNS = (
    "query_id",
    "epitope",
    "hla",
    "status",
    "candidate_count",
    "known_positive_count",
    "selected_training_epitopes",
    "training_reference_tcr_count",
    *(f"similarity_{name}" for name in METRIC_NAMES),
    *(f"random_{name}" for name in METRIC_NAMES),
    *(f"cnn_n1_{name}" for name in METRIC_NAMES),
    *(f"cnn_n2_{name}" for name in METRIC_NAMES),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _resolve(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def _relative(project_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(project_dir.resolve()).as_posix()


def _verified_rows(
    project_dir: Path, source: Mapping[str, Any], description: str
) -> tuple[Path, list[dict[str, str]]]:
    path = _resolve(project_dir, str(source["path"]))
    observed_sha = file_sha256(path)
    if observed_sha != source["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {description}: expected {source['sha256']}, "
            f"observed {observed_sha}"
        )
    rows = _read_csv(path)
    if len(rows) != int(source["rows"]):
        raise ValueError(f"row count mismatch for {description}")
    return path, rows


def normalized_levenshtein_similarity(left: str, right: str) -> float:
    return float(Levenshtein.normalized_similarity(left, right))


def build_training_references(
    train_positive_rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, tuple[str, ...]]]:
    references: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in train_positive_rows:
        references[row["hla"].strip().upper()][row["epitope"].strip().upper()].add(
            row["tcr_cdr3_beta"].strip().upper()
        )
    return {
        hla: {
            epitope: tuple(sorted(tcrs))
            for epitope, tcrs in sorted(epitopes.items())
        }
        for hla, epitopes in sorted(references.items())
    }


def select_training_epitopes(
    query_epitope: str,
    hla_references: Mapping[str, Sequence[str]],
    top_k: int,
) -> list[tuple[str, float]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ranked = sorted(
        (
            (epitope, normalized_levenshtein_similarity(query_epitope, epitope))
            for epitope in hla_references
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[:top_k]


def max_tcr_similarities(
    candidate_tcrs: Sequence[str],
    reference_tcrs: Sequence[str],
    *,
    candidate_chunk_size: int,
    workers: int,
) -> np.ndarray:
    if not reference_tcrs:
        raise ValueError("reference TCRs must be non-empty")
    if candidate_chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive")
    maxima = np.empty(len(candidate_tcrs), dtype=np.float32)
    for start in range(0, len(candidate_tcrs), candidate_chunk_size):
        stop = min(start + candidate_chunk_size, len(candidate_tcrs))
        similarities = process.cdist(
            candidate_tcrs[start:stop],
            reference_tcrs,
            scorer=Levenshtein.normalized_similarity,
            dtype=np.float32,
            workers=workers,
        )
        maxima[start:stop] = similarities.max(axis=1)
    return maxima


def score_similarity_query(
    candidate_tcrs: Sequence[str],
    selected_epitopes: Sequence[tuple[str, float]],
    hla_references: Mapping[str, Sequence[str]],
    tcr_similarity_cache: Mapping[str, np.ndarray],
) -> list[float]:
    if not selected_epitopes:
        raise ValueError("selected training epitopes must be non-empty")
    scores = np.zeros(len(candidate_tcrs), dtype=np.float32)
    for epitope, epitope_similarity in selected_epitopes:
        if epitope not in hla_references or epitope not in tcr_similarity_cache:
            raise ValueError(f"missing training reference for epitope {epitope}")
        scores = np.maximum(
            scores, np.float32(epitope_similarity) * tcr_similarity_cache[epitope]
        )
    return [float(value) for value in scores]


def canonical_result_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(rows), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _macro(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, float]:
    if not rows:
        raise ValueError("macro-average requires at least one evaluable query")
    return {
        name: sum(float(row[f"{prefix}_{name}"]) for row in rows) / len(rows)
        for name in METRIC_NAMES
    }


def _validate_reference_rows(
    pool: CandidatePool,
    n1_rows: Sequence[Mapping[str, str]],
    n2_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Mapping[str, str]], dict[str, Mapping[str, str]]]:
    expected_ids = [query.query_id for query in pool.queries]
    for name, rows in (("CNN N1", n1_rows), ("CNN N2", n2_rows)):
        if [row["query_id"] for row in rows] != expected_ids:
            raise ValueError(f"{name} per-query ordering differs from candidate pool")
        if any(int(row["candidate_count"]) != len(pool.tcrs) for row in rows):
            raise ValueError(f"{name} candidate count differs from candidate pool")
    return (
        {row["query_id"]: row for row in n1_rows},
        {row["query_id"]: row for row in n2_rows},
    )


def _render_report(summary: Mapping[str, Any]) -> str:
    macros = summary["macro_average_evaluable_queries"]
    deltas = summary["macro_delta"]
    metric_headers = "| Comparison | Recall@10 | Recall@50 | Precision@10 | MRR | NDCG@10 |"
    separator = "|---|---:|---:|---:|---:|---:|"

    def metric_row(values: Mapping[str, float], *, signed: bool = False) -> str:
        fmt = "+.6f" if signed else ".6f"
        return "| " + " | ".join(format(values[name], fmt) for name in METRIC_NAMES) + " |"

    lines = [
        "# Development unseen_epitope sequence-similarity retrieval baseline",
        "",
        "## Candidate pool",
        "",
        f"- Queries: {summary['query_count']:,}",
        f"- Evaluable queries: {summary['available_query_count']:,}",
        f"- Unavailable queries: {summary['unavailable_query_count']:,}",
        f"- Unique candidate TCRs per query: {summary['candidate_tcr_count']:,}",
        f"- Candidate pool SHA-256: `{summary['candidate_pool_sha256']}`",
        "- Candidate order and positive labels are inherited from the N1 retrieval manifest.",
        "- Similarity references use unseen_epitope training positives only.",
        "- Queries without same-HLA training positives are unavailable; there is no HLA fallback.",
        "",
        "## Macro-average on evaluable queries",
        "",
        "| Ranking | Recall@10 | Recall@50 | Precision@10 | MRR | NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
        "| Sequence similarity " + metric_row(macros["sequence_similarity"]),
        "| Random " + metric_row(macros["random_baseline"]),
        "| CNN N1 " + metric_row(macros["cnn_n1"]),
        "| CNN N2 " + metric_row(macros["cnn_n2"]),
        "",
        "## Sequence-similarity differences",
        "",
        metric_headers,
        separator,
        "| vs random " + metric_row(deltas["vs_random"], signed=True),
        "| vs CNN N1 " + metric_row(deltas["vs_cnn_n1"], signed=True),
        "| vs CNN N2 " + metric_row(deltas["vs_cnn_n2"], signed=True),
        "",
        "Differences are sequence similarity minus the named baseline.",
        "",
        "This report is generated by `scripts/evaluate_similarity_retrieval.py`.",
        "",
    ]
    return "\n".join(lines)


def _write_outputs(
    output_dir: Path,
    output_names: Mapping[str, str],
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / output_names["per_query_csv"]).open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_QUERY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / output_names["summary_json"]).open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (output_dir / output_names["report_md"]).write_text(
        _render_report(summary), encoding="utf-8"
    )


def evaluate_similarity_retrieval(
    config_path: Path, project_dir: Path, *, force: bool = False
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    seed = int(config["seed"])
    experiment = config["experiment"]
    if experiment != {
        "tier": "development",
        "positive_split_strategy": "unseen_epitope",
        "method": "sequence_similarity",
    }:
        raise ValueError("similarity retrieval config targets the wrong experiment")

    pool_config = config["candidate_pool"]
    manifest_path = _resolve(project_dir, pool_config["manifest"])
    manifest = _load_json(manifest_path)
    expected_sha = str(pool_config["expected_sha256"])
    if manifest.get("candidate_pool_sha256") != expected_sha:
        raise ValueError("candidate pool manifest SHA-256 value mismatch")
    if manifest.get("query_count") != int(pool_config["expected_query_count"]):
        raise ValueError("candidate pool manifest query count mismatch")
    if manifest.get("candidate_tcr_count") != int(
        pool_config["expected_candidate_tcr_count"]
    ):
        raise ValueError("candidate pool manifest TCR count mismatch")
    if manifest.get("seed") != seed:
        raise ValueError("candidate pool manifest seed mismatch")

    positive_manifest_path = _resolve(
        project_dir, config["data"]["positive_split_manifest"]
    )
    positive_manifest = _load_json(positive_manifest_path)
    if positive_manifest.get("seed") != seed:
        raise ValueError("positive split manifest seed mismatch")

    sources = manifest["sources"]["sources"]
    train_path, train_positive_rows = _verified_rows(
        project_dir, sources["train_positive"], "training positives"
    )
    train_split_entry = positive_manifest["tiers"]["development"][
        "unseen_epitope"
    ]["splits"]["train"]
    if (
        train_path.name != train_split_entry["file"]
        or sources["train_positive"]["sha256"] != train_split_entry["sha256"]
        or len(train_positive_rows) != int(train_split_entry["positive_pairs"])
    ):
        raise ValueError("training reference does not match the positive train split")
    test_positive_path, test_positive_rows = _verified_rows(
        project_dir, sources["test_positive"], "test positives"
    )
    test_negative_path, test_negative_rows = _verified_rows(
        project_dir, sources["test_negative"], "N1 test candidate source"
    )
    registry_source = manifest["sources"]["positive_registry"]
    registry_path = _resolve(project_dir, registry_source["path"])
    if file_sha256(registry_path) != registry_source["sha256"]:
        raise ValueError("complete positive registry SHA-256 mismatch")
    registry_rows = _read_csv(registry_path)
    if len(registry_rows) != int(registry_source["interaction_count"]):
        raise ValueError("complete positive registry row count mismatch")

    pool = build_candidate_pool(
        train_rows=train_positive_rows,
        test_positive_rows=test_positive_rows,
        test_negative_rows=test_negative_rows,
        registry_rows=registry_rows,
    )
    validate_reference_candidate_pool(pool, manifest)
    if pool.pool_sha256 != expected_sha:
        raise ValueError("rebuilt candidate pool SHA-256 mismatch")

    reference_config = config["references"]
    n1_summary_path = _resolve(project_dir, reference_config["cnn_n1_summary"])
    n2_summary_path = _resolve(project_dir, reference_config["cnn_n2_summary"])
    n1_per_query_path = _resolve(project_dir, reference_config["cnn_n1_per_query"])
    n2_per_query_path = _resolve(project_dir, reference_config["cnn_n2_per_query"])
    n1_summary = _load_json(n1_summary_path)
    n2_summary = _load_json(n2_summary_path)
    for name, summary in (("CNN N1", n1_summary), ("CNN N2", n2_summary)):
        if summary.get("candidate_pool_sha256") != pool.pool_sha256:
            raise ValueError(f"{name} candidate pool SHA-256 mismatch")
        if summary.get("query_count") != len(pool.queries):
            raise ValueError(f"{name} query count mismatch")
    n1_by_query, n2_by_query = _validate_reference_rows(
        pool, _read_csv(n1_per_query_path), _read_csv(n2_per_query_path)
    )

    training_references = build_training_references(train_positive_rows)
    training_peptides = {row["epitope"].strip().upper() for row in train_positive_rows}
    query_peptides = {query.epitope for query in pool.queries}
    leakage = sorted(training_peptides & query_peptides)
    if leakage:
        raise ValueError(f"query peptide leakage into training references: {leakage[:5]}")

    similarity_config = config["similarity"]
    top_k = int(similarity_config["top_k_epitopes"])
    chunk_size = int(similarity_config["candidate_chunk_size"])
    workers = int(similarity_config["workers"])
    selected_by_query: dict[str, list[tuple[str, float]]] = {}
    selected_keys: set[tuple[str, str]] = set()
    for query in pool.queries:
        hla_references = training_references.get(query.hla)
        if not hla_references:
            selected_by_query[query.query_id] = []
            continue
        selected = select_training_epitopes(query.epitope, hla_references, top_k)
        selected_by_query[query.query_id] = selected
        selected_keys.update((query.hla, epitope) for epitope, _ in selected)

    tcr_similarity_cache: dict[tuple[str, str], np.ndarray] = {}
    for hla, epitope in sorted(selected_keys):
        tcr_similarity_cache[(hla, epitope)] = max_tcr_similarities(
            pool.tcrs,
            training_references[hla][epitope],
            candidate_chunk_size=chunk_size,
            workers=workers,
        )

    rows: list[dict[str, Any]] = []
    evaluable_rows: list[dict[str, Any]] = []
    for query in pool.queries:
        labels = [
            int((tcr, query.epitope, query.hla) in pool.known_positive_registry)
            for tcr in pool.tcrs
        ]
        n1_row = n1_by_query[query.query_id]
        n2_row = n2_by_query[query.query_id]
        if int(n1_row["known_positive_count"]) != sum(labels):
            raise ValueError(f"N1 positive labels differ for {query.query_id}")
        if int(n2_row["known_positive_count"]) != sum(labels):
            raise ValueError(f"N2 positive labels differ for {query.query_id}")
        selected = selected_by_query[query.query_id]
        row: dict[str, Any] = {
            "query_id": query.query_id,
            "epitope": query.epitope,
            "hla": query.hla,
            "status": "available" if selected else "unavailable",
            "candidate_count": len(pool.tcrs),
            "known_positive_count": sum(labels),
            "selected_training_epitopes": json.dumps(
                [
                    {"epitope": epitope, "similarity": similarity}
                    for epitope, similarity in selected
                ],
                separators=(",", ":"),
                sort_keys=True,
            ),
            "training_reference_tcr_count": sum(
                len(training_references[query.hla][epitope])
                for epitope, _ in selected
            ),
        }
        if not selected:
            for prefix in ("similarity", "random", "cnn_n1", "cnn_n2"):
                row.update({f"{prefix}_{name}": "" for name in METRIC_NAMES})
            rows.append(row)
            continue

        scores = score_similarity_query(
            pool.tcrs,
            selected,
            training_references[query.hla],
            {
                epitope: tcr_similarity_cache[(query.hla, epitope)]
                for epitope, _ in selected
            },
        )
        similarity_metrics = ranking_metrics(rank_labels(pool.tcrs, labels, scores))
        random_metrics = ranking_metrics(
            rank_labels(
                pool.tcrs,
                labels,
                deterministic_random_scores(query, pool.tcrs, seed),
            )
        )
        row.update(
            {f"similarity_{name}": value for name, value in similarity_metrics.items()}
        )
        row.update({f"random_{name}": value for name, value in random_metrics.items()})
        row.update(
            {f"cnn_n1_{name}": float(n1_row[f"model_{name}"]) for name in METRIC_NAMES}
        )
        row.update(
            {f"cnn_n2_{name}": float(n2_row[f"model_{name}"]) for name in METRIC_NAMES}
        )
        rows.append(row)
        evaluable_rows.append(row)

    macros = {
        "sequence_similarity": _macro(evaluable_rows, "similarity"),
        "random_baseline": _macro(evaluable_rows, "random"),
        "cnn_n1": _macro(evaluable_rows, "cnn_n1"),
        "cnn_n2": _macro(evaluable_rows, "cnn_n2"),
    }
    deltas = {
        "vs_random": macro_metric_delta(
            macros["sequence_similarity"], macros["random_baseline"]
        ),
        "vs_cnn_n1": macro_metric_delta(
            macros["sequence_similarity"], macros["cnn_n1"]
        ),
        "vs_cnn_n2": macro_metric_delta(
            macros["sequence_similarity"], macros["cnn_n2"]
        ),
    }
    unavailable = [row["query_id"] for row in rows if row["status"] == "unavailable"]
    summary = {
        "schema_version": 1,
        "seed": seed,
        "experiment": experiment,
        "query_count": len(pool.queries),
        "available_query_count": len(evaluable_rows),
        "unavailable_query_count": len(unavailable),
        "unavailable_query_ids": unavailable,
        "candidate_tcr_count": len(pool.tcrs),
        "total_candidates": pool.total_candidates,
        "candidate_pool_sha256": pool.pool_sha256,
        "similarity_definition": similarity_config["definition"],
        "top_k_training_epitopes": top_k,
        "same_hla_only": True,
        "cross_hla_fallback": False,
        "ranking_tie_break": "TCR string ascending",
        "macro_average_evaluable_queries": macros,
        "macro_delta": deltas,
        "per_query_result_sha256": canonical_result_sha256(rows),
        "input_audit": {
            "candidate_pool_manifest": {
                "path": _relative(project_dir, manifest_path),
                "sha256": file_sha256(manifest_path),
            },
            "positive_split_manifest": {
                "path": _relative(project_dir, positive_manifest_path),
                "sha256": file_sha256(positive_manifest_path),
            },
            "training_reference": {
                "path": _relative(project_dir, train_path),
                "sha256": file_sha256(train_path),
                "rows": len(train_positive_rows),
                "partition": "train",
                "label": 1,
            },
            "test_positive_candidate_source": {
                "path": _relative(project_dir, test_positive_path),
                "sha256": file_sha256(test_positive_path),
            },
            "test_negative_candidate_source": {
                "path": _relative(project_dir, test_negative_path),
                "sha256": file_sha256(test_negative_path),
            },
            "positive_registry": {
                "path": _relative(project_dir, registry_path),
                "sha256": file_sha256(registry_path),
            },
            "cnn_n1_summary": {
                "path": _relative(project_dir, n1_summary_path),
                "sha256": file_sha256(n1_summary_path),
            },
            "cnn_n1_per_query": {
                "path": _relative(project_dir, n1_per_query_path),
                "sha256": file_sha256(n1_per_query_path),
            },
            "cnn_n2_summary": {
                "path": _relative(project_dir, n2_summary_path),
                "sha256": file_sha256(n2_summary_path),
            },
            "cnn_n2_per_query": {
                "path": _relative(project_dir, n2_per_query_path),
                "sha256": file_sha256(n2_per_query_path),
            },
            "query_train_peptide_overlap_count": len(leakage),
            "validation_or_test_reference_row_count": 0,
        },
    }

    output_dir = _resolve(project_dir, config["outputs"]["directory"])
    output_names = {
        key: value for key, value in config["outputs"].items() if key != "directory"
    }
    output_paths = [output_dir / value for value in output_names.values()]
    if not force and any(path.exists() for path in output_paths):
        raise FileExistsError("refusing to overwrite similarity retrieval outputs")
    _write_outputs(output_dir, output_names, summary, rows)
    return summary
