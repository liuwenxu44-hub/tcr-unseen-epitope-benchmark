from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .data import HlaVocab
from .model_cnn import TwoTowerCnn
from .vocab import AminoAcidVocab


PARTITIONS = ("train", "validation", "test")
METRIC_NAMES = ("recall_at_10", "recall_at_50", "precision_at_10", "mrr", "ndcg_at_10")
PER_QUERY_COLUMNS = (
    "query_id",
    "epitope",
    "hla",
    "candidate_count",
    "known_positive_count",
    *(f"model_{name}" for name in METRIC_NAMES),
    *(f"random_{name}" for name in METRIC_NAMES),
)


@dataclass(frozen=True, order=True)
class Query:
    epitope: str
    hla: str

    @property
    def query_id(self) -> str:
        return f"{self.epitope}|{self.hla}"


@dataclass(frozen=True)
class CandidatePool:
    queries: tuple[Query, ...]
    tcrs: tuple[str, ...]
    known_positive_registry: frozenset[tuple[str, str, str]]
    pool_sha256: str

    @property
    def total_candidates(self) -> int:
        return len(self.queries) * len(self.tcrs)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected: str, description: str) -> str:
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"SHA-256 mismatch for {description}: expected {expected}, observed {observed}"
        )
    return observed


def _verify_configured_sha256(
    path: Path,
    expected_hashes: Mapping[str, str] | None,
    key: str,
    description: str,
) -> str:
    observed = file_sha256(path)
    if expected_hashes is not None and key in expected_hashes:
        _verify_sha256(path, expected_hashes[key], description)
    return observed


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


def candidate_pool_sha256(queries: Sequence[Query], tcrs: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for query in sorted(queries):
        for tcr in sorted(tcrs):
            digest.update(
                f"{query.epitope}\t{query.hla}\t{tcr}\n".encode("utf-8")
            )
    return digest.hexdigest()


def build_candidate_pool(
    *,
    train_rows: Sequence[Mapping[str, str]],
    test_positive_rows: Sequence[Mapping[str, str]],
    test_negative_rows: Sequence[Mapping[str, str]],
    registry_rows: Sequence[Mapping[str, str]],
) -> CandidatePool:
    queries = tuple(
        sorted(
            {
                Query(row["epitope"].strip().upper(), row["hla"].strip().upper())
                for row in test_positive_rows
            }
        )
    )
    tcrs = tuple(
        sorted(
            {
                row["tcr_cdr3_beta"].strip().upper()
                for row in (*test_positive_rows, *test_negative_rows)
            }
        )
    )
    if not queries or not tcrs:
        raise ValueError("retrieval queries and candidate TCRs must be non-empty")
    train_peptides = {row["epitope"].strip().upper() for row in train_rows}
    query_peptides = {query.epitope for query in queries}
    overlap = query_peptides & train_peptides
    if overlap:
        raise ValueError(f"query peptide leakage into training set: {sorted(overlap)[:5]}")

    registry = frozenset(
        (
            row["tcr_cdr3_beta"].strip().upper(),
            row["epitope"].strip().upper(),
            row["hla"].strip().upper(),
        )
        for row in registry_rows
    )
    for query in queries:
        if not any((tcr, query.epitope, query.hla) in registry for tcr in tcrs):
            raise ValueError(f"query has no known positive candidate: {query.query_id}")
    return CandidatePool(
        queries=queries,
        tcrs=tcrs,
        known_positive_registry=registry,
        pool_sha256=candidate_pool_sha256(queries, tcrs),
    )


def validate_reference_candidate_pool(
    pool: CandidatePool, reference_manifest: Mapping[str, Any]
) -> None:
    expected_definitions = {
        "query_definition": "unique epitope-HLA pairs from unseen test positives",
        "candidate_definition": "all unique TCRs from unseen test positive and N1 rows",
        "label_definition": "complete positive registry membership",
        "ordering": "queries and TCRs sorted lexicographically",
    }
    for key, expected in expected_definitions.items():
        if reference_manifest.get(key) != expected:
            raise ValueError(f"reference candidate pool {key} mismatch")
    checks = {
        "query_count": (len(pool.queries), reference_manifest.get("query_count")),
        "candidate_tcr_count": (
            len(pool.tcrs),
            reference_manifest.get("candidate_tcr_count"),
        ),
        "total_candidates": (
            pool.total_candidates,
            reference_manifest.get("total_candidates"),
        ),
        "candidate_pool_sha256": (
            pool.pool_sha256,
            reference_manifest.get("candidate_pool_sha256"),
        ),
    }
    mismatches = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in checks.items()
        if observed != expected
    }
    if mismatches:
        raise ValueError(f"reference candidate pool mismatch: {mismatches}")


def macro_metric_delta(
    observed: Mapping[str, float], reference: Mapping[str, float]
) -> dict[str, float]:
    return {
        name: float(observed[name]) - float(reference[name]) for name in METRIC_NAMES
    }


def ranking_metrics(ranked_labels: Sequence[int]) -> dict[str, float]:
    labels = [int(label) for label in ranked_labels]
    relevant = sum(labels)
    if relevant <= 0:
        raise ValueError("ranking metrics require at least one relevant candidate")

    def recall_at(k: int) -> float:
        return sum(labels[:k]) / relevant

    precision_denominator = min(10, len(labels))
    reciprocal_rank = 1.0 / (labels.index(1) + 1)
    dcg = sum(
        label / math.log2(rank + 1)
        for rank, label in enumerate(labels[:10], start=1)
    )
    ideal_relevant = min(relevant, 10)
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1)
    )
    return {
        "recall_at_10": recall_at(10),
        "recall_at_50": recall_at(50),
        "precision_at_10": sum(labels[:10]) / precision_denominator,
        "mrr": reciprocal_rank,
        "ndcg_at_10": dcg / ideal_dcg,
    }


def rank_labels(
    tcrs: Sequence[str], labels: Sequence[int], scores: Sequence[float]
) -> list[int]:
    if len(tcrs) != len(labels) or len(labels) != len(scores):
        raise ValueError("TCRs, labels, and scores must have equal length")
    order = sorted(range(len(tcrs)), key=lambda index: (-scores[index], tcrs[index]))
    return [int(labels[index]) for index in order]


def deterministic_random_scores(
    query: Query, tcrs: Sequence[str], seed: int
) -> list[float]:
    scores = []
    for tcr in tcrs:
        payload = f"{seed}\0{query.epitope}\0{query.hla}\0{tcr}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        scores.append(value / 2**64)
    return scores


def _choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _encode_sequences(
    values: Sequence[str], vocab: AminoAcidVocab, max_length: int
) -> torch.Tensor:
    return torch.tensor(
        [vocab.encode(value, max_length).token_ids for value in values],
        dtype=torch.long,
    )


def score_query(
    *,
    model: TwoTowerCnn,
    query: Query,
    candidate_tcr_ids: torch.Tensor,
    candidate_count: int,
    amino_acid_vocab: AminoAcidVocab,
    hla_vocab: HlaVocab,
    epitope_max_length: int,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    epitope_ids = torch.tensor(
        amino_acid_vocab.encode(query.epitope, epitope_max_length).token_ids,
        dtype=torch.long,
    )
    hla_id = hla_vocab.encode(query.hla)
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, candidate_count, batch_size):
            stop = min(start + batch_size, candidate_count)
            current_size = stop - start
            logits = model(
                candidate_tcr_ids[start:stop].to(device, non_blocking=True),
                epitope_ids.unsqueeze(0)
                .expand(current_size, -1)
                .to(device, non_blocking=True),
                torch.full(
                    (current_size,), hla_id, dtype=torch.long, device=device
                ),
            )
            scores.extend(float(value) for value in logits.cpu().tolist())
    return scores


def _macro_average(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, float]:
    return {
        name: sum(float(row[f"{prefix}_{name}"]) for row in rows) / len(rows)
        for name in METRIC_NAMES
    }


def matched_macro_average(
    rows: Sequence[Mapping[str, Any]],
    prefix: str,
    query_ids: Sequence[str],
) -> dict[str, float]:
    indexed = {str(row["query_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate query IDs in matched macro input")
    missing = [query_id for query_id in query_ids if query_id not in indexed]
    if missing:
        raise ValueError(f"matched macro query IDs are missing: {missing[:5]}")
    matched_rows = [indexed[query_id] for query_id in query_ids]
    if not matched_rows:
        raise ValueError("matched macro query set must be non-empty")
    return _macro_average(matched_rows, prefix)


def _render_report(summary: Mapping[str, Any]) -> str:
    model = summary["macro_average"]["model"]
    random_baseline = summary["macro_average"]["random_baseline"]
    model_label = summary.get("model_label", "CNN")
    lines = [
        f"# Development unseen_epitope {model_label} retrieval evaluation",
        "",
        "## Candidate pool",
        "",
        f"- Queries: {summary['query_count']:,}",
        f"- Unique candidate TCRs per query: {summary['candidate_tcr_count']:,}",
        f"- Total query-candidate pairs: {summary['total_candidates']:,}",
        f"- Candidate pool SHA-256: `{summary['candidate_pool_sha256']}`",
        "- Query definition: unique `(epitope, HLA)` from the unseen test positives",
        "- Candidate definition: all unique TCRs from unseen test positive and N1 rows",
        "- Labels: complete positive registry membership; all others are putative negatives",
        "- Query peptide overlap with training: 0",
        "",
        "## Macro-average metrics",
        "",
        "| Ranking | Recall@10 | Recall@50 | Precision@10 | MRR | NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {model_label} | {model['recall_at_10']:.6f} | {model['recall_at_50']:.6f} | "
        f"{model['precision_at_10']:.6f} | {model['mrr']:.6f} | "
        f"{model['ndcg_at_10']:.6f} |",
        f"| Random (seed {summary['seed']}) | {random_baseline['recall_at_10']:.6f} | "
        f"{random_baseline['recall_at_50']:.6f} | "
        f"{random_baseline['precision_at_10']:.6f} | {random_baseline['mrr']:.6f} | "
        f"{random_baseline['ndcg_at_10']:.6f} |",
    ]
    if "macro_delta_vs_n1" in summary:
        delta = summary["macro_delta_vs_n1"]
        lines.extend(
            [
                "",
                f"## {model_label} minus CNN N1 macro-average",
                "",
                "| Recall@10 | Recall@50 | Precision@10 | MRR | NDCG@10 |",
                "|---:|---:|---:|---:|---:|",
                f"| {delta['recall_at_10']:+.6f} | {delta['recall_at_50']:+.6f} | "
                f"{delta['precision_at_10']:+.6f} | {delta['mrr']:+.6f} | "
                f"{delta['ndcg_at_10']:+.6f} |",
            ]
        )
    if "macro_delta_vs_n2" in summary:
        delta = summary["macro_delta_vs_n2"]
        lines.extend(
            [
                "",
                f"## {model_label} minus CNN N2 macro-average",
                "",
                "| Recall@10 | Recall@50 | Precision@10 | MRR | NDCG@10 |",
                "|---:|---:|---:|---:|---:|",
                f"| {delta['recall_at_10']:+.6f} | {delta['recall_at_50']:+.6f} | "
                f"{delta['precision_at_10']:+.6f} | {delta['mrr']:+.6f} | "
                f"{delta['ndcg_at_10']:+.6f} |",
            ]
        )
    if "macro_delta_vs_mixed" in summary:
        delta = summary["macro_delta_vs_mixed"]
        lines.extend(
            [
                "",
                f"## {model_label} minus mixed CNN macro-average",
                "",
                "| Recall@10 | Recall@50 | Precision@10 | MRR | NDCG@10 |",
                "|---:|---:|---:|---:|---:|",
                f"| {delta['recall_at_10']:+.6f} | {delta['recall_at_50']:+.6f} | "
                f"{delta['precision_at_10']:+.6f} | {delta['mrr']:+.6f} | "
                f"{delta['ndcg_at_10']:+.6f} |",
            ]
        )
    if "similarity_comparison" in summary:
        comparison = summary["similarity_comparison"]
        compared_model = comparison.get("model", comparison.get("mixed"))
        similarity = comparison["similarity"]
        delta = comparison.get(
            "model_minus_similarity", comparison.get("mixed_minus_similarity")
        )
        lines.extend(
            [
                "",
                f"## Matched similarity comparison ({comparison['query_count']} queries)",
                "",
                "| Ranking | Recall@10 | Recall@50 | Precision@10 | MRR | NDCG@10 |",
                "|---|---:|---:|---:|---:|---:|",
                f"| {model_label} | {compared_model['recall_at_10']:.6f} | "
                f"{compared_model['recall_at_50']:.6f} | "
                f"{compared_model['precision_at_10']:.6f} | "
                f"{compared_model['mrr']:.6f} | "
                f"{compared_model['ndcg_at_10']:.6f} |",
                f"| Similarity | {similarity['recall_at_10']:.6f} | "
                f"{similarity['recall_at_50']:.6f} | "
                f"{similarity['precision_at_10']:.6f} | {similarity['mrr']:.6f} | "
                f"{similarity['ndcg_at_10']:.6f} |",
                f"| {model_label} - similarity | {delta['recall_at_10']:+.6f} | "
                f"{delta['recall_at_50']:+.6f} | {delta['precision_at_10']:+.6f} | "
                f"{delta['mrr']:+.6f} | {delta['ndcg_at_10']:+.6f} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Checkpoint: `{summary['checkpoint']['path']}`",
            f"- Checkpoint SHA-256: `{summary['checkpoint']['sha256']}`",
            f"- Best epoch: {summary['checkpoint']['best_epoch']}",
            f"- Positive registry SHA-256: `{summary['input_audit']['positive_registry']['sha256']}`",
            "",
            "This report is generated by `scripts/evaluate_retrieval.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs(
    *,
    output_dir: Path,
    output_names: Mapping[str, str],
    summary: Mapping[str, Any],
    per_query_rows: list[dict[str, Any]],
    candidate_manifest: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / output_names["per_query_csv"]).open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_QUERY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_query_rows)
    with (output_dir / output_names["summary_json"]).open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if "candidate_pool_manifest" in output_names:
        with (output_dir / output_names["candidate_pool_manifest"]).open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(candidate_manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
    (output_dir / output_names["report_md"]).write_text(
        _render_report(summary), encoding="utf-8"
    )


def evaluate_retrieval(
    config_path: Path,
    project_dir: Path,
    *,
    device_override: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    experiment = config["experiment"]
    if (
        experiment.get("tier") != "development"
        or experiment.get("positive_split_strategy") != "unseen_epitope"
        or experiment.get("negative_strategy") not in ("N1", "N2", "MIXED")
    ):
        raise ValueError(
            "retrieval config must target Development unseen_epitope N1/N2/MIXED"
        )

    data_config = config["data"]
    positive_manifest_path = _resolve(
        project_dir, data_config["positive_split_manifest"]
    )
    negative_manifest_path = _resolve(
        project_dir, data_config["negative_sampling_manifest"]
    )
    positive_manifest = _load_json(positive_manifest_path)
    negative_manifest = _load_json(negative_manifest_path)
    if positive_manifest["seed"] != seed or negative_manifest["seed"] != seed:
        raise ValueError("data manifest seed mismatch")
    positive_manifest_sha = file_sha256(positive_manifest_path)
    if negative_manifest["positive_splits_manifest"]["sha256"] != positive_manifest_sha:
        raise ValueError("negative manifest references a different positive split manifest")

    tier = experiment["tier"]
    split_strategy = experiment["positive_split_strategy"]
    negative_strategy = experiment["negative_strategy"]
    positive_dir = _resolve(project_dir, data_config["positive_split_directory"])
    negative_dir = _resolve(project_dir, data_config["negative_directory"])
    mixed_manifest_path = None
    mixed_manifest = None
    mixed_negative_dir = None
    if negative_strategy == "MIXED":
        mixed_manifest_path = _resolve(
            project_dir, data_config["mixed_negative_manifest"]
        )
        mixed_manifest = _load_json(mixed_manifest_path)
        mixed_negative_dir = _resolve(
            project_dir, data_config["mixed_negative_directory"]
        )
        expected_mixed_experiment = {
            "tier": tier,
            "positive_split_strategy": split_strategy,
        }
        if (
            mixed_manifest.get("seed") != seed
            or mixed_manifest.get("experiment") != expected_mixed_experiment
            or mixed_manifest.get("candidate_negatives_regenerated") is not False
        ):
            raise ValueError("mixed negative manifest provenance mismatch")
        if (
            mixed_manifest["input_manifests"]["negative_sampling"]["sha256"]
            != file_sha256(negative_manifest_path)
        ):
            raise ValueError("mixed manifest references another negative manifest")
    source_audit: dict[str, Any] = {}
    source_rows: dict[str, list[dict[str, str]]] = {}

    def load_manifest_source(partition: str, kind: str) -> None:
        positive_entry = positive_manifest["tiers"][tier][split_strategy]["splits"][
            partition
        ]
        if negative_strategy == "MIXED":
            if mixed_manifest is None or mixed_negative_dir is None:
                raise ValueError("mixed negative inputs are unavailable")
            negative_entry = mixed_manifest["partitions"][partition]["output"]
            negative_source_dir = mixed_negative_dir
        else:
            negative_entry = negative_manifest["tiers"][tier][split_strategy][partition][
                "negatives"
            ][negative_strategy]
            negative_source_dir = negative_dir
        sources = {
            source_kind: (directory / entry["file"], entry)
            for source_kind, directory, entry in (
            ("positive", positive_dir, positive_entry),
            ("negative", negative_source_dir, negative_entry),
            )
        }
        path, entry = sources[kind]
        key = f"{partition}_{kind}"
        source_audit[key] = {
            "path": _relative(project_dir, path),
            "sha256": _verify_sha256(
                path, entry["sha256"], f"{partition} {kind} source"
            ),
        }
        source_rows[key] = _read_csv(path)
        expected_count = (
            entry["positive_pairs"]
            if kind == "positive"
            else entry.get("generated_negative_pairs", entry.get("rows"))
        )
        if len(source_rows[key]) != expected_count:
            raise ValueError(f"row count mismatch for {partition} {kind} source")
        source_audit[key]["rows"] = len(source_rows[key])

    load_manifest_source("train", "positive")
    load_manifest_source("train", "negative")
    reference_manifest = None
    reference_summary = None
    reference_per_query_rows: list[dict[str, str]] | None = None
    reference_audit = None
    if "candidate_pool" in config:
        reference_config = config["candidate_pool"]
        reference_manifest_path = _resolve(
            project_dir, reference_config["reference_manifest"]
        )
        reference_manifest = _load_json(reference_manifest_path)
        if reference_manifest.get("seed") != seed:
            raise ValueError("reference candidate pool seed mismatch")
        if "expected_sha256" in reference_config and (
            reference_manifest.get("candidate_pool_sha256")
            != reference_config["expected_sha256"]
            or reference_manifest.get("query_count")
            != int(reference_config["expected_query_count"])
            or reference_manifest.get("candidate_tcr_count")
            != int(reference_config["expected_candidate_tcr_count"])
        ):
            raise ValueError("reference candidate pool differs from fixed expectation")
        reference_sources = reference_manifest["sources"]["sources"]
        for kind in ("positive", "negative"):
            key = f"test_{kind}"
            source = reference_sources[key]
            path = _resolve(project_dir, source["path"])
            source_audit[key] = {
                "path": _relative(project_dir, path),
                "sha256": _verify_sha256(
                    path, source["sha256"], f"reference {key} source"
                ),
            }
            source_rows[key] = _read_csv(path)
            if len(source_rows[key]) != source["rows"]:
                raise ValueError(f"reference candidate source row count mismatch: {key}")
            source_audit[key]["rows"] = len(source_rows[key])
        reference_summary_path = _resolve(
            project_dir, reference_config["reference_summary"]
        )
        reference_per_query_path = _resolve(
            project_dir, reference_config["reference_per_query"]
        )
        expected_reference_hashes = reference_config.get("expected_file_sha256")
        _verify_configured_sha256(
            reference_manifest_path,
            expected_reference_hashes,
            "manifest",
            "reference candidate-pool manifest",
        )
        _verify_configured_sha256(
            reference_summary_path,
            expected_reference_hashes,
            "summary",
            "reference N1 retrieval summary",
        )
        _verify_configured_sha256(
            reference_per_query_path,
            expected_reference_hashes,
            "per_query",
            "reference N1 per-query metrics",
        )
        reference_summary = _load_json(reference_summary_path)
        reference_per_query_rows = _read_csv(reference_per_query_path)
        reference_audit = {
            "manifest_path": _relative(project_dir, reference_manifest_path),
            "manifest_sha256": file_sha256(reference_manifest_path),
            "summary_path": _relative(project_dir, reference_summary_path),
            "summary_sha256": file_sha256(reference_summary_path),
            "per_query_path": _relative(project_dir, reference_per_query_path),
            "per_query_sha256": file_sha256(reference_per_query_path),
        }
    else:
        load_manifest_source("test", "positive")
        load_manifest_source("test", "negative")

    registry_path = _resolve(project_dir, data_config["positive_registry"])
    registry_sha = _verify_sha256(
        registry_path,
        negative_manifest["positive_registry"]["sha256"],
        "complete positive registry",
    )
    registry_rows = _read_csv(registry_path)
    if mixed_manifest is not None and (
        mixed_manifest["input_manifests"]["positive_registry"]["sha256"]
        != registry_sha
    ):
        raise ValueError("mixed manifest positive registry mismatch")
    train_rows = source_rows["train_positive"] + source_rows["train_negative"]
    pool = build_candidate_pool(
        train_rows=train_rows,
        test_positive_rows=source_rows["test_positive"],
        test_negative_rows=source_rows["test_negative"],
        registry_rows=registry_rows,
    )
    rebuilt_hash = candidate_pool_sha256(pool.queries, pool.tcrs)
    if rebuilt_hash != pool.pool_sha256:
        raise ValueError("candidate pool SHA-256 is not reproducible")
    if reference_manifest is not None:
        validate_reference_candidate_pool(pool, reference_manifest)
        if reference_manifest["sources"]["positive_registry"]["sha256"] != registry_sha:
            raise ValueError("reference candidate pool positive registry mismatch")
        if reference_summary is None or reference_per_query_rows is None:
            raise ValueError("reference retrieval artifacts are incomplete")
        for name, observed in (
            ("query_count", len(pool.queries)),
            ("candidate_tcr_count", len(pool.tcrs)),
            ("total_candidates", pool.total_candidates),
            ("candidate_pool_sha256", pool.pool_sha256),
        ):
            if reference_summary.get(name) != observed:
                raise ValueError(f"reference retrieval summary {name} mismatch")
        expected_query_ids = [query.query_id for query in pool.queries]
        if [row["query_id"] for row in reference_per_query_rows] != expected_query_ids:
            raise ValueError("reference per-query ordering mismatch")
        if any(
            int(row["candidate_count"]) != len(pool.tcrs)
            for row in reference_per_query_rows
        ):
            raise ValueError("reference per-query candidate count mismatch")

    baseline_config_path = _resolve(project_dir, config["baseline"]["config"])
    metrics_path = _resolve(project_dir, config["baseline"]["metrics"])
    checkpoint_path = _resolve(project_dir, config["baseline"]["checkpoint"])
    expected_baseline_hashes = config["baseline"].get("expected_file_sha256")
    _verify_configured_sha256(
        baseline_config_path,
        expected_baseline_hashes,
        "config",
        "training config",
    )
    _verify_configured_sha256(
        metrics_path,
        expected_baseline_hashes,
        "metrics",
        "training metrics",
    )
    _verify_configured_sha256(
        checkpoint_path,
        expected_baseline_hashes,
        "checkpoint",
        "best checkpoint",
    )
    baseline_config = _load_yaml(baseline_config_path)
    if baseline_config["experiment"] != experiment or baseline_config["seed"] != seed:
        raise ValueError("baseline config does not match retrieval experiment")
    training_metrics = _load_json(metrics_path)
    if training_metrics["data_spec"] != experiment or training_metrics["seed"] != seed:
        raise ValueError("training metrics do not match retrieval experiment")

    requested_device = device_override or config["evaluation"]["device"]
    device = _choose_device(requested_device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint["seed"] != seed or checkpoint["data_spec"] != experiment:
        raise ValueError("checkpoint provenance mismatch")
    if checkpoint["best_epoch"] != training_metrics["best_epoch"]:
        raise ValueError("checkpoint best epoch mismatch")
    model = TwoTowerCnn(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    amino_acid_vocab = AminoAcidVocab()
    if checkpoint["amino_acid_vocab"]["token_to_id"] != amino_acid_vocab.token_to_id:
        raise ValueError("checkpoint amino-acid vocabulary mismatch")
    hla_vocab = HlaVocab(checkpoint["hla_to_id"])
    sequence_config = baseline_config["sequence"]
    candidate_tcr_ids = _encode_sequences(
        pool.tcrs,
        amino_acid_vocab,
        int(sequence_config["tcr_max_length"]),
    )

    per_query_rows: list[dict[str, Any]] = []
    batch_size = int(config["evaluation"]["batch_size"])
    for query_index, query in enumerate(pool.queries):
        labels = [
            int((tcr, query.epitope, query.hla) in pool.known_positive_registry)
            for tcr in pool.tcrs
        ]
        if reference_per_query_rows is not None and int(
            reference_per_query_rows[query_index]["known_positive_count"]
        ) != sum(labels):
            raise ValueError(f"reference positive labels mismatch: {query.query_id}")
        model_scores = score_query(
            model=model,
            query=query,
            candidate_tcr_ids=candidate_tcr_ids,
            candidate_count=len(pool.tcrs),
            amino_acid_vocab=amino_acid_vocab,
            hla_vocab=hla_vocab,
            epitope_max_length=int(sequence_config["epitope_max_length"]),
            batch_size=batch_size,
            device=device,
        )
        model_metrics = ranking_metrics(rank_labels(pool.tcrs, labels, model_scores))
        random_scores = deterministic_random_scores(query, pool.tcrs, seed)
        random_metrics = ranking_metrics(
            rank_labels(pool.tcrs, labels, random_scores)
        )
        per_query_rows.append(
            {
                "query_id": query.query_id,
                "epitope": query.epitope,
                "hla": query.hla,
                "candidate_count": len(pool.tcrs),
                "known_positive_count": sum(labels),
                **{f"model_{name}": value for name, value in model_metrics.items()},
                **{f"random_{name}": value for name, value in random_metrics.items()},
            }
        )

    model_macro = _macro_average(per_query_rows, "model")
    random_macro = _macro_average(per_query_rows, "random")
    delta_vs_n1 = None
    if reference_summary is not None:
        reference_random = reference_summary["macro_average"]["random_baseline"]
        if any(
            not math.isclose(
                random_macro[name], float(reference_random[name]), rel_tol=0, abs_tol=1e-15
            )
            for name in METRIC_NAMES
        ):
            raise ValueError("random baseline differs from N1 reference on the shared pool")
        delta_vs_n1 = macro_metric_delta(
            model_macro, reference_summary["macro_average"]["model"]
        )

    delta_vs_n2 = None
    delta_vs_mixed = None
    similarity_comparison = None
    comparison_audit = None
    if "comparisons" in config:
        comparison_config = config["comparisons"]
        n2_summary_path = _resolve(project_dir, comparison_config["cnn_n2_summary"])
        n2_per_query_path = _resolve(
            project_dir, comparison_config["cnn_n2_per_query"]
        )
        similarity_summary_path = _resolve(
            project_dir, comparison_config["similarity_summary"]
        )
        similarity_per_query_path = _resolve(
            project_dir, comparison_config["similarity_per_query"]
        )
        comparison_paths = {
            "cnn_n2_summary": n2_summary_path,
            "cnn_n2_per_query": n2_per_query_path,
            "similarity_summary": similarity_summary_path,
            "similarity_per_query": similarity_per_query_path,
        }
        mixed_summary_path = None
        mixed_per_query_path = None
        if "cnn_mixed_summary" in comparison_config:
            mixed_summary_path = _resolve(
                project_dir, comparison_config["cnn_mixed_summary"]
            )
            mixed_per_query_path = _resolve(
                project_dir, comparison_config["cnn_mixed_per_query"]
            )
            comparison_paths.update(
                {
                    "cnn_mixed_summary": mixed_summary_path,
                    "cnn_mixed_per_query": mixed_per_query_path,
                }
            )
        expected_comparison_hashes = comparison_config.get("expected_file_sha256")
        for key, path in comparison_paths.items():
            _verify_configured_sha256(
                path, expected_comparison_hashes, key, f"comparison {key}"
            )
        n2_summary = _load_json(n2_summary_path)
        similarity_summary = _load_json(similarity_summary_path)
        n2_rows = _read_csv(n2_per_query_path)
        similarity_rows = _read_csv(similarity_per_query_path)
        expected_query_ids = [query.query_id for query in pool.queries]
        model_comparisons = [("CNN N2", n2_summary, n2_rows)]
        mixed_summary = None
        mixed_rows = None
        if mixed_summary_path is not None and mixed_per_query_path is not None:
            mixed_summary = _load_json(mixed_summary_path)
            mixed_rows = _read_csv(mixed_per_query_path)
            model_comparisons.append(("CNN mixed", mixed_summary, mixed_rows))
        for name, comparison_summary, rows in (
            *model_comparisons,
            ("similarity", similarity_summary, similarity_rows),
        ):
            if comparison_summary.get("candidate_pool_sha256") != pool.pool_sha256:
                raise ValueError(f"{name} candidate pool SHA-256 mismatch")
            if comparison_summary.get("query_count") != len(pool.queries):
                raise ValueError(f"{name} query count mismatch")
            if [row["query_id"] for row in rows] != expected_query_ids:
                raise ValueError(f"{name} per-query ordering mismatch")
            if any(
                int(row["candidate_count"]) != len(pool.tcrs)
                or int(row["known_positive_count"])
                != int(per_query_rows[index]["known_positive_count"])
                for index, row in enumerate(rows)
            ):
                raise ValueError(f"{name} candidate count or positive labels mismatch")
        delta_vs_n2 = macro_metric_delta(
            model_macro, n2_summary["macro_average"]["model"]
        )
        for name, comparison_summary, rows in model_comparisons:
            per_query_macro = _macro_average(rows, "model")
            if any(
                not math.isclose(
                    per_query_macro[metric],
                    float(comparison_summary["macro_average"]["model"][metric]),
                    rel_tol=0,
                    abs_tol=1e-15,
                )
                for metric in METRIC_NAMES
            ):
                raise ValueError(f"{name} summary differs from its per-query metrics")
        if mixed_summary is not None:
            delta_vs_mixed = macro_metric_delta(
                model_macro, mixed_summary["macro_average"]["model"]
            )
        available_similarity_ids = [
            row["query_id"] for row in similarity_rows if row.get("status") == "available"
        ]
        if len(available_similarity_ids) != 51 or len(set(available_similarity_ids)) != 51:
            raise ValueError("similarity comparison must contain 51 available queries")
        model_matched = matched_macro_average(
            per_query_rows, "model", available_similarity_ids
        )
        similarity_matched = matched_macro_average(
            similarity_rows, "similarity", available_similarity_ids
        )
        expected_similarity_macro = similarity_summary[
            "macro_average_evaluable_queries"
        ]["sequence_similarity"]
        if any(
            not math.isclose(
                similarity_matched[name],
                float(expected_similarity_macro[name]),
                rel_tol=0,
                abs_tol=1e-15,
            )
            for name in METRIC_NAMES
        ):
            raise ValueError("similarity summary differs from its 51-query metrics")
        similarity_comparison = {
            "query_count": len(available_similarity_ids),
            "query_ids": available_similarity_ids,
            "model": model_matched,
            "similarity": similarity_matched,
            "model_minus_similarity": macro_metric_delta(
                model_matched, similarity_matched
            ),
        }
        comparison_audit = {
            key: {
                "path": _relative(project_dir, path),
                "sha256": file_sha256(path),
            }
            for key, path in comparison_paths.items()
        }

    input_audit = {
        "baseline_config": {
            "path": _relative(project_dir, baseline_config_path),
            "sha256": file_sha256(baseline_config_path),
        },
        "training_metrics": {
            "path": _relative(project_dir, metrics_path),
            "sha256": file_sha256(metrics_path),
        },
        "positive_split_manifest": {
            "path": _relative(project_dir, positive_manifest_path),
            "sha256": positive_manifest_sha,
        },
        "negative_sampling_manifest": {
            "path": _relative(project_dir, negative_manifest_path),
            "sha256": file_sha256(negative_manifest_path),
        },
        "positive_registry": {
            "path": _relative(project_dir, registry_path),
            "sha256": registry_sha,
            "interaction_count": len(pool.known_positive_registry),
        },
        "sources": source_audit,
        "query_train_peptide_overlap_count": 0,
    }
    if mixed_manifest_path is not None:
        input_audit["mixed_negative_manifest"] = {
            "path": _relative(project_dir, mixed_manifest_path),
            "sha256": file_sha256(mixed_manifest_path),
        }
    if reference_audit is not None:
        input_audit["reference_candidate_pool"] = reference_audit
    if comparison_audit is not None:
        input_audit["comparisons"] = comparison_audit
    summary = {
        "schema_version": 1,
        "seed": seed,
        "experiment": experiment,
        "model_label": config.get("model_label", "CNN"),
        "device": str(device),
        "query_count": len(pool.queries),
        "candidate_tcr_count": len(pool.tcrs),
        "total_candidates": pool.total_candidates,
        "candidate_pool_sha256": pool.pool_sha256,
        "macro_average": {
            "model": model_macro,
            "random_baseline": random_macro,
        },
        "checkpoint": {
            "path": _relative(project_dir, checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "best_epoch": checkpoint["best_epoch"],
        },
        "input_audit": input_audit,
    }
    if delta_vs_n1 is not None:
        summary["macro_delta_vs_n1"] = delta_vs_n1
    if delta_vs_n2 is not None:
        summary["macro_delta_vs_n2"] = delta_vs_n2
    if delta_vs_mixed is not None:
        summary["macro_delta_vs_mixed"] = delta_vs_mixed
    if similarity_comparison is not None:
        summary["similarity_comparison"] = similarity_comparison
    candidate_manifest = {
        "schema_version": 1,
        "seed": seed,
        "query_definition": "unique epitope-HLA pairs from unseen test positives",
        "candidate_definition": "all unique TCRs from unseen test positive and N1 rows",
        "label_definition": "complete positive registry membership",
        "ordering": "queries and TCRs sorted lexicographically",
        "query_count": len(pool.queries),
        "candidate_tcr_count": len(pool.tcrs),
        "total_candidates": pool.total_candidates,
        "candidate_pool_sha256": pool.pool_sha256,
        "rebuild_sha256": rebuilt_hash,
        "query_train_peptide_overlap_count": 0,
        "sources": input_audit,
    }
    output_dir = _resolve(project_dir, config["outputs"]["directory"])
    output_names = {
        key: value for key, value in config["outputs"].items() if key != "directory"
    }
    output_paths = [output_dir / name for name in output_names.values()]
    if not force and any(path.exists() for path in output_paths):
        raise FileExistsError("refusing to overwrite retrieval outputs")
    _write_outputs(
        output_dir=output_dir,
        output_names=output_names,
        summary=summary,
        per_query_rows=per_query_rows,
        candidate_manifest=candidate_manifest,
    )
    return summary
