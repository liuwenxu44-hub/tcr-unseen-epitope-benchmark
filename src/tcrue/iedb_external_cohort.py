from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .degree_matched_negatives import (
    audit_degree_matched_partition,
    solve_degree_matched_partition,
)
from .iedb_external_audit import (
    _maximum_similarity,
    _select_capped_records,
    _source_category,
    build_candidate_tiers,
    load_iedb_references,
    load_vdjdb_registry,
    parse_iedb_receptors,
)
from .vdjdb import file_sha256


RECORD_COLUMNS = (
    "record_id",
    "interaction_key",
    "tcr_cdr3_beta",
    "epitope",
    "hla",
    "query_id",
    "reference_id",
    "max_similarity_to_vdjdb",
    "tcr_seen_in_vdjdb",
    "hla_seen_in_vdjdb",
    "antigen_source",
    "source_database",
    "label",
)
ANNOTATION_COLUMNS = (
    "record_id",
    "interaction_key",
    "label",
    "fold_id",
    "tcr_seen_in_internal_fold_train",
    "hla_seen_in_internal_fold_train",
    "tcr_train_frequency",
)
DATA_OUTPUTS = {
    "positives": "data/external/iedb_strict_cap100/positives.csv",
    "negatives": "data/external/iedb_strict_cap100/negatives.csv",
    "test": "data/external/iedb_strict_cap100/test.csv",
    "manifest": "data/external/iedb_strict_cap100/manifest.json",
}
RESULT_OUTPUTS = {
    "report": "results/external/iedb/final_cohort_report.md",
    "internal_annotations": "results/external/iedb/internal_overlap_annotations.csv",
}
DEGREE_CONFIG_PATH = Path("configs/degree_matched_negatives.yaml")


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _pair_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["tcr_cdr3_beta"]),
        str(row["epitope"]),
        str(row["hla"]),
    )


def _query_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["epitope"]), str(row["hla"])


def _join(values: Sequence[str] | set[str]) -> str:
    return "|".join(sorted(value for value in values if value))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_frozen_config(config: Mapping[str, Any]) -> None:
    rule = config.get("final_inclusion_rule")
    required = {
        "tier": "C",
        "minimum_positive_records_per_query": 1,
        "query_cap_for_degree_matching": 100,
        "require_mapped_publication_reference": True,
        "publication_reference_overlap": 0,
        "development_peptide_overlap": 0,
        "maximum_development_peptide_similarity": 0.5,
        "exact_degree_matching_rate": 1.0,
        "status": "frozen",
    }
    if rule != required:
        raise ValueError(f"frozen IEDB cohort rule differs from required rule: {rule}")
    policy = config.get("reference_policy", {})
    if policy.get("require_mapped_publication_reference") is not True:
        raise ValueError("mapped PMID/DOI policy is not frozen")
    if policy.get("canonical_identifiers") != ["PMID", "DOI"]:
        raise ValueError("reference identifiers are not frozen to PMID/DOI")
    amendments = config.get("protocol_amendments", [])
    if len(amendments) != 1 or amendments[0].get("timing") != "pre-scoring":
        raise ValueError("cap100 pre-scoring amendment is absent")
    if amendments[0].get("external_model_performance_viewed") is not False:
        raise ValueError("external model performance was viewed before cohort freeze")


def _load_strict_source(
    project_dir: Path,
    cohort_config: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    set[tuple[str, str, str]],
    dict[str, float],
    dict[str, Any],
]:
    audit_config_path = project_dir / cohort_config["source_audit"]["config"]["path"]
    if file_sha256(audit_config_path) != cohort_config["source_audit"]["config"]["sha256"]:
        raise ValueError("source IEDB audit config hash mismatch")
    audit_config = yaml.safe_load(audit_config_path.read_text(encoding="utf-8"))
    receptor = audit_config["source"]["archives"]["receptor"]
    reference = audit_config["source"]["archives"]["reference"]
    for specification in (receptor, reference):
        path = project_dir / specification["path"]
        if path.stat().st_size != int(specification["size_bytes"]):
            raise ValueError(f"IEDB source archive size mismatch: {path}")
        if file_sha256(path) != specification["sha256"]:
            raise ValueError(f"IEDB source archive hash mismatch: {path}")
    references, _ = load_iedb_references(
        project_dir / reference["path"], reference["member"]
    )
    all_iedb, _, _ = parse_iedb_receptors(
        project_dir / receptor["path"],
        receptor["member"],
        references,
        audit_config["scope"],
    )
    frozen_vdjdb = audit_config["frozen_vdjdb"]
    registry_path = project_dir / frozen_vdjdb["positive_registry"]
    development_path = project_dir / frozen_vdjdb["development_registry"]
    if file_sha256(registry_path) != frozen_vdjdb["positive_registry_sha256"]:
        raise ValueError("VDJdb complete positive registry hash mismatch")
    if file_sha256(development_path) != frozen_vdjdb["development_registry_sha256"]:
        raise ValueError("VDJdb Development registry hash mismatch")
    vdjdb = load_vdjdb_registry(registry_path)
    development = load_vdjdb_registry(development_path)
    tiers, similarities = build_candidate_tiers(
        all_iedb, vdjdb, development, audit_config
    )
    positive_registry = set(vdjdb["pairs"]) | {
        _pair_key(row) for row in all_iedb
    }
    return (
        tiers["C"],
        vdjdb,
        development,
        positive_registry,
        similarities,
        audit_config,
    )


def _load_internal_fold_train(
    project_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    degree_config_path = project_dir / DEGREE_CONFIG_PATH
    degree_config = yaml.safe_load(degree_config_path.read_text(encoding="utf-8"))
    source = degree_config["inputs"]["positive_manifest"]
    manifest_path = project_dir / source["path"]
    if file_sha256(manifest_path) != source["sha256"]:
        raise ValueError("internal five-fold positive manifest hash mismatch")
    manifest = _read_json(manifest_path)
    folds = {}
    for fold_id in degree_config["fold_ids"]:
        output = manifest["folds"][fold_id]["partitions"]["train"]["positive_output"]
        path = project_dir / output["path"]
        if file_sha256(path) != output["sha256"]:
            raise ValueError(f"internal train hash mismatch: {fold_id}")
        rows = _read_csv(path)
        tcr_frequency = Counter(row["tcr_cdr3_beta"] for row in rows)
        folds[fold_id] = {
            "path": output["path"],
            "sha256": output["sha256"],
            "row_count": len(rows),
            "tcr_frequency": tcr_frequency,
            "hlas": {row["hla"] for row in rows},
        }
    provenance = {
        "config": {
            "path": DEGREE_CONFIG_PATH.as_posix(),
            "sha256": file_sha256(degree_config_path),
        },
        "manifest": {"path": source["path"], "sha256": source["sha256"]},
        "folds": {
            fold_id: {
                "path": value["path"],
                "sha256": value["sha256"],
                "row_count": value["row_count"],
            }
            for fold_id, value in folds.items()
        },
    }
    return folds, provenance


def _record_metadata(
    source: Mapping[str, Any],
    vdjdb: Mapping[str, Any],
    similarity: float,
    *,
    label: int,
    record_id: str,
    reference_id: str,
    antigen_source: str,
) -> dict[str, Any]:
    tcr = str(source["tcr_cdr3_beta"])
    epitope = str(source["epitope"])
    hla = str(source["hla"])
    return {
        "record_id": record_id,
        "interaction_key": f"{tcr}|{epitope}|{hla}",
        "tcr_cdr3_beta": tcr,
        "epitope": epitope,
        "hla": hla,
        "query_id": f"{epitope}|{hla}",
        "reference_id": reference_id,
        "max_similarity_to_vdjdb": f"{similarity:.12f}",
        "tcr_seen_in_vdjdb": tcr in vdjdb["tcrs"],
        "hla_seen_in_vdjdb": hla in vdjdb["hlas"],
        "antigen_source": antigen_source,
        "source_database": "IEDB receptor_full_v3",
        "label": label,
    }


def _proportion(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _seen_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    tcrs: set[str],
    hlas: set[str],
) -> dict[str, Any]:
    unique_tcrs = {str(row["tcr_cdr3_beta"]) for row in records}
    unique_hlas = {str(row["hla"]) for row in records}
    seen_rows_tcr = sum(str(row["tcr_cdr3_beta"]) in tcrs for row in records)
    seen_rows_hla = sum(str(row["hla"]) in hlas for row in records)
    return {
        "record_count": len(records),
        "tcr_seen_record_count": seen_rows_tcr,
        "tcr_seen_record_fraction": _proportion(seen_rows_tcr, len(records)),
        "hla_seen_record_count": seen_rows_hla,
        "hla_seen_record_fraction": _proportion(seen_rows_hla, len(records)),
        "unique_tcr_count": len(unique_tcrs),
        "unique_tcr_seen_count": len(unique_tcrs & tcrs),
        "unique_tcr_seen_fraction": _proportion(len(unique_tcrs & tcrs), len(unique_tcrs)),
        "unique_hla_count": len(unique_hlas),
        "unique_hla_seen_count": len(unique_hlas & hlas),
        "unique_hla_seen_fraction": _proportion(len(unique_hlas & hlas), len(unique_hlas)),
    }


def _compute_cohort(project_dir: Path, config_path: Path) -> dict[str, Any]:
    cohort_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _verify_frozen_config(cohort_config)
    strict, vdjdb, development, positive_registry, similarities, audit_config = (
        _load_strict_source(project_dir, cohort_config)
    )
    rule = cohort_config["final_inclusion_rule"]
    minimum = int(rule["minimum_positive_records_per_query"])
    counts = Counter(_query_key(row) for row in strict)
    eligible_queries = {query for query, count in counts.items() if count >= minimum}
    eligible = [row for row in strict if _query_key(row) in eligible_queries]
    selected = _select_capped_records(
        eligible,
        int(rule["query_cap_for_degree_matching"]),
        int(audit_config["degree_matching"]["seed"]),
    )

    feasibility_source = cohort_config["source_audit"]["degree_matching_feasibility"]
    feasibility_path = project_dir / feasibility_source["path"]
    if file_sha256(feasibility_path) != feasibility_source["sha256"]:
        raise ValueError("cap100 feasibility evidence hash mismatch")
    feasibility_rows = _read_csv(feasibility_path)
    evidence = next(
        row
        for row in feasibility_rows
        if int(row["query_cap"]) == int(rule["query_cap_for_degree_matching"])
    )
    if len(eligible_queries) != int(evidence["retained_queries"]):
        raise ValueError("formal cohort query count differs from frozen feasibility")
    if len(selected) != int(evidence["retained_positive_records"]):
        raise ValueError("formal cohort positive count differs from frozen feasibility")

    solver_positives = [
        {
            "tcr_cdr3_beta": row["tcr_cdr3_beta"],
            "epitope": row["epitope"],
            "hla": row["hla"],
            "label": 1,
        }
        for row in selected
    ]
    negative_edges, degree = solve_degree_matched_partition(
        solver_positives,
        positive_registry,
        fold_id="fold_1",
        partition="test",
        seed=int(audit_config["degree_matching"]["seed"])
        + int(rule["query_cap_for_degree_matching"]),
    )
    degree_audit = audit_degree_matched_partition(
        solver_positives, negative_edges, positive_registry
    )

    query_references: dict[tuple[str, str], set[str]] = defaultdict(set)
    query_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in selected:
        query = _query_key(row)
        query_references[query].update(row["publication_tokens"])
        query_sources[query].update(row["antigen_sources"])

    positives = []
    for row in selected:
        key = _pair_key(row)
        positives.append(
            _record_metadata(
                row,
                vdjdb,
                similarities[row["epitope"]],
                label=1,
                record_id=_stable_id("iedb-positive", *key),
                reference_id=_join(row["publication_tokens"]),
                antigen_source=_join(row["antigen_sources"]) or "unknown",
            )
        )
    negatives = []
    for row in negative_edges:
        query = _query_key(row)
        key = _pair_key(row)
        negatives.append(
            _record_metadata(
                row,
                vdjdb,
                similarities[row["epitope"]],
                label=0,
                record_id=_stable_id("iedb-putative-negative", *key),
                reference_id=_join(query_references[query]),
                antigen_source=_join(query_sources[query]) or "unknown",
            )
        )
    positives.sort(key=lambda row: (row["query_id"], row["tcr_cdr3_beta"]))
    negatives.sort(key=lambda row: (row["query_id"], row["tcr_cdr3_beta"]))
    test_rows = sorted(
        positives + negatives,
        key=lambda row: (row["query_id"], -int(row["label"]), row["tcr_cdr3_beta"]),
    )

    internal_folds, internal_provenance = _load_internal_fold_train(project_dir)
    annotations = []
    internal_seen = {}
    for fold_id, fold in internal_folds.items():
        tcrs = set(fold["tcr_frequency"])
        hlas = set(fold["hlas"])
        internal_seen[fold_id] = _seen_summary(test_rows, tcrs=tcrs, hlas=hlas)
        for row in test_rows:
            tcr = row["tcr_cdr3_beta"]
            annotations.append(
                {
                    "record_id": row["record_id"],
                    "interaction_key": row["interaction_key"],
                    "label": row["label"],
                    "fold_id": fold_id,
                    "tcr_seen_in_internal_fold_train": tcr in tcrs,
                    "hla_seen_in_internal_fold_train": row["hla"] in hlas,
                    "tcr_train_frequency": int(fold["tcr_frequency"][tcr]),
                }
            )
    annotations.sort(key=lambda row: (row["record_id"], row["fold_id"]))

    selected_keys = {_pair_key(row) for row in selected}
    publication_tokens = {
        token for row in selected for token in row["publication_tokens"]
    }
    selected_peptides = {row["epitope"] for row in selected}
    maximum_similarity = max(
        (similarities[peptide] for peptide in selected_peptides), default=0.0
    )
    negative_keys = [_pair_key(row) for row in negative_edges]
    checks = {
        "mapped_publication_reference_missing_count": sum(
            not row["publication_tokens"] for row in selected
        ),
        "shared_vdjdb_publication_reference_count": len(
            publication_tokens & vdjdb["publications"]
        ),
        "exact_positive_interaction_overlap_with_vdjdb": len(
            selected_keys & vdjdb["pairs"]
        ),
        "development_peptide_overlap_count": len(
            selected_peptides & development["peptides"]
        ),
        "maximum_similarity_to_development_peptide": maximum_similarity,
        "maximum_similarity_threshold": float(
            rule["maximum_development_peptide_similarity"]
        ),
        "negative_complete_positive_registry_overlap_count": len(
            set(negative_keys) & positive_registry
        ),
        "duplicate_positive_interaction_count": len(selected_keys) - len(selected),
        "duplicate_negative_interaction_count": len(negative_keys)
        - len(set(negative_keys)),
        "positive_negative_interaction_overlap_count": len(
            selected_keys & set(negative_keys)
        ),
        "tcr_degree_exact": degree_audit["tcr_degree_exact"],
        "query_degree_exact": degree_audit["query_degree_exact"],
        "degree_matching_rate": degree_audit["matching_rate"],
        "all_independence_checks_passed": True,
    }
    expected_checks = (
        checks["mapped_publication_reference_missing_count"] == 0
        and checks["shared_vdjdb_publication_reference_count"] == 0
        and checks["exact_positive_interaction_overlap_with_vdjdb"] == 0
        and checks["development_peptide_overlap_count"] == 0
        and checks["maximum_similarity_to_development_peptide"]
        <= checks["maximum_similarity_threshold"]
        and checks["negative_complete_positive_registry_overlap_count"] == 0
        and checks["duplicate_positive_interaction_count"] == 0
        and checks["duplicate_negative_interaction_count"] == 0
        and checks["positive_negative_interaction_overlap_count"] == 0
        and checks["tcr_degree_exact"] is True
        and checks["query_degree_exact"] is True
        and checks["degree_matching_rate"] == float(rule["exact_degree_matching_rate"])
    )
    if not expected_checks:
        checks["all_independence_checks_passed"] = False
        raise ValueError(f"formal IEDB cohort invariant failed: {checks}")

    categories = Counter(
        _source_category(row, audit_config["antigen_source_categories"])
        for row in selected
    )
    summary = {
        "query_count": len(eligible_queries),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "test_count": len(test_rows),
        "unique_tcr_count": len({row["tcr_cdr3_beta"] for row in test_rows}),
        "unique_peptide_count": len(selected_peptides),
        "unique_hla_count": len({row["hla"] for row in test_rows}),
        "publication_count": len(publication_tokens),
        "vdjdb_seen": _seen_summary(
            test_rows, tcrs=set(vdjdb["tcrs"]), hlas=set(vdjdb["hlas"])
        ),
        "internal_fold_train_seen": internal_seen,
        "positive_antigen_source_distribution": dict(sorted(categories.items())),
    }
    source_provenance = {
        "cohort_config": {
            "path": str(config_path.relative_to(project_dir)),
            "sha256": file_sha256(config_path),
        },
        "audit_config": cohort_config["source_audit"]["config"],
        "feasibility_evidence": cohort_config["source_audit"][
            "degree_matching_feasibility"
        ],
        "iedb_archives": audit_config["source"]["archives"],
        "vdjdb": audit_config["frozen_vdjdb"],
        "internal_five_fold_train": internal_provenance,
    }
    return {
        "positives": positives,
        "negatives": negatives,
        "test": test_rows,
        "annotations": annotations,
        "summary": summary,
        "checks": checks,
        "source_provenance": source_provenance,
        "frozen_rule": dict(rule),
        "degree_audit": degree,
        "no_model_prediction_used": True,
        "training_performed": False,
        "scoring_performed": False,
        "statistical_testing_performed": False,
        "plotting_performed": False,
    }


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _render_report(computation: Mapping[str, Any]) -> str:
    summary = computation["summary"]
    checks = computation["checks"]
    lines = [
        "# Final IEDB strict external test cohort",
        "",
        "## Frozen construction protocol",
        "",
        "- Tier: strict C",
        "- Minimum positives per query: 1",
        "- Deterministic query cap: 100",
        "- Publication/reference: canonical PMID or DOI required",
        "- Negatives: exact bipartite degree matching against the complete IEDB + VDJdb positive registry",
        "- No model prediction, performance result, or manual record selection was used",
        "",
        "## Cohort summary",
        "",
        f"- Queries: {summary['query_count']}",
        f"- Positives: {summary['positive_count']}",
        f"- Putative negatives: {summary['negative_count']}",
        f"- Unique TCRs: {summary['unique_tcr_count']}",
        f"- Unique peptides: {summary['unique_peptide_count']}",
        f"- Unique HLAs: {summary['unique_hla_count']}",
        f"- Unique publications: {summary['publication_count']}",
        "",
        "## VDJdb overlap annotations",
        "",
        f"- TCR seen, record-level: {summary['vdjdb_seen']['tcr_seen_record_fraction']:.6f}",
        f"- HLA seen, record-level: {summary['vdjdb_seen']['hla_seen_record_fraction']:.6f}",
        f"- TCR seen, unique-level: {summary['vdjdb_seen']['unique_tcr_seen_fraction']:.6f}",
        f"- HLA seen, unique-level: {summary['vdjdb_seen']['unique_hla_seen_fraction']:.6f}",
        "",
        "## Internal fold-train overlap annotations",
        "",
        "| Fold | TCR seen records | HLA seen records | Unique TCR seen | Unique HLA seen |",
        "|---|---:|---:|---:|---:|",
    ]
    for fold_id, seen in summary["internal_fold_train_seen"].items():
        lines.append(
            f"| {fold_id} | {seen['tcr_seen_record_fraction']:.6f} | "
            f"{seen['hla_seen_record_fraction']:.6f} | "
            f"{seen['unique_tcr_seen_fraction']:.6f} | "
            f"{seen['unique_hla_seen_fraction']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Positive antigen-source distribution",
            "",
            "| Category | Records |",
            "|---|---:|",
        ]
    )
    for category, count in summary["positive_antigen_source_distribution"].items():
        lines.append(f"| {category} | {count} |")
    lines.extend(
        [
            "",
            "## Independence and matching checks",
            "",
            "| Check | Result |",
            "|---|---:|",
        ]
    )
    for name, value in checks.items():
        lines.append(f"| {name} | {value} |")
    lines.extend(
        [
            "",
            "All checks passed. No training, model scoring, statistical testing, or plotting was performed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_once(root: Path, computation: Mapping[str, Any]) -> dict[str, str]:
    paths = {name: root / relative for name, relative in {**DATA_OUTPUTS, **RESULT_OUTPUTS}.items()}
    _write_csv(paths["positives"], RECORD_COLUMNS, computation["positives"])
    _write_csv(paths["negatives"], RECORD_COLUMNS, computation["negatives"])
    _write_csv(paths["test"], RECORD_COLUMNS, computation["test"])
    _write_csv(
        paths["internal_annotations"], ANNOTATION_COLUMNS, computation["annotations"]
    )
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(_render_report(computation), encoding="utf-8")
    non_manifest = {
        name: {
            "path": ({**DATA_OUTPUTS, **RESULT_OUTPUTS})[name],
            "sha256": file_sha256(path),
            "rows": len(computation[name])
            if name in {"positives", "negatives", "test"}
            else len(computation["annotations"])
            if name == "internal_annotations"
            else None,
        }
        for name, path in paths.items()
        if name != "manifest"
    }
    manifest = {
        "schema_version": 1,
        "source_commit": "9b308b57bfdde991dad9cda2df6dd1707b568ec3",
        "frozen_rule": computation["frozen_rule"],
        "source_provenance": computation["source_provenance"],
        "summary": computation["summary"],
        "independence_and_matching_checks": computation["checks"],
        "degree_matching_audit": computation["degree_audit"],
        "columns": {
            "cohort_records": list(RECORD_COLUMNS),
            "internal_annotations": list(ANNOTATION_COLUMNS),
        },
        "outputs": non_manifest,
        "reproducibility": {
            "complete_rebuilds_compared": 2,
            "all_output_sha256_identical": True,
        },
        "prohibited_operations": {
            "model_prediction_used": False,
            "training_performed": False,
            "scoring_performed": False,
            "statistical_testing_performed": False,
            "plotting_performed": False,
        },
    }
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {name: file_sha256(path) for name, path in paths.items()}


def build_iedb_external_cohort(project_dir: Path, config_path: Path) -> dict[str, Any]:
    data_directory = project_dir / "data/external/iedb_strict_cap100"
    expected_data = {Path(path).name for path in DATA_OUTPUTS.values()}
    if data_directory.exists():
        unexpected = sorted(
            path.name for path in data_directory.iterdir() if path.name not in expected_data
        )
        if unexpected:
            raise ValueError(
                "refusing to overwrite external cohort directory with unexpected files: "
                + ", ".join(unexpected)
            )
    with tempfile.TemporaryDirectory(prefix="iedb-cohort-build-1-") as first_temp, tempfile.TemporaryDirectory(
        prefix="iedb-cohort-build-2-"
    ) as second_temp:
        first_root = Path(first_temp)
        second_root = Path(second_temp)
        first = _compute_cohort(project_dir, config_path)
        first_hashes = _write_once(first_root, first)
        second = _compute_cohort(project_dir, config_path)
        second_hashes = _write_once(second_root, second)
        if first_hashes != second_hashes:
            raise ValueError("two complete IEDB cohort rebuilds produced different hashes")
        for relative in {**DATA_OUTPUTS, **RESULT_OUTPUTS}.values():
            source = first_root / relative
            destination = project_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    return {
        "summary": first["summary"],
        "checks": first["checks"],
        "output_sha256": first_hashes,
        "reproducibility_passed": True,
    }
