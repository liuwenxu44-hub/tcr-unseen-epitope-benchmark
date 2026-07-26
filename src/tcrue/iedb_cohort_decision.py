from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .degree_matched_negatives import ExactDegreeMatchError, solve_degree_matched_partition
from .iedb_external_audit import (
    _select_capped_records,
    _source_category,
    build_candidate_tiers,
    load_iedb_references,
    load_vdjdb_registry,
    parse_iedb_receptors,
)
from .vdjdb import file_sha256


CSV_COLUMNS = (
    "minimum_positives_per_query",
    "selection_order",
    "query_count",
    "positive_record_count",
    "unique_tcr_count",
    "unique_hla_count",
    "publication_count",
    "pathogen_or_nonself_records",
    "tumor_associated_records",
    "self_antigen_records",
    "other_nonself_records",
    "unknown_source_records",
    "degree_matching_query_cap",
    "degree_matching_retained_query_count",
    "degree_matching_retained_positive_count",
    "degree_matching_matched_edge_count",
    "degree_matching_rate",
    "degree_matching_exact",
    "records_gate_passed",
    "queries_gate_passed",
    "publications_gate_passed",
    "capped_records_gate_passed",
    "capped_queries_gate_passed",
    "all_gates_passed",
    "selected",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_source_files(project_dir: Path, config: Mapping[str, Any]) -> None:
    for name, source in config["source_audit"].items():
        path = project_dir / source["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = file_sha256(path)
        if observed != source["sha256"]:
            raise ValueError(
                f"source audit hash mismatch for {name}: {observed} != {source['sha256']}"
            )


def audit_original_decision(
    audit_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    tier_summary_rows: Sequence[Mapping[str, str]],
    degree_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    frozen_rule = manifest.get("predeclared_decision_rule", {}).get("pass")
    if not isinstance(frozen_rule, dict):
        raise ValueError("source audit did not freeze a PASS quantity rule")
    config_rule = audit_config.get("decision_rule", {}).get("pass")
    if frozen_rule != config_rule:
        raise ValueError("audit config and manifest PASS rules differ")
    required = {
        "strict_positive_records_at_least",
        "strict_queries_with_at_least_10_positives",
        "strict_publications_at_least",
        "cap20_exact_degree_match",
        "cap20_queries_at_least",
        "cap20_positive_records_at_least",
    }
    if set(frozen_rule) != required:
        raise ValueError("source audit PASS rule is incomplete or has changed")
    strict = next(row for row in tier_summary_rows if row["tier"] == "C")
    cap20 = next(row for row in degree_rows if int(row["query_cap"]) == 20)
    conditions = [
        {
            "condition": "strict_positive_records_at_least",
            "required": int(frozen_rule["strict_positive_records_at_least"]),
            "actual": int(strict["positive_records"]),
            "passed": int(strict["positive_records"])
            >= int(frozen_rule["strict_positive_records_at_least"]),
        },
        {
            "condition": "strict_queries_with_at_least_10_positives",
            "required": int(frozen_rule["strict_queries_with_at_least_10_positives"]),
            "actual": int(strict["queries_min_10_positives"]),
            "passed": int(strict["queries_min_10_positives"])
            >= int(frozen_rule["strict_queries_with_at_least_10_positives"]),
        },
        {
            "condition": "strict_publications_at_least",
            "required": int(frozen_rule["strict_publications_at_least"]),
            "actual": int(strict["unique_publications"]),
            "passed": int(strict["unique_publications"])
            >= int(frozen_rule["strict_publications_at_least"]),
        },
        {
            "condition": "cap20_exact_degree_match",
            "required": bool(frozen_rule["cap20_exact_degree_match"]),
            "actual": cap20["exact_solution"] == "True",
            "passed": (cap20["exact_solution"] == "True")
            is bool(frozen_rule["cap20_exact_degree_match"]),
        },
        {
            "condition": "cap20_queries_at_least",
            "required": int(frozen_rule["cap20_queries_at_least"]),
            "actual": int(cap20["retained_queries"]),
            "passed": int(cap20["retained_queries"])
            >= int(frozen_rule["cap20_queries_at_least"]),
        },
        {
            "condition": "cap20_positive_records_at_least",
            "required": int(frozen_rule["cap20_positive_records_at_least"]),
            "actual": int(cap20["retained_positive_records"]),
            "passed": int(cap20["retained_positive_records"])
            >= int(frozen_rule["cap20_positive_records_at_least"]),
        },
    ]
    failures = [row["condition"] for row in conditions if not row["passed"]]
    if failures != ["strict_queries_with_at_least_10_positives"]:
        raise ValueError(f"CONDITIONAL does not have exactly the expected cause: {failures}")
    if manifest["audit"]["decision"] != "CONDITIONAL":
        raise ValueError("source audit decision is not CONDITIONAL")
    return {
        "conditions": conditions,
        "failed_conditions": failures,
        "sole_conditional_reason": (
            "Strict tier C has 7 queries with at least 10 positives, below the "
            "predeclared minimum of 20 queries."
        ),
    }


def _reconstruct_strict_records(
    project_dir: Path, audit_config: Mapping[str, Any]
) -> tuple[
    list[dict[str, Any]],
    set[tuple[str, str, str]],
    dict[str, Any],
    dict[str, dict[str, int]],
]:
    receptor = audit_config["source"]["archives"]["receptor"]
    reference = audit_config["source"]["archives"]["reference"]
    references, _ = load_iedb_references(
        project_dir / reference["path"], reference["member"]
    )
    records, _, _ = parse_iedb_receptors(
        project_dir / receptor["path"],
        receptor["member"],
        references,
        audit_config["scope"],
    )
    full = load_vdjdb_registry(
        project_dir / audit_config["frozen_vdjdb"]["positive_registry"]
    )
    development = load_vdjdb_registry(
        project_dir / audit_config["frozen_vdjdb"]["development_registry"]
    )
    tiers, similarities = build_candidate_tiers(
        records, full, development, audit_config
    )
    strict = tiers["C"]
    complete_registry = set(full["pairs"]) | {
        (row["tcr_cdr3_beta"], row["epitope"], row["hla"]) for row in records
    }
    independence = {
        "publication_reference_overlap": sum(
            bool(set(row["publication_tokens"]) & full["publications"])
            for row in strict
        ),
        "development_peptide_overlap": len(
            {row["epitope"] for row in strict} & development["peptides"]
        ),
        "maximum_development_peptide_similarity": max(
            (similarities[row["epitope"]] for row in strict), default=0.0
        ),
    }
    tier_counts = {
        tier: {
            "positive_records": len(tier_records),
            "unique_pmhc_queries": len(
                {(row["epitope"], row["hla"]) for row in tier_records}
            ),
        }
        for tier, tier_records in tiers.items()
    }
    return strict, complete_registry, independence, tier_counts


def _validate_reference_policy(
    decision_config: Mapping[str, Any], audit_config: Mapping[str, Any]
) -> dict[str, Any]:
    policy = decision_config.get("reference_policy")
    if not isinstance(policy, dict):
        raise ValueError("final cohort config does not explicitly freeze reference policy")
    if policy.get("require_mapped_publication_reference") is not True:
        raise ValueError("mapped publication/reference must be required")
    if audit_config["independence"].get("require_mapped_publication_for_tier_a") is not True:
        raise ValueError("upstream tier-A mapped-publication rule is not enabled")
    provenance = policy.get("provenance", {})
    if (
        provenance.get("upstream_config_key")
        != "independence.require_mapped_publication_for_tier_a"
        or provenance.get("upstream_value") is not True
    ):
        raise ValueError("reference-policy provenance does not identify upstream rule")
    return {
        "require_mapped_publication_reference": True,
        "excluded_reference_states": [
            "missing",
            "blank",
            "not standardizable to canonical PMID or DOI",
        ],
        "action": "exclude from formal external validation",
        "rationale": policy["rationale"],
        "normalization_reused": provenance["normalization_functions"],
        "upstream_rule_reused": provenance["upstream_config_key"],
        "new_reference_rule_implemented": False,
    }


def _summarize_threshold(
    strict_records: Sequence[Mapping[str, Any]],
    threshold: int,
    order: int,
    positive_registry: set[tuple[str, str, str]],
    audit_config: Mapping[str, Any],
    decision_config: Mapping[str, Any],
) -> dict[str, Any]:
    query_counts = Counter((row["epitope"], row["hla"]) for row in strict_records)
    eligible_queries = {
        query for query, count in query_counts.items() if count >= threshold
    }
    records = [
        dict(row)
        for row in strict_records
        if (row["epitope"], row["hla"]) in eligible_queries
    ]
    publications = {
        token for row in records for token in row["publication_tokens"]
    }
    sources = Counter(
        _source_category(row, audit_config["antigen_source_categories"])
        for row in records
    )
    rule = decision_config["original_pass_conditions"]
    query_cap = int(rule["degree_matching_query_cap"])
    capped = _select_capped_records(
        records, query_cap, int(audit_config["degree_matching"]["seed"])
    )
    positives = [
        {
            "tcr_cdr3_beta": row["tcr_cdr3_beta"],
            "epitope": row["epitope"],
            "hla": row["hla"],
            "label": 1,
        }
        for row in capped
    ]
    try:
        _, degree = solve_degree_matched_partition(
            positives,
            positive_registry,
            fold_id="fold_1",
            partition="test",
            seed=int(audit_config["degree_matching"]["seed"]) + threshold,
        )
        exact = True
        matched = int(degree["maximum_flow_value"])
        matching_rate = float(degree["matching_rate"])
    except ExactDegreeMatchError as error:
        exact = False
        matched = int(error.report["maximum_matched_edge_count"])
        matching_rate = float(error.report["maximum_matching_ratio"])

    gates = {
        "records_gate_passed": len(records)
        >= int(rule["strict_positive_records_at_least"]),
        "queries_gate_passed": len(eligible_queries)
        >= int(rule["strict_queries_at_least"]),
        "publications_gate_passed": len(publications)
        >= int(rule["strict_publications_at_least"]),
        "capped_records_gate_passed": len(capped)
        >= int(rule["capped_positive_records_at_least"]),
        "capped_queries_gate_passed": len(eligible_queries)
        >= int(rule["capped_queries_at_least"]),
    }
    all_gates = (
        all(gates.values())
        and exact is bool(rule["exact_degree_match"])
        and matching_rate
        == float(decision_config["immutable_independence"]["exact_degree_matching_rate"])
    )
    return {
        "minimum_positives_per_query": threshold,
        "selection_order": order,
        "query_count": len(eligible_queries),
        "positive_record_count": len(records),
        "unique_tcr_count": len({row["tcr_cdr3_beta"] for row in records}),
        "unique_hla_count": len({row["hla"] for row in records}),
        "publication_count": len(publications),
        "pathogen_or_nonself_records": sources["pathogen_or_nonself"],
        "tumor_associated_records": sources["tumor_associated"],
        "self_antigen_records": sources["self_antigen"],
        "other_nonself_records": sources["other_nonself"],
        "unknown_source_records": sources["unknown_source"],
        "degree_matching_query_cap": query_cap,
        "degree_matching_retained_query_count": len(eligible_queries),
        "degree_matching_retained_positive_count": len(capped),
        "degree_matching_matched_edge_count": matched,
        "degree_matching_rate": matching_rate,
        "degree_matching_exact": exact,
        **gates,
        "all_gates_passed": all_gates,
        "selected": False,
    }


def _render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# IEDB strict-C cohort decision",
        "",
        "## Original predeclared PASS conditions",
        "",
        "| Condition | Required | Actual | Passed |",
        "|---|---:|---:|---|",
    ]
    for condition in result["original_decision_audit"]["conditions"]:
        lines.append(
            f"| {condition['condition']} | {condition['required']} | "
            f"{condition['actual']} | {condition['passed']} |"
        )
    lines.extend(
        [
            "",
            "## Explicit reference policy",
            "",
            "A mapped canonical publication/reference (PMID or DOI) is required. Records whose reference is missing, blank, or cannot be standardized are excluded because publication independence from VDJdb cannot otherwise be demonstrated.",
            "",
            "This is the existing upstream `require_mapped_publication_for_tier_a` and reference-normalization logic made explicit; no new normalization rule was introduced.",
            "",
            "## Quantity re-verification",
            "",
            "| Tier | Before records | After records | Before queries | After queries | Changed |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for tier in ("A", "B", "C"):
        row = result["quantity_reverification"]["tiers"][tier]
        lines.append(
            f"| {tier} | {row['before']['positive_records']} | "
            f"{row['after']['positive_records']} | "
            f"{row['before']['unique_pmhc_queries']} | "
            f"{row['after']['unique_pmhc_queries']} | {row['changed']} |"
        )
    lines.extend(
        [
            "",
            f"Final minimum-positive threshold before/after: {result['quantity_reverification']['final_threshold_before']} / {result['quantity_reverification']['final_threshold_after']}.",
            "",
            result["quantity_reverification"]["conclusion"],
            "",
            "## Sole reason for CONDITIONAL",
            "",
            result["original_decision_audit"]["sole_conditional_reason"],
            "",
            "Publication/reference independence, exact-unseen peptide status, the distance-aware threshold, and exact degree matching are immutable and were not relaxed.",
            "",
            "## Fixed-order minimum-positive evaluation",
            "",
            "| Minimum positives/query | Queries | Positive records | TCRs | HLAs | Publications | Pathogen/nonself | Tumor | Self | Other | Unknown | Cap-20 retained | Exact | All gates |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in result["thresholds"]:
        lines.append(
            f"| {row['minimum_positives_per_query']} | {row['query_count']} | "
            f"{row['positive_record_count']} | {row['unique_tcr_count']} | "
            f"{row['unique_hla_count']} | {row['publication_count']} | "
            f"{row['pathogen_or_nonself_records']} | {row['tumor_associated_records']} | "
            f"{row['self_antigen_records']} | {row['other_nonself_records']} | "
            f"{row['unknown_source_records']} | "
            f"{row['degree_matching_retained_positive_count']} | "
            f"{row['degree_matching_exact']} | {row['all_gates_passed']} |"
        )
    selected = result["selection"]
    lines.extend(
        [
            "",
            "## Frozen inclusion rule",
            "",
            f"The first fixed-order threshold satisfying every original gate is **{selected['minimum_positives_per_query']} positives per query**.",
            "",
            f"This retains {selected['query_count']} strict-C queries and {selected['positive_record_count']} positive records before the unchanged cap=20 degree-matching simulation; {selected['degree_matching_retained_positive_count']} positives are retained after capping, with an exact matching rate of {selected['degree_matching_rate']:.6f}.",
            "",
            "No formal external cohort or negative CSV was generated.",
        ]
    )
    return "\n".join(lines) + "\n"


def decide_iedb_cohort(
    project_dir: Path,
    config_path: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _verify_source_files(project_dir, config)
    audit_config = yaml.safe_load(
        (project_dir / config["source_audit"]["config"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    manifest = _read_json(project_dir / config["source_audit"]["manifest"]["path"])
    tier_rows = _read_csv(
        project_dir / config["source_audit"]["candidate_tier_summary"]["path"]
    )
    degree_rows = _read_csv(
        project_dir / config["source_audit"]["degree_matching_feasibility"]["path"]
    )
    original = audit_original_decision(
        audit_config, manifest, tier_rows, degree_rows
    )
    reference_policy = _validate_reference_policy(config, audit_config)
    strict, positive_registry, independence, recomputed_tier_counts = _reconstruct_strict_records(
        project_dir, audit_config
    )
    frozen_tier_counts = {
        row["tier"]: {
            "positive_records": int(row["positive_records"]),
            "unique_pmhc_queries": int(row["unique_pmhc_queries"]),
        }
        for row in tier_rows
    }
    tier_comparison = {
        tier: {
            "before": frozen_tier_counts[tier],
            "after": recomputed_tier_counts[tier],
            "changed": frozen_tier_counts[tier] != recomputed_tier_counts[tier],
        }
        for tier in ("A", "B", "C")
    }
    changed_tiers = [tier for tier, row in tier_comparison.items() if row["changed"]]
    if changed_tiers:
        raise ValueError(
            "explicit reference policy changed frozen tier quantities: "
            + ", ".join(changed_tiers)
        )
    immutable = config["immutable_independence"]
    if independence["publication_reference_overlap"] != int(
        immutable["publication_reference_overlap"]
    ):
        raise ValueError("strict-C publication/reference independence failed")
    if independence["development_peptide_overlap"] != int(
        immutable["development_peptide_overlap"]
    ):
        raise ValueError("strict-C exact-unseen peptide independence failed")
    if independence["maximum_development_peptide_similarity"] > float(
        immutable["maximum_development_peptide_similarity"]
    ):
        raise ValueError("strict-C peptide-distance independence failed")

    threshold_order = [int(value) for value in config["allowed_relaxation"]["threshold_order"]]
    if threshold_order != [20, 10, 5, 1]:
        raise ValueError("minimum-positive threshold order must be 20, 10, 5, 1")
    rows = [
        _summarize_threshold(
            strict,
            threshold,
            order,
            positive_registry,
            audit_config,
            config,
        )
        for order, threshold in enumerate(threshold_order, start=1)
    ]
    selected = next((row for row in rows if row["all_gates_passed"]), None)
    if selected is None:
        raise ValueError("no allowed minimum-positive threshold satisfies original gates")
    selected["selected"] = True
    previous_result = _read_json(
        project_dir / config["source_audit"]["manifest"]["path"]
    )
    if previous_result["audit"]["decision"] != "CONDITIONAL":
        raise ValueError("source feasibility decision changed")
    cohort_result_path = project_dir / config["outputs"]["directory"] / config["outputs"]["json"]
    previous_threshold = (
        _read_json(cohort_result_path)["selection"]["minimum_positives_per_query"]
        if cohort_result_path.is_file()
        else selected["minimum_positives_per_query"]
    )
    if previous_threshold != selected["minimum_positives_per_query"]:
        raise ValueError("explicit reference policy changed frozen cohort threshold")
    result = {
        "schema_version": 1,
        "base_commit": config["base_commit"],
        "source_audit": config["source_audit"],
        "original_decision_audit": original,
        "reference_policy": reference_policy,
        "quantity_reverification": {
            "tiers": tier_comparison,
            "all_tier_counts_unchanged": True,
            "final_threshold_before": previous_threshold,
            "final_threshold_after": selected["minimum_positives_per_query"],
            "final_threshold_unchanged": True,
            "conclusion": "\u89c4\u5219\u663e\u5f0f\u5316\uff0c\u65e0\u6570\u91cf\u53d8\u5316",
        },
        "immutable_independence": {
            **independence,
            "exact_degree_matching_rate_required": float(
                immutable["exact_degree_matching_rate"]
            ),
            "all_conditions_unchanged": True,
        },
        "allowed_relaxation": config["allowed_relaxation"],
        "thresholds": rows,
        "selection": dict(selected),
        "formal_external_dataset_generated": False,
        "training_performed": False,
        "model_scoring_performed": False,
        "plotting_performed": False,
    }
    config["final_inclusion_rule"].update(
        {
            "minimum_positive_records_per_query": selected[
                "minimum_positives_per_query"
            ],
            "status": "frozen",
        }
    )
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    destination = output_dir or project_dir / config["outputs"]["directory"]
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / config["outputs"]["csv"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (destination / config["outputs"]["json"]).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / config["outputs"]["report"]).write_text(
        _render_report(result), encoding="utf-8"
    )
    return result
