from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .degree_matched_negatives import ExactDegreeMatchError, solve_degree_matched_partition
from .iedb_cohort_decision import _reconstruct_strict_records
from .iedb_external_audit import _select_capped_records
from .vdjdb import file_sha256


REPORT_MARKER = "## Protocol amendment: formal cohort cap=100"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _history_sha256(decision: Mapping[str, Any]) -> str:
    historical = {
        "thresholds": decision["thresholds"],
        "selection": decision["selection"],
    }
    payload = json.dumps(
        historical, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_amendment(
    project_dir: Path, config: Mapping[str, Any], decision: Mapping[str, Any]
) -> dict[str, Any]:
    amendments = config.get("protocol_amendments")
    if not isinstance(amendments, list) or len(amendments) != 1:
        raise ValueError("exactly one cap100 protocol amendment must be frozen")
    amendment = amendments[0]
    if amendment.get("timing") != "pre-scoring":
        raise ValueError("cap100 amendment must be recorded as pre-scoring")
    if amendment.get("external_model_scoring_performed_before_amendment") is not False:
        raise ValueError("external scoring occurred before cap amendment")
    if amendment.get("external_model_performance_viewed") is not False:
        raise ValueError("external model performance was viewed before cap amendment")
    if decision.get("model_scoring_performed") is not False:
        raise ValueError("cohort decision records external model scoring")
    if amendment.get("previous_value") != 20 or amendment.get("amended_value") != 100:
        raise ValueError("cap amendment must retain the 20 to 100 history")
    if config["original_pass_conditions"]["degree_matching_query_cap"] != 20:
        raise ValueError("historical cap20 PASS condition was overwritten")
    if config["final_inclusion_rule"]["query_cap_for_degree_matching"] != 100:
        raise ValueError("formal cohort cap is not frozen at 100")
    evidence = amendment["feasibility_evidence"]
    if file_sha256(project_dir / evidence["path"]) != evidence["sha256"]:
        raise ValueError("cap100 feasibility evidence hash mismatch")
    history = amendment["preserved_cap20_history"]
    if file_sha256(project_dir / history["decision_csv_path"]) != history[
        "decision_csv_sha256"
    ]:
        raise ValueError("historical cap20 decision CSV changed")
    if "protocol_amendment" not in decision:
        decision_path = project_dir / config["outputs"]["directory"] / config["outputs"]["json"]
        report_path = project_dir / config["outputs"]["directory"] / config["outputs"]["report"]
        if file_sha256(decision_path) != history["pre_amendment_decision_json_sha256"]:
            raise ValueError("pre-amendment decision JSON hash mismatch")
        if file_sha256(report_path) != history["pre_amendment_report_sha256"]:
            raise ValueError("pre-amendment report hash mismatch")
    return dict(amendment)


def _render_amendment(
    original_report: str,
    amendment: Mapping[str, Any],
    simulation: Mapping[str, Any],
    history_sha256: str,
) -> str:
    base = original_report.split(REPORT_MARKER, maxsplit=1)[0].rstrip()
    lines = [
        base,
        "",
        REPORT_MARKER,
        "",
        "This amendment was frozen before any external model scoring. No external model performance was viewed.",
        "",
        "The formal-cohort cap changes from 20 to 100 to resolve the mismatch between the earlier cohort-decision configuration and the already audited cap=100 cohort size. All independence, minimum-positive, and reference rules remain unchanged.",
        "",
        "The original cap=20 threshold table and `cohort_decision.csv` remain unchanged as historical records.",
        "",
        f"- Amendment ID: `{amendment['amendment_id']}`",
        f"- Timing: `{amendment['timing']}`",
        f"- Historical cap=20 decision SHA-256: `{history_sha256}`",
        f"- Formal cohort query cap: `{simulation['query_cap']}`",
        f"- Reconstructed strict-C queries: `{simulation['query_count']}`",
        f"- Deterministically retained positives: `{simulation['positive_count']}`",
        f"- Exact degree matching: `{simulation['exact_degree_match']}`",
        f"- Degree-matching rate: `{simulation['degree_matching_rate']:.6f}`",
        "",
        "No formal external CSV, negative sample, model score, statistical test, or figure was generated.",
    ]
    return "\n".join(lines) + "\n"


def freeze_cap100_policy(project_dir: Path, config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = project_dir / config["outputs"]["directory"]
    decision_path = output_dir / config["outputs"]["json"]
    report_path = output_dir / config["outputs"]["report"]
    decision = _read_json(decision_path)
    amendment = _validate_amendment(project_dir, config, decision)
    history_sha256 = (
        decision.get("protocol_amendment", {}).get("preserved_cap20_history_sha256")
        or _history_sha256(decision)
    )

    audit_config = yaml.safe_load(
        (project_dir / config["source_audit"]["config"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    strict, positive_registry, independence, tier_counts = _reconstruct_strict_records(
        project_dir, audit_config
    )
    minimum = int(config["final_inclusion_rule"]["minimum_positive_records_per_query"])
    query_counts: dict[tuple[str, str], int] = {}
    for row in strict:
        query = (row["epitope"], row["hla"])
        query_counts[query] = query_counts.get(query, 0) + 1
    eligible_queries = {
        query for query, count in query_counts.items() if count >= minimum
    }
    eligible = [
        row
        for row in strict
        if (row["epitope"], row["hla"]) in eligible_queries
    ]
    query_cap = int(config["final_inclusion_rule"]["query_cap_for_degree_matching"])
    capped = _select_capped_records(
        eligible, query_cap, int(audit_config["degree_matching"]["seed"])
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
            seed=int(audit_config["degree_matching"]["seed"]) + query_cap,
        )
    except ExactDegreeMatchError as error:
        raise ValueError(
            "cap100 did not reproduce the previously audited exact solution: "
            + json.dumps(error.report, sort_keys=True)
        ) from error

    evidence_rows = _read_csv(
        project_dir / amendment["feasibility_evidence"]["path"]
    )
    evidence = next(
        row for row in evidence_rows if int(row["query_cap"]) == query_cap
    )
    query_count = len(eligible_queries)
    positive_count = len(capped)
    if query_count != int(evidence["retained_queries"]):
        raise ValueError("cap100 query count differs from prior feasibility audit")
    if positive_count != int(evidence["retained_positive_records"]):
        raise ValueError("cap100 positive count differs from prior feasibility audit")
    if evidence["exact_solution"] != "True" or float(evidence["matching_rate"]) != 1.0:
        raise ValueError("prior cap100 feasibility was not an exact solution")
    if float(degree["matching_rate"]) != float(
        config["final_inclusion_rule"]["exact_degree_matching_rate"]
    ):
        raise ValueError("reconstructed cap100 degree matching is not exact")

    simulation = {
        "tier": config["final_inclusion_rule"]["tier"],
        "minimum_positive_records_per_query": minimum,
        "query_cap": query_cap,
        "query_count": query_count,
        "positive_count": positive_count,
        "exact_degree_match": True,
        "degree_matching_rate": float(degree["matching_rate"]),
        "matched_edge_count": int(degree["maximum_flow_value"]),
        "tier_counts_reconstructed": tier_counts,
        "independence_reverified": independence,
        "expected_values_source": amendment["feasibility_evidence"],
        "formal_external_csv_generated": False,
        "negative_samples_persisted": False,
    }
    decision["protocol_amendment"] = {
        **amendment,
        "preserved_cap20_history_sha256": history_sha256,
        "pre_scoring_verified": True,
        "external_model_performance_viewed": False,
        "external_model_scoring_performed": False,
    }
    decision["final_inclusion_rule"] = config["final_inclusion_rule"]
    decision["formal_cohort_cap_simulation"] = simulation
    decision["model_scoring_performed"] = False
    decision["training_performed"] = False
    decision["plotting_performed"] = False
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        _render_amendment(
            report_path.read_text(encoding="utf-8"),
            amendment,
            simulation,
            history_sha256,
        ),
        encoding="utf-8",
    )
    return {
        "protocol_amendment": decision["protocol_amendment"],
        "simulation": simulation,
        "decision_json_sha256": file_sha256(decision_path),
        "report_sha256": file_sha256(report_path),
    }
