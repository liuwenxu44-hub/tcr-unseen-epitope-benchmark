"""H1 fixed descriptive estimands. No training, model scoring, CI, P or simulation.

Real data processing requires the separately issued root scoring gate and a closed
prediction index. Formula checks can import pure functions without opening data.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

RESAMPLES = tuple(f"resample_{i}" for i in range(1, 6))
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))
MASK_IDS = (
    "three_paper_joint__direct_selected_positive_support",
    "three_paper_joint__direct_full_query_support",
    "three_paper_joint__full_dependency_associations",
)
FIELDS = ("interaction_AP", "interaction_AUROC", "additive_AP", "additive_AUROC",
          "delta_Additive_AP", "delta_Additive_AUROC", "finite_random_no_tie_AP",
          "random_AUROC", "delta_finite_random_AP", "delta_random_AUROC")
EVIDENCE = "POSTHOC_DESCRIPTIVE_NOT_INFERENTIAL_NO_CI_NO_P"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("empty artifact is not permitted")
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("mean of empty analysis set")
    return math.fsum(values) / len(values)


def binary_metrics(labels, probabilities):
    """Threshold-group AP and midrank AUROC, preserving all equal-score ties."""
    if not labels or len(labels) != len(probabilities) or set(labels) != {0, 1}:
        raise ValueError("binary metric requires paired nonempty two-class rows")
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in probabilities):
        raise ValueError("invalid probability")
    positive, negative = sum(labels), len(labels) - sum(labels)
    groups = defaultdict(lambda: [0, 0])
    for label, score in zip(labels, probabilities, strict=True):
        groups[float(score)][label] += 1
    # AP equals sum over distinct score thresholds of recall increment * precision.
    seen_p = seen_n = 0
    ap_terms = []
    for score in sorted(groups, reverse=True):
        group_n, group_p = groups[score]
        seen_p += group_p
        seen_n += group_n
        ap_terms.append(group_p / positive * seen_p / (seen_p + seen_n))
    # AUROC counts positive-negative wins and half of exact ties.
    lower_n = 0
    auc_terms = []
    for score in sorted(groups):
        group_n, group_p = groups[score]
        auc_terms.append(group_p * (lower_n + group_n / 2))
        lower_n += group_n
    return {"AP": math.fsum(ap_terms), "AUROC": math.fsum(auc_terms) / (positive * negative)}


def finite_random_ap(positive, negative):
    """Exact E[AP] under a uniformly random no-tie permutation with fixed P,N."""
    if positive <= 0 or negative <= 0:
        raise ValueError("reference requires positive P and N")
    total = positive + negative
    harmonic = math.fsum(1 / rank for rank in range(1, total + 1))
    return positive / total + negative * (harmonic - 1) / (total * (total - 1))


def identity(row):
    tcr = row["tcr_cdr3_beta"].strip().upper()
    peptide = row["epitope"].strip().upper()
    hla = row["hla"].strip().upper()
    query = f"{peptide}|{hla}"
    key = f"{tcr}|{query}"
    if row["interaction_key"] != key or row["query_id"] != query:
        raise ValueError("semantic key mismatch")
    if int(row["label"]) not in (0, 1):
        raise ValueError("invalid label")
    if float(row["additive_probability"]) != 0.5:
        raise ValueError("analytic Additive must be exactly constant 0.5")
    return key, query, int(row["label"])


def summarize(rows, scope, set_id, train_resample="", test_resample=""):
    if len({identity(row)[0] for row in rows}) != len(rows):
        raise ValueError("duplicated interaction within prediction universe")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)
    queries = []
    for query_id, selected in sorted(grouped.items()):
        labels = [int(row["label"]) for row in selected]
        positive, negative = sum(labels), len(labels) - sum(labels)
        if positive != negative or positive <= 0:
            raise ValueError("H1 must retain paired complete balanced queries")
        interaction = binary_metrics(labels, [float(row["interaction_probability"]) for row in selected])
        additive = binary_metrics(labels, [float(row["additive_probability"]) for row in selected])
        reference = finite_random_ap(positive, negative)
        queries.append({
            "scope": scope, "set_id": set_id, "train_resample_id": train_resample,
            "test_resample_id": test_resample, "query_id": query_id,
            "positive_count": positive, "negative_count": negative, "record_count": len(selected),
            "singleton_positive_query": positive == 1,
            "interaction_AP": interaction["AP"], "interaction_AUROC": interaction["AUROC"],
            "additive_AP": additive["AP"], "additive_AUROC": additive["AUROC"],
            "delta_Additive_AP": interaction["AP"] - additive["AP"],
            "delta_Additive_AUROC": interaction["AUROC"] - additive["AUROC"],
            "finite_random_no_tie_AP": reference, "random_AUROC": 0.5,
            "delta_finite_random_AP": interaction["AP"] - reference,
            "delta_random_AUROC": interaction["AUROC"] - 0.5,
            "evidence_status": EVIDENCE,
        })
    if not queries:
        raise ValueError("empty query universe")
    summary = {
        "scope": scope, "set_id": set_id, "train_resample_id": train_resample,
        "test_resample_id": test_resample, "query_count": len(queries),
        "positive_count": sum(q["positive_count"] for q in queries),
        "negative_count": sum(q["negative_count"] for q in queries), "record_count": len(rows),
        "singleton_positive_query_count": sum(q["singleton_positive_query"] for q in queries),
        **{field: mean(q[field] for q in queries) for field in FIELDS},
        "evidence_status": EVIDENCE,
    }
    return summary, queries


def degree_audit(rows, full_rows, scope, set_id, train_resample="", test_resample=""):
    full, selected = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for row in full_rows:
        full[row["tcr_cdr3_beta"]][int(row["label"])] += 1
    for row in rows:
        selected[row["tcr_cdr3_beta"]][int(row["label"])] += 1
    detail = []
    for tcr in sorted(full):
        n0, p0 = full[tcr]
        n, p = selected[tcr]
        detail.append({"scope": scope, "set_id": set_id, "train_resample_id": train_resample,
                       "test_resample_id": test_resample, "tcr_cdr3_beta": tcr,
                       "full_positive_degree": p0, "full_negative_degree": n0,
                       "retained_positive_degree": p, "retained_negative_degree": n,
                       "removed_positive_degree": p0 - p, "removed_negative_degree": n0 - n,
                       "retained_degree_imbalance": p - n, "retained_tcr": p + n > 0})
    retained = [row for row in detail if row["retained_tcr"]]
    summary = {"scope": scope, "set_id": set_id, "train_resample_id": train_resample,
               "test_resample_id": test_resample, "full_tcr_count": len(full),
               "retained_tcr_count": len(retained), "deleted_tcr_count": len(full) - len(retained),
               "tcr_degree_mismatch_count": sum(row["retained_degree_imbalance"] != 0 for row in retained),
               "total_absolute_tcr_degree_imbalance": sum(abs(row["retained_degree_imbalance"]) for row in retained),
               "maximum_absolute_tcr_degree_imbalance": max(abs(row["retained_degree_imbalance"]) for row in retained),
               "retained_tcr_degrees_all_equal": all(row["retained_degree_imbalance"] == 0 for row in retained),
               "postdeletion_both_side_degree_theorem_reasserted": False}
    return summary, detail


def ensemble(member_rows):
    expected = {f"{a}__{f}" for a in RESAMPLES for f in FOLDS}
    if set(member_rows) != expected:
        raise ValueError("exact 25 distinct external member slots required")
    ordered = sorted(expected)
    reference = member_rows[ordered[0]]
    keys = [identity(row) for row in reference]
    for member in ordered:
        if [identity(row) for row in member_rows[member]] != keys:
            raise ValueError("external member ordered record universe differs")
    # Member-specific model identity/logits do not describe an ensemble.
    return [{**{key: row[key] for key in ("record_id", "tcr_cdr3_beta", "epitope", "hla", "interaction_key", "query_id", "label") if key in row},
             "ensemble_member_count": 25, "interaction_probability": math.fsum(
                float(member_rows[member][i]["interaction_probability"]) for member in ordered) / 25,
             "additive_probability": math.fsum(
                float(member_rows[member][i]["additive_probability"]) for member in ordered) / 25}
            for i, row in enumerate(reference)]


def mean_cells(cells, set_id):
    if len(cells) != 25 or {(x["train_resample_id"], x["test_resample_id"]) for x in cells} != {
            (a, b) for a in RESAMPLES for b in RESAMPLES}:
        raise ValueError("25 distinct cells required")
    count_keys = ("query_count", "positive_count", "negative_count", "record_count", "singleton_positive_query_count")
    if any(len({cell[field] for cell in cells}) != 1 for field in count_keys):
        raise ValueError("cell query/label coverage differs")
    return {"scope": "internal", "set_id": set_id, "cell_count": 25,
            **{field: cells[0][field] for field in count_keys},
            **{field: mean(cell[field] for cell in cells) for field in FIELDS},
            "aggregation": "equal query weights within five-fold OOF cell, then equal 25-cell weights",
            "evidence_status": EVIDENCE}


def run(index_path, contract_path, gate_path, masks_path, historical_path, output):
    contract = json.loads(Path(contract_path).read_text())
    gate = json.loads(Path(gate_path).read_text())
    if gate.get("status") != "PASS_H1_ALL75_GRAPH_AND25_MEMBER_ADMISSION" or gate.get("metric_contract_sha256") != sha(contract_path):
        raise PermissionError("root all75 graph/all25 member admission and frozen metric contract required")
    index = json.loads(Path(index_path).read_text())
    if index.get("status") != "PASS_H1_EXACT125_INTERNAL_AND25_EXTERNAL_PREDICTIONS":
        raise PermissionError("closed scoring receipt required")
    if index.get("metric_contract_sha256") != sha(contract_path) or index.get("root_gate_sha256") != sha(gate_path):
        raise PermissionError("prediction index must bind same frozen contract and root gate")
    expected_tasks = {f"{a}__{b}__{f}" for a in RESAMPLES for b in RESAMPLES for f in FOLDS}
    if len(index["internal"]) != 125 or {entry["task_id"] for entry in index["internal"]} != expected_tasks or len(index["external"]) != 25:
        raise ValueError("incomplete/duplicate prediction index slots")
    output = Path(output)
    if output.exists():
        raise FileExistsError("exclusive new metrics output root required")
    masks = read_csv(masks_path)
    if sha(masks_path) != contract["fixed_masks"]["sha256"]:
        raise ValueError("fixed masks hash drift")
    removed, mask_universes = {}, {}
    for mask_id in MASK_IDS:
        for scope, count in [("internal", 168), ("external", 141)]:
            subset = [row for row in masks if row["mask_id"] == mask_id and row["scope"] == scope]
            if len(subset) != count or len({row["query_id"] for row in subset}) != count:
                raise ValueError("mask universe incomplete or duplicate")
            removed[(mask_id, scope)] = {row["query_id"] for row in subset if row["remove_complete_query"] == "True"}
            mask_universes[(mask_id, scope)] = {row["query_id"] for row in subset}
    internal = defaultdict(list)
    for item in index["internal"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("internal prediction hash drift")
        rows = read_csv(item["path"])
        if any(row["fold_id"] != item["fold_id"] for row in rows):
            raise ValueError("fold identity differs")
        internal[(item["train_resample_id"], item["test_resample_id"])].extend(rows)
    external_members = {}
    for item in index["external"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("external prediction hash drift")
        external_members[item["member_id"]] = read_csv(item["path"])
    external = ensemble(external_members)
    if (len(external), sum(int(row["label"]) for row in external), len({row["query_id"] for row in external})) != (1612, 806, 141):
        raise ValueError("original external universe changed")
    cell_summaries, query_metrics, aggregate, coverage, degrees = [], [], [], [], []
    for set_id in ("full", *MASK_IDS):
        selected_cells = []
        for (train, test), full in sorted(internal.items()):
            if (len(full), sum(int(row["label"]) for row in full), len({row["query_id"] for row in full})) != (20578, 10289, 168):
                raise ValueError("original internal cell universe changed")
            per_query_folds = defaultdict(set)
            for row in full:
                per_query_folds[row["query_id"]].add(row["fold_id"])
            if {row["fold_id"] for row in full} != set(FOLDS) or any(len(v) != 1 for v in per_query_folds.values()):
                raise ValueError("query-disjoint five-fold OOF violation")
            if set_id != "full" and set(per_query_folds) != mask_universes[(set_id, "internal")]:
                raise ValueError("fixed internal mask query universe differs")
            rows = full if set_id == "full" else [row for row in full if row["query_id"] not in removed[(set_id, "internal")]]
            summary, queries = summarize(rows, "internal", set_id, train, test)
            if set_id != "full" and summary["query_count"] != contract["fixed_masks"]["expected_retained_queries"][set_id]["internal"]:
                raise ValueError("fixed internal mask retained query count differs")
            degree_summary, degree_detail = degree_audit(rows, full, "internal", set_id, train, test)
            coverage.append({**{k: summary[k] for k in ("scope", "set_id", "train_resample_id", "test_resample_id", "query_count", "positive_count", "negative_count", "record_count", "singleton_positive_query_count")},
                             "removed_complete_query_count": 168 - summary["query_count"],
                             "query_coverage_fraction": summary["query_count"] / 168, **degree_summary})
            selected_cells.append(summary)
            cell_summaries.append(summary)
            query_metrics.extend(queries)
            degrees.extend(degree_detail)
        aggregate.append(mean_cells(selected_cells, set_id))
        rows = external if set_id == "full" else [row for row in external if row["query_id"] not in removed[(set_id, "external")]]
        if set_id != "full" and {row["query_id"] for row in external} != mask_universes[(set_id, "external")]:
            raise ValueError("fixed external mask query universe differs")
        summary, queries = summarize(rows, "external", set_id)
        if set_id != "full" and summary["query_count"] != contract["fixed_masks"]["expected_retained_queries"][set_id]["external"]:
            raise ValueError("fixed external mask retained query count differs")
        degree_summary, degree_detail = degree_audit(rows, external, "external", set_id)
        coverage.append({**{k: summary[k] for k in ("scope", "set_id", "train_resample_id", "test_resample_id", "query_count", "positive_count", "negative_count", "record_count", "singleton_positive_query_count")},
                         "removed_complete_query_count": 141 - summary["query_count"],
                         "query_coverage_fraction": summary["query_count"] / 141, **degree_summary})
        aggregate.append({**summary, "cell_count": 1, "aggregation": "math.fsum/25 member probabilities per record before query-equal metrics"})
        query_metrics.extend(queries)
        degrees.extend(degree_detail)
    # One locked six-summary file: all three fixed masks for both scopes.
    sensitivities = [row for row in aggregate if row["set_id"] != "full"]
    if len(sensitivities) != 6:
        raise ValueError("all six source-sensitivity summaries required")
    historical = json.loads(Path(historical_path).read_text())
    comparison = []
    for row in aggregate:
        if row["set_id"] == "full":
            for metric in ("interaction_AP", "interaction_AUROC", "delta_Additive_AP", "delta_Additive_AUROC"):
                old = historical[row["scope"]][metric]
                comparison.append({"scope": row["scope"], "metric": metric, "R02_corrected_historical": old,
                                   "H1_repaired": row[metric], "H1_minus_R02_historical": row[metric] - old,
                                   "evidence_status": EVIDENCE})
    output.mkdir(parents=False)
    write_json(output / "primary_summary.json", [row for row in aggregate if row["set_id"] == "full"])
    # CSV writer uses a common schema for internal/external aggregate rows.
    columns = sorted(set().union(*(row.keys() for row in sensitivities)))
    write_csv(output / "six_fixed_source_sensitivity_summaries.csv", [{k: row.get(k, "") for k in columns} for row in sensitivities])
    write_csv(output / "cell_metrics.csv", cell_summaries)
    write_csv(output / "per_query_metrics.csv", query_metrics)
    write_csv(output / "coverage_and_deletion_audit.csv", coverage)
    write_csv(output / "actual_tcr_degree_audit.csv", degrees)
    write_csv(output / "vs_R02_corrected_historical.csv", comparison)
    write_csv(output / "external_25_member_ensemble_predictions.csv", external)
    write_json(output / "metric_execution_receipt.json", {
        "status": "PASS_FIXED_H1_DESCRIPTIVE_METRICS", "input_index_sha256": sha(index_path),
        "metric_contract_sha256": sha(contract_path), "root_gate_sha256": sha(gate_path),
        "fixed_masks_sha256": sha(masks_path), "historical_sha256": sha(historical_path),
        "source_sensitivity_summary_count": 6, "internal_cells": 25,
        "inference_status": "NOT_INFERENTIAL", "new_CI_count": 0, "new_P_count": 0,
        "fresh_blind_validation": False, "old_R03_exclusions_reapplied": False,
        "outputs": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(output.iterdir())],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("index", "contract", "gate", "masks", "historical", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    run(args.index, args.contract, args.gate, args.masks, args.historical, args.output)
