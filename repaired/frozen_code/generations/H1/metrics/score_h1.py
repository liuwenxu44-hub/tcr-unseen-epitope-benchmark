"""Gated H1 prediction generation: exactly 109 internal +18 external new calls.

The 16 internal and7 external reusable predictions are copied by verified row
identity. No fit, tuning, performance calculation, simulation or inference.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import sys
from pathlib import Path

from build_metrics import FOLDS, RESAMPLES, identity, read_csv, sha, write_json


def checked(path, expected):
    path = Path(path)
    if sha(path) != expected:
        raise ValueError(f"input hash mismatch: {path}")
    return path


def core(row, label=None):
    if label is not None and "label" in row and int(row["label"]) != label:
        raise ValueError("positive/negative source label differs")
    if "interaction_key" in row:
        tcr, peptide, hla = row["interaction_key"].split("|", 2)
    else:
        tcr, peptide, hla = (row[k].strip().upper() for k in ("tcr_cdr3_beta", "epitope", "hla"))
    return {"record_id": row.get("record_id", row.get("sample_id", row.get("interaction_id", row.get("negative_id", f"{tcr}|{peptide}|{hla}")))),
            "tcr_cdr3_beta": tcr, "epitope": peptide, "hla": hla,
            "query_id": f"{peptide}|{hla}", "interaction_key": f"{tcr}|{peptide}|{hla}",
            "label": int(row["label"] if label is None else label)}


def semantic_rows(rows):
    return [(r["interaction_key"], r["query_id"], int(r["label"])) for r in rows]


def write_predictions(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    with Path(path).open("xb") as out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as handle:
            handle.write(stream.getvalue().encode())


def run(plan_path, contract_path, gate_path, output):
    # No model or real cohort is opened until all three fixed identities agree.
    gate = json.loads(Path(gate_path).read_text())
    if gate.get("status") != "PASS_H1_ALL75_GRAPH_AND25_MEMBER_ADMISSION":
        raise PermissionError("root all75 graph/all25 member gate required")
    if gate.get("metric_contract_sha256") != sha(contract_path) or gate.get("scoring_plan_sha256") != sha(plan_path):
        raise PermissionError("root gate must bind exact contract and scoring plan")
    plan = json.loads(Path(plan_path).read_text())
    if plan["status"] != "FROZEN_H1_EXACT_SCORING_PLAN":
        raise PermissionError("scoring plan must be frozen")
    output = Path(output)
    if output.exists():
        raise FileExistsError("exclusive new prediction output root required")
    tasks = plan["internal_tasks"]
    expected_tasks = {f"{train}__{test}__{fold}" for train in RESAMPLES for test in RESAMPLES for fold in FOLDS}
    if len(tasks) != 125 or {task["task_id"] for task in tasks} != expected_tasks:
        raise ValueError("exact125 distinct internal task slots required")
    if sum(task["action"] == "NEW_SCORE" for task in tasks) != 109 or sum(task["action"] == "VERIFIED_SCORE_REUSE" for task in tasks) != 16:
        raise ValueError("frozen109/16 scoring budget changed")
    members = {member["member_id"]: member for member in plan["members"]}
    expected_members = {f"{train}__{fold}" for train in RESAMPLES for fold in FOLDS}
    if len(plan["members"]) != 25 or set(members) != expected_members:
        raise ValueError("exact25 shared member slots required")
    if sum(member["external_action"] == "NEW_SCORE" for member in members.values()) != 18 or sum(member["external_action"] == "VERIFIED_SCORE_REUSE" for member in members.values()) != 7:
        raise ValueError("frozen18/7 external scoring budget changed")
    for entry in plan["source_code"]:
        checked(entry["path"], entry["sha256"])
    import joblib
    import numpy as np
    sys.path.insert(0, plan["source_root"])
    from src.degree_matched_orthogonalized_logistic import _build_feature_matrix_and_audit
    from src.iedb_external_scoring import _logistic_scores
    checkpoints = {}
    for member_id, member in members.items():
        checked(member["model_path"], member["model_sha256"])
        checked(member["additive_path"], member["additive_sha256"])
        if member["R01_admission_status"] != "PASS_R01_ANALYTIC_ADDITIVE":
            raise PermissionError("all25 paired analytic artifacts require R01 admission")
        checkpoint = joblib.load(member["model_path"])
        if checkpoint["fold_id"] != member["fold_id"] or checkpoint["resample_id"] != member["train_resample_id"]:
            raise ValueError("shared member identity mismatch")
        checkpoints[member_id] = checkpoint
    # Preserve original positive-then-negative row order for all internal tasks.
    tests = {}
    for task in tasks:
        key = (task["test_resample_id"], task["fold_id"])
        test_identity = (task["positive_path"], task["positive_sha256"], task["negative_path"], task["negative_sha256"])
        if key not in tests:
            positive = [core(row, 1) for row in read_csv(checked(task["positive_path"], task["positive_sha256"]))]
            negative = [core(row, 0) for row in read_csv(checked(task["negative_path"], task["negative_sha256"]))]
            tests[key] = (test_identity, positive + negative)
        elif tests[key][0] != test_identity:
            raise ValueError("same testdraw/fold task received different graph binding")
    external = [core(row) for row in read_csv(checked(plan["external_cohort"]["path"], plan["external_cohort"]["sha256"]))]
    if (len(external), sum(row["label"] for row in external), len({row["query_id"] for row in external})) != (1612, 806, 141):
        raise ValueError("original external cohort identity changed")
    feature_rows = [{k: row[k] for k in ("tcr_cdr3_beta", "epitope", "hla", "interaction_key", "query_id")} for row in external]
    reused_containers = {}
    for member in members.values():
        if member["external_action"] == "VERIFIED_SCORE_REUSE":
            source = member["external_reuse_source"]
            if source["path"] not in reused_containers:
                reused_containers[source["path"]] = read_csv(checked(source["path"], source["container_sha256"]))
    # Labels are attached to output after model scoring; no label is in features.
    output.mkdir(parents=False)
    (output / "internal").mkdir()
    (output / "external").mkdir()
    index = {"status": "IN_PROGRESS", "internal": [], "external": [],
             "metric_contract_sha256": sha(contract_path), "root_gate_sha256": sha(gate_path),
             "scoring_plan_sha256": sha(plan_path), "internal_new_score_calls": 0,
             "external_new_score_calls": 0, "internal_exact_reuses": 0, "external_exact_reuses": 0}
    for task in sorted(tasks, key=lambda item: item["task_id"]):
        member_id = f"{task['train_resample_id']}__{task['fold_id']}"
        member = members[member_id]
        checkpoint = checkpoints[member_id]
        rows = tests[(task["test_resample_id"], task["fold_id"])][1]
        if task["action"] == "NEW_SCORE":
            features, feature_audit = _build_feature_matrix_and_audit(rows, checkpoint["hla_mapping"])
            logits = np.asarray(checkpoint["estimator"].decision_function(features), dtype=np.float64)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            index["internal_new_score_calls"] += 1
        else:
            old_rows = read_csv(checked(task["reuse_prediction_path"], task["reuse_prediction_sha256"]))
            if semantic_rows([core(row) for row in old_rows]) != semantic_rows(rows):
                raise ValueError("internal exact reuse ordered records do not match final graph")
            logits = [float(row["logit"]) for row in old_rows]
            probabilities = [float(row["probability"]) for row in old_rows]
            index["internal_exact_reuses"] += 1
        if len(probabilities) != len(rows) or any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for p in probabilities):
            raise ValueError("invalid internal prediction vector")
        if any(not math.isfinite(float(logit)) for logit in logits):
            raise ValueError("invalid internal logit vector")
        scored = [{**row, "task_id": task["task_id"], "member_id": member_id,
                   "train_resample_id": task["train_resample_id"], "test_resample_id": task["test_resample_id"],
                   "fold_id": task["fold_id"], "logit": float(logit), "interaction_probability": float(probability),
                   "additive_probability": 0.5} for row, logit, probability in zip(rows, logits, probabilities, strict=True)]
        for row in scored:
            identity(row)
        path = output / "internal" / f"{task['task_id']}.csv.gz"
        write_predictions(path, scored)
        index["internal"].append({"path": str(path.resolve()), "sha256": sha(path), "rows": len(scored),
            "task_id": task["task_id"], "train_resample_id": task["train_resample_id"],
            "test_resample_id": task["test_resample_id"], "fold_id": task["fold_id"],
            "action": task["action"], "member_id": member_id, "model_sha256": member["model_sha256"],
            "additive_sha256": member["additive_sha256"], "positive_sha256": task["positive_sha256"],
            "negative_sha256": task["negative_sha256"], "reuse_prediction_sha256": task.get("reuse_prediction_sha256")})
    for member_id, member in sorted(members.items()):
        if member["external_action"] == "NEW_SCORE":
            logits, probabilities = _logistic_scores(Path(plan["source_root"]), {
                "path": member["model_path"], "fold_id": member["fold_id"], "method": "orthogonalized_interaction"}, feature_rows)
            index["external_new_score_calls"] += 1
        else:
            source = member["external_reuse_source"]
            old_rows = [row for row in reused_containers[source["path"]] if row["model_id"] == source["model_id"]]
            canonical = json.dumps(old_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            if hashlib.sha256(canonical.encode()).hexdigest() != source["ordered_member_rows_sha256"]:
                raise ValueError("external source member row-slice digest mismatch")
            if [row["interaction_key"] for row in old_rows] != [row["interaction_key"] for row in external]:
                raise ValueError("external exact reuse ordered records do not match original cohort")
            if any(row.get("checkpoint_sha256", member["model_sha256"]) != member["model_sha256"] for row in old_rows):
                raise ValueError("external reuse checkpoint binding mismatch")
            logits = [float(row["logit"]) for row in old_rows]
            probabilities = [float(row["probability"]) for row in old_rows]
            index["external_exact_reuses"] += 1
        if len(probabilities) != 1612 or any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for p in probabilities):
            raise ValueError("invalid external member probability vector")
        if any(not math.isfinite(float(logit)) for logit in logits):
            raise ValueError("invalid external logit vector")
        scored = [{**row, "member_id": member_id, "logit": float(logit),
                   "interaction_probability": float(probability), "additive_probability": 0.5}
                  for row, logit, probability in zip(external, logits, probabilities, strict=True)]
        path = output / "external" / f"{member_id}.csv.gz"
        write_predictions(path, scored)
        index["external"].append({"path": str(path.resolve()), "sha256": sha(path), "rows": len(scored),
            "member_id": member_id, "action": member["external_action"], "model_sha256": member["model_sha256"],
            "additive_sha256": member["additive_sha256"], "cohort_sha256": plan["external_cohort"]["sha256"],
            "reuse_prediction_sha256": member.get("external_reuse_source", {}).get("ordered_member_rows_sha256")})
    if tuple(index[field] for field in ("internal_new_score_calls", "internal_exact_reuses", "external_new_score_calls", "external_exact_reuses")) != (109, 16, 18, 7):
        raise ValueError("exact scoring budget not achieved")
    # Verify source/model/graph identities again at completion to detect drift.
    for entry in plan["source_code"]:
        checked(entry["path"], entry["sha256"])
    for member in members.values():
        checked(member["model_path"], member["model_sha256"])
        checked(member["additive_path"], member["additive_sha256"])
    for task in tasks:
        checked(task["positive_path"], task["positive_sha256"])
        checked(task["negative_path"], task["negative_sha256"])
    checked(plan["external_cohort"]["path"], plan["external_cohort"]["sha256"])
    index["status"] = "PASS_H1_EXACT125_INTERNAL_AND25_EXTERNAL_PREDICTIONS"
    index["performance_computed"] = False
    index["numeric_additive_fits"] = 0
    index["same25_members_shared_internal_external"] = True
    write_json(output / "prediction_index.json", index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("plan", "contract", "gate", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    run(args.plan, args.contract, args.gate, args.output)
