"""Bind final admitted graphs/models to frozen125/25 score slots; no scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_metrics import FOLDS, RESAMPLES, sha, write_json


def load(path):
    return json.loads(Path(path).read_text())


def run(work, source_root):
    work, source_root = Path(work), Path(source_root)
    if load(work / "structure/structural_gate.json")["status"] != "PASS_H1_ORIGINAL_STRUCTURE_ALL75":
        raise PermissionError("all75 graph gate must pass")
    if load(work / "additive/artifact_index.json")["status"] != "PASS25_ANALYTIC_ADMISSIONS_AND_RELOADS":
        raise PermissionError("all25 analytic artifact admissions required")
    model_index = load(work / "training/model_index25.json")
    model_rows = model_index["members"]
    internal_authority = load(work / "authority/internal_prediction_sources.json")
    external_authority = load(work / "authority/external_prediction_sources.json")
    if internal_authority["status"] != "PASS" or external_authority["status"] != "PASS":
        raise PermissionError("all historical prediction identities must pass")
    graphs = {(row["resample_id"], row["fold_id"], row["role"]): row for row in load(work / "structure/input_index75.json")}
    if len(graphs) != 75:
        raise ValueError("all75 graph identities required")
    original_external = {row["member_key"]: row for row in external_authority["members"]}
    members = []
    for model in model_rows:
        member_id = model["task_id"]
        old = original_external[member_id]
        if model.get("status") != "PASS":
            raise PermissionError("all25 shared models must pass admission")
        member = {"member_id": member_id, "train_resample_id": model["train_resample_id"],
                  "fold_id": model["fold_id"], "model_path": model["model_path"],
                  "model_sha256": model["model_sha256"], "additive_path": model["additive_path"],
                  "additive_sha256": model["additive_sha256"], "R01_admission_status": "PASS_R01_ANALYTIC_ADDITIVE",
                  "external_action": "NEW_SCORE" if old["action"] == "NEW_SCORE_AFTER_REFIT" else "VERIFIED_SCORE_REUSE"}
        if member["external_action"] == "VERIFIED_SCORE_REUSE":
            if member["model_sha256"] != old["checkpoint_sha256"]:
                raise ValueError("external reuse model identity changed")
            member["external_reuse_source"] = {"path": old["original_path"],
                "container_sha256": old["container_sha256"], "model_id": old["model_id"],
                "ordered_member_rows_sha256": old["ordered_member_rows_sha256"],
                "ordered_keys_sha256": old["ordered_keys_sha256"], "input_sha256": old["input_sha256"]}
        members.append(member)
    tasks = []
    member_lookup = {row["member_id"]: row for row in members}
    for original in internal_authority["tasks"]:
        graph = graphs[(original["test_resample_id"], original["fold_id"], "test")]
        task = {"task_id": original["task_id"], "train_resample_id": original["train_resample_id"],
                "test_resample_id": original["test_resample_id"], "fold_id": original["fold_id"],
                "positive_path": graph["positive"]["path"], "positive_sha256": graph["positive"]["sha256"],
                "negative_path": graph["negative"]["path"], "negative_sha256": graph["negative"]["sha256"],
                "action": "NEW_SCORE" if original["action"] == "NEW_SCORE_REQUIRED" else "VERIFIED_SCORE_REUSE"}
        if task["action"] == "VERIFIED_SCORE_REUSE":
            member = member_lookup[f"{task['train_resample_id']}__{task['fold_id']}"]
            if member["model_sha256"] != original["checkpoint_sha256"] or any(
                graph[kind]["sha256"] != original["test_sources"][kind]["sha256"] for kind in ("positive", "negative")):
                raise ValueError("exact internal reuse input/model identity changed")
            task.update({"reuse_prediction_path": original["original_path"], "reuse_prediction_sha256": original["sha256"]})
        tasks.append(task)
    if len(tasks) != 125 or sum(t["action"] == "NEW_SCORE" for t in tasks) != 109:
        raise ValueError("original125/109/16 task mapping changed")
    if len(members) != 25 or sum(m["external_action"] == "NEW_SCORE" for m in members) != 18:
        raise ValueError("original25/18/7 member mapping changed")
    cohort_sources = {(row["cohort_path"], row["cohort_sha256"]) for row in original_external.values()}
    if len(cohort_sources) != 1:
        raise ValueError("external members bind different cohorts")
    cohort_path, cohort_sha = next(iter(cohort_sources))
    sources = ["src/degree_matched_orthogonalized_logistic.py", "src/degree_matched_additive_logistic.py",
               "src/iedb_external_scoring.py", "src/cross_resample_evaluation.py",
               "src/interaction_resample_robustness.py", "src/kmer_logistic.py", "src/vocab.py", "src/metrics.py"]
    plan = {"status": "FROZEN_H1_EXACT_SCORING_PLAN", "source_root": str(source_root),
            "members": sorted(members, key=lambda row: row["member_id"]),
            "internal_tasks": sorted(tasks, key=lambda row: row["task_id"]),
            "external_cohort": {"path": cohort_path, "sha256": cohort_sha},
            "source_code": [{"path": str(source_root / path), "sha256": sha(source_root / path)} for path in sources],
            "authorities": [{"path": path, "sha256": sha(work / path)} for path in [
                "structure/structural_gate.json", "structure/input_index75.json", "additive/artifact_index.json",
                "training/model_index25.json", "authority/internal_prediction_sources.json", "authority/external_prediction_sources.json"]],
            "internal_new_score_budget": 109, "internal_reuse_budget": 16,
            "external_new_score_budget": 18, "external_reuse_budget": 7,
            "performance_computed": False}
    write_json(work / "metrics/scoring_plan.json", plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    run(args.work, args.source_root)
