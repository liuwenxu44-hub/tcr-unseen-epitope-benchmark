"""Explicit post-execution catalogue; does not discover or alter outputs."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--work", type=Path, required=True)
work = parser.parse_args().work


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


fixed = [
    ("audit_additive_g2.py", "ADDITIVE_ADMISSION_AND_ANALYTIC_ARTIFACT_PRODUCER"),
    ("audit_additive_shared_transform.py", "POSTRUN_SHARED_TRANSFORM_RECONCILIATION_CODE"),
    ("audit_additive_launch.py", "FIXED_CONTAINER_LAUNCHER"),
    ("audit_additive_catalog.py", "EXPLICIT_OUTPUT_CATALOG_CODE"),
    ("additive/protocol.json", "FIXED_ADMISSION_PROTOCOL_AND_TEN_REJECTION_REGRESSIONS"),
    ("additive/attempt1/protocol.json", "SUPERSEDED_PREPARE_ONLY_PROTOCOL_NO_NEW_TASK_ADMISSION_OR_ARTIFACT"),
    ("additive/attempt1/audit_additive_g2.py", "SUPERSEDED_PREPARE_ONLY_CODE"),
    ("additive/prepare_stdout.log", "PREPARE_EXECUTION_STDOUT"),
    ("additive/admission_stdout.log", "REAL105_ADMISSION_AND_ARTIFACT_STDOUT"),
    ("additive/reconcile_stdout.log", "SHARED_TRANSFORM_RECONCILIATION_STDOUT"),
    ("additive/admission_index.json", "REAL105_ADMISSION_RESULT_INDEX"),
    ("additive/admission_table.csv", "REAL105_ADMISSION_NINE_GATES_TABLE"),
    ("additive/artifact_index.json", "105_ANALYTIC_ARTIFACT_AND_ACTUAL_RELOAD_RESULT_INDEX"),
    ("additive/shared_transform_validation.json", "105_SHARED_TRAIN_ONLY_TRANSFORM_RECONCILIATION_RESULT"),
]
mapping = json.loads((work / "authority/task_mapping105.json").read_text())
admissions = json.loads((work / "additive/admission_index.json").read_text())
artifacts = json.loads((work / "additive/artifact_index.json").read_text())
shared = json.loads((work / "additive/shared_transform_validation.json").read_text())
assert admissions["status"] == "PASS_ALL105_NEW_TRAIN_ADDITIVE_ADMISSION"
assert artifacts["status"] == "PASS105_INDEPENDENT_ANALYTIC_ARTIFACTS"
assert shared["status"] == "PASS105_SHARED_TRAIN_ONLY_TRANSFORMS_AND_INDEPENDENT_ARTIFACTS"
by_admission = {r["task_id"]: r for r in admissions["admissions"]}
by_artifact = {r["task_id"]: r for r in artifacts["artifacts"]}
assert len(mapping["tasks"]) == len(by_admission) == len(by_artifact) == 105
for task in mapping["tasks"]:
    identifier = task["task_id"]
    admission = by_admission[identifier]
    artifact = by_artifact[identifier]
    assert admission["status"] == artifact["status"] == "PASS"
    assert admission["path"] == f"additive/admissions/{identifier}.json"
    assert artifact["artifact_file"] == f"additive/models/{identifier}.joblib"
    assert sha(work / admission["path"]) == admission["sha256"]
    assert sha(work / artifact["artifact_file"]) == artifact["artifact_sha256"]
    detail = json.loads((work / admission["path"]).read_text())
    node = detail["node_counts"]
    assert node["path"] == f"additive/node_counts/{identifier}.csv.gz"
    assert sha(work / node["path"]) == node["sha256"]
    fixed.extend([(admission["path"], "TASK_NEW_TRAIN_ADMISSION"), (node["path"], "EXACT_TCR_FULL_QUERY_CLASS_COUNTS_AND_EFFECTIVE_WEIGHTS"), (artifact["artifact_file"], "INDEPENDENT_ANALYTIC_ADDITIVE_ARTIFACT_NO_OPTIMIZATION")])
assert len(fixed) == len({p for p, role in fixed}) == 329
files = [{"path": p, "role": role, "bytes": (work / p).stat().st_size, "sha256": sha(work / p)} for p, role in fixed]
result = {"schema_version": 1, "status": "EXPLICIT_CLOSED_ADDITIVE_OWNERSHIP_CATALOG", "file_count": len(files), "selection_method": "Fixed14 paths plus105 task-key paths each for admission/node/model; no glob or recursive discovery", "self_reference": "additive/owned_catalog.json; catalogue excludes itself to avoid circular hash", "task_mapping_sha256": sha(work / "authority/task_mapping105.json"), "scientific_scope": "New105 pure Additive exact admission and analytic artifacts; no optimizer fit, interaction coefficient mutation, metrics, effects, CI or P", "files": files}
with (work / "additive/owned_catalog.json").open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write("\n")
print(json.dumps({"status": result["status"], "file_count": len(files), "catalog_sha256": sha(work / "additive/owned_catalog.json")}))
