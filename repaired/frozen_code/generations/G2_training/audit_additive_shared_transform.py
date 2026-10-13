"""Post-run reconciliation of independent Additive and interaction transforms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def run(work):
    taskpath = work / "authority/task_mapping105.json"
    indexpath = work / "additive/artifact_index.json"
    tasks = json.loads(taskpath.read_text())["tasks"]
    index = json.loads(indexpath.read_text())
    artifacts = {v["task_id"]: v for v in index["artifacts"]}
    rows = []
    for task in tasks:
        identifier = task["task_id"]
        rootpath = work / "execution/tasks" / task["recipe_id"] / task["fold_id"] / task["draw_id"] / "model_metadata.json"
        admissionpath = work / "additive/admissions" / (identifier + ".json")
        detail = {"task_id": identifier, "interaction_metadata_file": str(rootpath.relative_to(work)), "admission_file": str(admissionpath.relative_to(work)), "checks": {}}
        try:
            interaction = json.loads(rootpath.read_text())
            admission = json.loads(admissionpath.read_text())
            artifact = artifacts[identifier]
            checks = detail["checks"]
            checks["task_identity"] = interaction["task_id"] == admission["task_id"] == identifier
            checks["pure_Additive_all_nine_PASS"] = admission["admission"]["status"] == "PASS" and all(v == "PASS" for v in admission["admission"]["gates"].values())
            checks["new_Additive_artifact_PASS"] = artifact["status"] == "PASS" and sha(work / artifact["artifact_file"]) == artifact["artifact_sha256"]
            checks["HLA_mapping_exact_shared"] = interaction["hla_mapping"] == admission["hla_mapping"]
            checks["HLA_mapping_hash_shared_and_recomputed"] = interaction["hla_mapping_sha256"] == admission["hla_mapping_sha256"] == artifact["hla_mapping_sha256"] == canonical_sha(interaction["hla_mapping"])
            checks["params_exact_shared"] = interaction["params"] == admission["actual_params"]
            checks["interaction_feature_and_model_train_only"] = interaction["feature_fit_partitions"] == interaction["model_fit_partitions"] == ["train"]
            checks["actual_interaction_unit_fit_weights"] = interaction["sample_weight_passed"] is False and interaction["effective_weight_rule"] == "unit_no_sample_weight_class_weight_none" and interaction["params"]["class_weight"] is None
            checks["Additive_artifact_is_independent"] = artifact["model_id"] == "additive_only" and artifact["artifact_file"] != interaction["model_file"] and artifact["artifact_sha256"] != interaction["model_sha256"]
            for partition in ("train", "validation", "test"):
                source = interaction["source_inputs"][partition]
                expected = task["roles"][partition]
                for kind in ("positive", "negative"):
                    checks[partition + "_" + kind + "_same_G2_input"] = source[kind + "_file"] == expected[kind]["relative_path"] and source[kind + "_sha256"] == expected[kind]["sha256"]
            detail["interaction_metadata_sha256"] = sha(rootpath)
            detail["admission_sha256"] = sha(admissionpath)
            detail["status"] = "PASS" if all(checks.values()) else "FAIL"
        except Exception as error:
            detail["status"] = "UNVERIFIED"
            detail["error"] = type(error).__name__ + ": " + str(error)
        rows.append(detail)
    success = len(rows) == 105 and len(artifacts) == 105 and all(v["status"] == "PASS" for v in rows)
    result = {"status": "PASS105_SHARED_TRAIN_ONLY_TRANSFORMS_AND_INDEPENDENT_ARTIFACTS" if success else "FAIL_OR_UNVERIFIED_SHARED_TRANSFORMS", "task_count": len(rows), "task_mapping_sha256": sha(taskpath), "artifact_index_sha256": sha(indexpath), "implementation_sha256": sha(Path(__file__)), "scope": "Transforms, configuration and provenance only; no fitting, prediction recomputation, metrics, CI or P values", "tasks": rows}
    with (work / "additive/shared_transform_validation.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
    if not success:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    run(parser.parse_args().work)
