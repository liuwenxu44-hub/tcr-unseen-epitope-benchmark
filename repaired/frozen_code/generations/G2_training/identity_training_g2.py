#!/usr/bin/env python3
"""R04 training intake: explicit frozen identity checks; no model/data decoding."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

G2_SHA = "98bfca307c68f01f6980f3f2f0d6bf01529254cd554ffff4a253d6fc54942d35"
PLAN_SHA = "56b8d6d745528a6a7daa185186ab48faebf1f647296815104441c9db754b3713"
RUN = Path("/data/lwx/proj/TCR_unseen_epitope/runs/revision_20261008_v1_1")
SOURCE = Path("/data/lwx/proj/TCR_unseen_epitope/workspaces/codex-tcrue-figure4-r")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def check_one(entry):
    p = Path(entry["path"])
    actual = sha(p) if p.is_file() else None
    size = p.stat().st_size if p.is_file() else None
    expected_size = entry.get("bytes")
    return {**entry, "actual_sha256": actual, "actual_bytes": size,
            "pass": actual == entry["sha256"] and
                    (expected_size is None or expected_size == size)}


def check_entries(entries):
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(check_one, entries))


def resolve_r01_file(entry):
    prefix = "results/revision_20261008_v1_1/r01/"
    assert entry["path"].startswith(prefix), entry
    return RUN / "r01" / entry["path"][len(prefix):]


def checked_copy(source, target, expected):
    assert sha(source) == expected, str(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(source.read_bytes())
    assert sha(target) == expected


def source_closure(core):
    queue = [Path(p) for p in core]
    found = set()
    while queue:
        rel = queue.pop(0)
        if rel in found or not (SOURCE / rel).is_file():
            continue
        found.add(rel)
        tree = ast.parse((SOURCE / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level == 0:
                continue
            parent = rel.parent
            for _ in range(node.level - 1):
                parent = parent.parent
            if node.module:
                dependency = parent.joinpath(*node.module.split("."))
                queue.append(dependency.with_suffix(".py"))
                init = dependency / "__init__.py"
                if (SOURCE / init).is_file():
                    queue.append(init)
            else:
                for alias in node.names:
                    queue.append((parent / alias.name).with_suffix(".py"))
    for rel in list(found):
        for parent in rel.parents:
            init = parent / "__init__.py"
            if (SOURCE / init).is_file():
                found.add(init)
    return sorted(found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--phase", choices=["before", "after"], required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    authority = work / "authority"
    g2 = RUN / "r04_structure_g2_20261009"
    manifest_path = g2 / "g2_manifest.json"
    assert sha(manifest_path) == G2_SHA
    manifest = read(manifest_path)
    assert len(manifest["files"]) == 857
    entries = [{"path": str(g2 / x["path"]), "sha256": x["sha256"],
                "bytes": x["bytes"], "authority": "G2_manifest", "relative_path": x["path"]}
               for x in manifest["files"]]
    assert len({x["relative_path"] for x in entries}) == 857
    checks = check_entries(entries)
    g2_report = {"status": "PASS" if all(x["pass"] for x in checks) else "FAIL",
                 "phase": args.phase, "created_utc": datetime.now(timezone.utc).isoformat(),
                 "manifest_path": str(manifest_path), "manifest_sha256": G2_SHA,
                 "files": 857, "failed": sum(not x["pass"] for x in checks),
                 "models_deserialized": 0, "data_rows_read": 0,
                 "matching_or_performance_recomputed": False, "checks": checks}
    write_new(authority / f"g2_manifest_{args.phase}.json", g2_report)
    assert g2_report["status"] == "PASS", "G2 drift"
    index = {x["path"]: x for x in manifest["files"]}

    def g2_read(rel):
        assert rel in index, rel
        assert sha(g2 / rel) == index[rel]["sha256"]
        return read(g2 / rel)

    protected = {}

    def protect(path, expected, size=None, origin=None):
        key = str(path)
        if key in protected:
            assert protected[key]["sha256"] == expected, key
            if protected[key].get("bytes") is None and size is not None:
                protected[key]["bytes"] = size
            protected[key]["origins"].append(origin)
        else:
            protected[key] = {"path": key, "sha256": expected, "bytes": size, "origins": [origin]}

    prerequisite = g2_read("authority/prerequisite_identity_before_v3.json")
    for x in prerequisite["checks"]:
        protect(x["path"], x["expected_sha256"], origin="G2_prerequisite_v3")
    parent = g2_read("authority/parent_authority_identity_before.json")
    for x in parent["additional_hash_size_checks"]:
        protect(x["path"], x["expected_sha256"], origin="G2_parent_authorities")
    historical = g2_read("authority/historical_protection_inventory.json")
    for x in historical["entries"]:
        protect(SOURCE / x["path"], x["sha256"], x.get("size_bytes"), "G2_historical794")
    source_identity = g2_read("authority/input_identity.json")
    for x in source_identity["entries"]:
        protect(SOURCE / x["path"], x["sha256"], x["bytes"], "G2_input_identity")

    r01_manifest = g2_read("authority/r01_manifest.json")
    static_entry = next(x for x in r01_manifest["files"] if x["path"].endswith("/static_feature_objective_audit.json"))
    static_path = resolve_r01_file(static_entry)
    assert sha(static_path) == static_entry["sha256"]
    static = read(static_path)
    assert static["status"] == "PASS_STATIC_FEATURE_OBJECTIVE_CONTRACT"
    code_hashes = {p: x["sha256"] for p, x in static["feature_code_contracts"].items()}
    for p, expected in code_hashes.items():
        protect(SOURCE / p, expected, origin="R01_static_feature_objective_audit")

    if args.phase == "after":
        prior = read(authority / "protected_identity_before.json")
        for x in prior["checks"]:
            protect(x["path"], x["sha256"], x.get("bytes"), "NEW_before_identity")
        contract = read(authority / "model_contract.json")
        for x in contract["source_files"] + contract["config_files"]:
            protect(x["path"], x["sha256"], x["bytes"], "NEW_pretraining_code_configuration")
            snapshot = authority / "source_snapshot" / x["relative_path"]
            protect(snapshot, x["sha256"], x["bytes"], "NEW_pretraining_code_snapshot")

    parent_checks = check_entries(sorted(protected.values(), key=lambda x: x["path"]))
    parent_report = {"status": "PASS" if all(x["pass"] for x in parent_checks) else "FAIL",
                     "phase": args.phase, "created_utc": datetime.now(timezone.utc).isoformat(),
                     "unique_files": len(parent_checks), "failed": sum(not x["pass"] for x in parent_checks),
                     "models_deserialized": 0, "data_rows_read": 0,
                     "checks": parent_checks}
    write_new(authority / f"protected_identity_{args.phase}.json", parent_report)
    assert parent_report["status"] == "PASS", "Parent drift"

    if args.phase == "after":
        print(json.dumps({"status": "PASS_TRAINING_AFTER_IDENTITIES", "g2_files": 857,
                          "protected_unique_files": len(parent_checks), "failed": 0}))
        return

    assert sha(g2 / "authority/revision_plan_20261008_v1_1.md") == PLAN_SHA
    for rel in ["g2_manifest.json", "authority/revision_plan_20261008_v1_1.md",
                "execution/future_task_ledger.json", "execution/final_allocations.json",
                "authority/execution_contract.json", "decision/release_decision.json"]:
        dest = authority / ("g2_manifest_parent.json" if rel == "g2_manifest.json" else Path(rel).name)
        expected = G2_SHA if rel == "g2_manifest.json" else index[rel]["sha256"]
        checked_copy(g2 / rel, dest, expected)
    checked_copy(static_path, authority / "r01_static_feature_objective_audit.json", static_entry["sha256"])

    closure = source_closure(code_hashes)
    source_files = []
    for rel in closure:
        path = SOURCE / rel
        expected = code_hashes.get(str(rel), sha(path))
        checked_copy(path, authority / "source_snapshot" / rel, expected)
        source_files.append({"path": str(path), "relative_path": str(rel), "sha256": expected,
                             "bytes": path.stat().st_size,
                             "snapshot_path": str(authority / "source_snapshot" / rel),
                             "authority": "R01_STATIC_FROZEN" if str(rel) in code_hashes else "READONLY_PRETRAINING_IMPORT_CLOSURE_CAPTURE"})
    config_files = []
    names = ["configs/degree_matched_orthogonalized_logistic_5fold.yaml",
             "configs/interaction_resample_robustness.yaml",
             "configs/degree_matched_additive_logistic_5fold.yaml",
             "configs/degree_matched_full_logistic_5fold.yaml"]
    history_index = {x["path"]: x for x in historical["entries"]}
    for rel in names:
        item = history_index[rel]
        checked_copy(SOURCE / rel, authority / "source_snapshot" / rel, item["sha256"])
        config_files.append({"path": str(SOURCE / rel), "relative_path": rel, "sha256": item["sha256"],
                             "bytes": item["size_bytes"], "authority": "G2_HISTORICAL_PROTECTION_INVENTORY"})

    params = {"C": 1.0, "class_weight": None, "dual": False, "fit_intercept": True,
              "intercept_scaling": 1, "l1_ratio": None, "max_iter": 5000,
              "multi_class": "deprecated", "n_jobs": None, "penalty": "l2",
              "random_state": 13, "solver": "liblinear", "tol": 0.0001,
              "verbose": 0, "warm_start": False}
    model_contract = {
        "schema_version": 1, "status": "FIXED_BEFORE_FITS", "model": "Orthogonalized-interaction Logistic",
        "g2_manifest": {"path": str(manifest_path), "sha256": G2_SHA},
        "plan_sha256": PLAN_SHA, "model_seed": 13, "model_params": params,
        "actual_estimator_params": params,
        "original_model_config": {"estimator":"sklearn.linear_model.LogisticRegression", "penalty":"l2",
                                  "C":1.0, "class_weight":None, "max_iter":5000, "solver":"liblinear",
                                  "random_state_source":"seed"},
        "model_constructor": "src.interaction_resample_robustness._model_from_config",
        "model_constructor_call": "_model_from_config(frozen_config['model'], 13)",
        "fit_call": "model.fit(train_features, train_labels)",
        "sample_weight": None, "effective_sample_weight": 1.0,
        "class_weight": None, "fit_count_per_task": 1, "fit_budget": 105,
        "model_fit_partitions": ["train"], "feature_fit_partitions": ["train"],
        "validation_policy": "Transform using train HLA mapping only; no hyperparameter selection, cutoff selection or early stopping.",
        "same_draw_policy": "Each task's train, validation and test use its one draw_id; no draw-by-draw crossing.",
        "analytic_additive_policy": "Separate strict zero artifact after per-task admission; no numerical fit; never alter interaction model's jointly fitted blocks.",
        "warnings_policy": "ConvergenceWarning is error; no automatic retry or parameter change.",
        "probability_policy": "Use frozen estimator predict_proba[:,1] or equivalent original sigmoid(decision_function), report exact implementation.",
        "features": {"alphabet": "ACDEFGHIKLMNPQRSTVWY", "tcr_kmers": [1,2], "epitope_kmers": [1,2],
                     "sequence_normalization": "strip().upper()", "frequency_denominator": "len(sequence)-k+1",
                     "main_sequence_features": 840, "length_features": 2, "length_scale": "raw lengths, no global scaling",
                     "interaction_features": 400, "interaction": "outer(H@tcr_1mer,H@epitope_1mer).ravel(order='C'); H=I-11.T/20",
                     "orthogonality_tolerance": 1e-12, "HLA": "train-only sorted one-hot; UNK_HLA=0",
                     "total_dimension": "1242+len(train_hla_mapping)",
                     "builder": "src.degree_matched_orthogonalized_logistic._build_feature_matrix_and_audit"},
        "L2_objective": "sum(logaddexp(0,z)-y*z)+(||w||^2+(b/intercept_scaling)^2)/(2*C)",
        "source_files": source_files, "config_files": config_files,
        "codeidentities": source_files,
        "r01_static_contract": {"path": str(static_path), "sha256": static_entry["sha256"],
                                "via": "G2 manifest -> authority/r01_manifest.json -> static_feature_objective_audit.json"},
        "environment": {"docker_image": "sha256:d7c8838e67a156a9ed1abe981fee56f55a3637f2af47bae7b256a3c3c7551b59",
                        "sklearn_version": "1.7.2", "resolved_defaults_observed_without_fit": True},
        "forbidden_executor_calls": ["train_interaction_resample_robustness", "train_degree_matched_orthogonalized_logistic_five_fold"]}
    write_new(authority / "model_contract.json", model_contract)

    ledger = g2_read("execution/future_task_ledger.json")
    allocations = g2_read("execution/final_allocations.json")
    input_results_rel = "execution/input_results.csv"
    assert sha(g2 / input_results_rel) == index[input_results_rel]["sha256"]
    with (g2 / input_results_rel).open(newline="", encoding="utf-8") as handle:
        results = {(x["recipe_id"],x["fold_id"],x["partition"],x["draw_id"]): x for x in csv.DictReader(handle)}
    assert len(results) == 315
    assert len(ledger) == 105
    tasks = []
    for item in ledger:
        recipe, fold, draw = (item[k] for k in ["recipe_id", "fold_id", "draw_id"])
        assert item["future_fit_eligible"] and item["structural_eligible"]
        assert item["future_model_seed"] == 13
        roles = {}
        source_inputs = {}
        for role in ["train", "validation", "test"]:
            value = dict(item["roles"][role])
            counts = results[(recipe,fold,role,draw)]
            assert counts["status"] == "PASS"
            for kind in ["positive", "negative"]:
                rel = value[kind + "_file"]
                selected = index[rel]
                if kind == "negative":
                    assert rel.endswith("_" + draw + ".csv.gz")
                value[kind] = {"path": str(g2 / rel), "relative_path": rel,
                               "sha256": selected["sha256"], "bytes": selected["bytes"],
                               "rows": int(counts[kind+"_count"])}
            roles[role] = value
            assert counts["negative_file"] == value["negative_file"]
            assert counts["new_negative_sha256"] == value["negative"]["sha256"]
            source_inputs[role] = {"positive_file":value["positive_file"], "positive_sha256":value["positive"]["sha256"],
                                  "negative_file":value["negative_file"], "negative_sha256":value["negative"]["sha256"],
                                  "positive_rows":int(counts["positive_count"]), "negative_rows":int(counts["negative_count"]),
                                  "negative_derived_seed":int(counts["derived_seed"])}
        allocation = allocations[recipe]
        tasks.append({"task_id": "__".join([recipe, fold, draw]), "recipe_id": recipe,
                      "fold_id": fold, "draw_id": draw, "model_seed": 13,
                      "negative_draw_base_seed": {"draw_1":13,"draw_2":1013,"draw_3":2013}[draw],
                      "complete_input_task_sha256": item["complete_input_task_sha256"],
                      "roles": roles, "source_inputs":source_inputs, "recipe_config": allocation["config"],
                      "allocation_mode": allocation["allocation_mode"],
                      "canonical_component_group_hash": allocation["identity"]["canonical_component_group_hash"],
                      "structural_eligible": True, "numerical_fit_calls_planned": 1,
                      "checkpoint_reuse": False,
                      "reuse_note": "No performance-dependent reuse; exactly one fresh interaction fit planned per unique task."})
    assert len({x["task_id"] for x in tasks}) == 105
    expected_keys = {(f"S{r}",f"fold_{f}",f"draw_{d}") for r in range(7) for f in range(1,6) for d in range(1,4)}
    assert {(x["recipe_id"],x["fold_id"],x["draw_id"]) for x in tasks} == expected_keys
    assert len({x["complete_input_task_sha256"] for x in tasks}) == 105
    mapping = {"schema_version": 1, "status": "FIXED_105_SAME_DRAW_TASKS", "g2_root": str(g2),
               "g2_manifest": {"path": str(manifest_path), "sha256": G2_SHA},
               "g2_ledger": {"path": "execution/future_task_ledger.json", "sha256": index["execution/future_task_ledger.json"]["sha256"]},
               "model_contract_sha256": sha(authority / "model_contract.json"),
               "task_count": 105, "interaction_fit_max": 105, "tasks": tasks}
    write_new(authority / "task_mapping105.json", mapping)
    fields = ["task_id","recipe_id","fold_id","draw_id","model_seed","negative_draw_base_seed",
              "complete_input_task_sha256","allocation_mode","canonical_component_group_hash"]
    with (authority / "task_mapping105.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k:x[k] for k in fields} for x in tasks])
    write_new(authority / "intake_summary.json", {
        "status": "PASS_PRETRAINING_IDENTITY_AND_FIXED_MAPPING", "g2_files_verified": 857,
        "protected_unique_files_verified": len(parent_checks), "source_snapshot_files": len(source_files),
        "task_count": 105, "interaction_fit_max": 105, "fits_executed_by_identity_script": 0,
        "scientific_recomputations": 0, "models_deserialized": 0,
        "S3_S4_actual_allocation_modes": {r:allocations[r]["allocation_mode"] for r in ["S3","S4"]},
        "S3_S4_actual_operations": {r:allocations[r]["operation"] for r in ["S3","S4"]},
        "limitations": ["Identity checks verify bytes only; frozen G2 scientific structure is adopted.",
                        "Old R02 graphs/performance were not repaired or reevaluated by this intake.",
                        "Supporting import closure is frozen now; model/feature core inherits R01 exact hashes."]})
    print(json.dumps({"status":"PASS_PRETRAINING_IDENTITY_AND_FIXED_MAPPING", "g2_files":857,
                      "protected_unique_files":len(parent_checks),"source_files":len(source_files),"tasks":105}))


if __name__ == "__main__":
    main()
