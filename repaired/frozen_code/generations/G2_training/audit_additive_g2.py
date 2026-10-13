"""R04 G2 pure Additive: exact admission then independent analytic artifacts.

No optimizer is called here. R01's hash-verified admission implementation is
reused on the new G2 training rows and actual train-only feature matrix.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

G2_SHA = "98bfca307c68f01f6980f3f2f0d6bf01529254cd554ffff4a253d6fc54942d35"
R01_MANIFEST_SHA = "5c0b0ab330f0c3b97d218846fd5fbf23b6aed2a2dc8504fa0b9f523742620ce8"
R01_CODE_SHA = "3bfcaab5cb923c300637e83100307c47c8b55826f675587a56bb783d3ae4f744"
R01_STATIC_SHA = "fe08f78dd0fc2cfc1d553979ff115a0ff53678f7cfb75d89be0896ebcf17b2d9"
PLAN_SHA = "56b8d6d745528a6a7daa185186ab48faebf1f647296815104441c9db754b3713"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def ref(path, root):
    return {"path": str(Path(path).relative_to(root)), "bytes": Path(path).stat().st_size, "sha256": sha(path)}


def task_key(task):
    return task["task_id"]


def dependencies(work, g2, source, r01):
    """Verify all static identities before importing any source module."""
    checks = {}
    checks["g2_manifest_identity"] = sha(g2 / "g2_manifest.json") == G2_SHA
    checks["frozen_plan_identity"] = sha(g2 / "authority/revision_plan_20261008_v1_1.md") == PLAN_SHA
    checks["r01_manifest_identity"] = sha(r01 / "r01_manifest.json") == R01_MANIFEST_SHA
    checks["r01_admission_implementation_identity"] = sha(r01 / "r01_audit.py") == R01_CODE_SHA
    checks["r01_static_theorem_identity"] = sha(r01 / "static_feature_objective_audit.json") == R01_STATIC_SHA
    static = json.loads((r01 / "static_feature_objective_audit.json").read_text())
    checks["r01_static_status"] = static["status"] == "PASS_STATIC_FEATURE_OBJECTIVE_CONTRACT"
    code = []
    for relative, item in static["feature_code_contracts"].items():
        actual = sha(source / relative)
        code.append({"path": relative, "expected_sha256": item["sha256"], "sha256": actual, "match": actual == item["sha256"]})
    checks["all_original_feature_factory_code_identical"] = bool(code) and all(v["match"] for v in code)
    mapping_path, contract_path = work / "authority/task_mapping105.json", work / "authority/model_contract.json"
    mapping, contract = json.loads(mapping_path.read_text()), json.loads(contract_path.read_text())
    checks["task_mapping_contract_identity"] = mapping["model_contract_sha256"] == sha(contract_path)
    checks["contract_frozen_G2_plan_and_model_identity"] = (contract["status"] == "FIXED_BEFORE_FITS" and contract["g2_manifest"]["sha256"] == G2_SHA and contract["plan_sha256"] == PLAN_SHA and contract["model_seed"] == 13 and contract["model_params"] == contract["actual_estimator_params"])
    config_identity = [{"path": item["relative_path"], "expected_sha256": item["sha256"], "match": sha(source / item["relative_path"]) == item["sha256"]} for item in contract["config_files"]]
    checks["all_original_config_identities"] = bool(config_identity) and all(v["match"] for v in config_identity)
    tasks = mapping["tasks"]
    expected = {f"S{s}__fold_{f}__draw_{d}" for s in range(7) for f in range(1, 6) for d in range(1, 4)}
    checks["closed_105_unique_task_keys"] = len(tasks) == 105 and {task_key(t) for t in tasks} == expected
    checks["model_seed13_all_tasks"] = all(t["model_seed"] == 13 for t in tasks)
    files = {v["path"]: v for v in json.loads((g2 / "g2_manifest.json").read_text())["files"]}
    input_identity = []
    for task in tasks:
        for partition in ("train", "validation", "test"):
            for kind in ("positive", "negative"):
                item = task["roles"][partition][kind]
                relative = item["relative_path"]
                path = g2 / relative
                registered = files.get(relative)
                ok = registered is not None and Path(item["path"]) == Path(mapping["g2_root"]) / relative
                ok = ok and all(item[k] == registered[k] for k in ("sha256", "bytes"))
                ok = ok and path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
                input_identity.append({"task_id": task_key(task), "partition": partition, "kind": kind, "path": relative, "sha256": item["sha256"], "bytes": item["bytes"], "match": bool(ok)})
    checks["all_630_task_input_references_bound_only_to_G2"] = len(input_identity) == 630 and all(v["match"] for v in input_identity)
    if not all(checks.values()):
        raise ValueError("Additive static admission blocked: " + repr({k: v for k, v in checks.items() if not v}))
    return mapping, contract, {"checks": checks, "source_code": code, "config_identity": config_identity, "input_identity": input_identity}


def load_api(source, r01):
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("r01_reused_admission", r01 / "r01_audit.py")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    from src.degree_matched_additive_logistic import additive_feature_dimensions, build_additive_feature_matrix
    from src.interaction_resample_robustness import _model_from_config
    from src.kmer_logistic import fit_hla_mapping
    return api, additive_feature_dimensions, build_additive_feature_matrix, fit_hla_mapping, _model_from_config


def model_params_from_contract(contract):
    """The authority stores every resolved sklearn parameter under model_params."""
    return contract["model_params"]


def read_partition(task, partition, g2):
    rows, audit = [], {}
    for kind, label in (("positive", 1), ("negative", 0)):
        item = task["roles"][partition][kind]
        path = g2 / item["relative_path"]
        if sha(path) != item["sha256"]:
            raise ValueError("input changed since admission precheck")
        opener = gzip.open if path.suffix == ".gz" else Path.open
        with opener(path, mode="rt", newline="", encoding="utf-8-sig") as f:
            data = list(csv.DictReader(f))
        if not data or len(data) != item["rows"] or not all(str(r["label"]) in (str(label), str(float(label))) for r in data):
            raise ValueError("input label does not match its positive/negative role")
        rows.extend({**r, "label": label} for r in data)
        audit[kind] = {**item, "rows": len(data)}
    return rows, audit


def write_nodes(path, nodes):
    with Path(path).open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(nodes[0]))
                writer.writeheader()
                writer.writerows(nodes)


def prepare(work, g2, source, r01):
    mapping, contract, evidence = dependencies(work, g2, source, r01)
    import sklearn
    import numpy as np
    import scipy
    import joblib
    api, dimensions, builder, fit_hla_mapping, factory = load_api(source, r01)
    expected = model_params_from_contract(contract)
    explicit = {"estimator": "sklearn.linear_model.LogisticRegression", "penalty": "l2", "C": 1.0, "class_weight": None, "max_iter": 5000, "solver": "liblinear", "random_state_source": "seed"}
    observed = factory(explicit, 13).get_params(deep=False)
    if observed != expected or sklearn.__version__ != "1.7.2":
        raise ValueError("resolved original model factory/environment mismatch")
    synthetic = [
        {"tcr_cdr3_beta": "CASSA", "epitope": "GILG", "hla": "HLA-A*02:01", "label": 1},
        {"tcr_cdr3_beta": "CASSC", "epitope": "NLVP", "hla": "HLA-B*07:02", "label": 1},
        {"tcr_cdr3_beta": "CASSA", "epitope": "NLVP", "hla": "HLA-B*07:02", "label": 0},
        {"tcr_cdr3_beta": "CASSC", "epitope": "GILG", "hla": "HLA-A*02:01", "label": 0},
    ]
    synthetic_matrix = builder(synthetic, fit_hla_mapping(synthetic))
    gates = {"identity": True, "train_only": True, "static_feature_contract": True, "actual_weight_rule_verified": True, "actual_fit_call_verified": True}
    regressions = {}
    baseline, _ = api.admit(synthetic, [1] * 4, synthetic_matrix, "additive_only", observed, gates)
    regressions["exact_balanced_pure_marginal_admitted"] = baseline["status"] == "PASS"
    broken = [dict(r) for r in synthetic]; broken[-1]["tcr_cdr3_beta"] = "CASSD"
    result, _ = api.admit(broken, [1] * 4, builder(broken, fit_hla_mapping(broken)), "additive_only", observed, gates)
    regressions["broken_TCR_degree_denied"] = result["gates"]["tcr_counts"] == "FAIL"
    broken = [dict(r) for r in synthetic]; broken[-1]["hla"] = "HLA-C*01:01"
    result, _ = api.admit(broken, [1] * 4, builder(broken, fit_hla_mapping(broken)), "additive_only", observed, gates)
    regressions["same_peptide_broken_full_HLA_query_denied"] = result["gates"]["complete_query_counts"] == "FAIL"
    result, _ = api.admit(synthetic, [2, 1, 1, 1], synthetic_matrix, "additive_only", observed, gates)
    regressions["broken_exact_effective_weight_denied"] = result["gates"]["effective_weights"] == "FAIL"
    broken_matrix = synthetic_matrix.tolil(); broken_matrix[2, 0] += 0.1
    result, _ = api.admit(synthetic, [1] * 4, broken_matrix.tocsr(), "additive_only", observed, gates)
    regressions["pair_dependent_marginal_feature_denied"] = result["gates"]["separable_features"] == "FAIL"
    for field, value, gate in (("fit_intercept", False, "actual_l2_and_intercept"), ("C", 0, "actual_l2_and_intercept"), ("class_weight", "balanced", "actual_l2_and_intercept")):
        result, _ = api.admit(synthetic, [1] * 4, synthetic_matrix, "additive_only", {**observed, field: value}, gates)
        regressions["unsupported_" + field + "_denied"] = result["gates"][gate] == "FAIL"
    for name in ("identity", "train_only"):
        result, _ = api.admit(synthetic, [1] * 4, synthetic_matrix, "additive_only", observed, {**gates, name: None})
        regressions["missing_" + name + "_unverified"] = result["gates"][name] == "UNVERIFIED"
    if not all(regressions.values()):
        raise ValueError("Additive admission rejection regression failure")
    protocol = {
        "schema_version": 1, "status": "FIXED_BEFORE_NEW_ADDITIVE_ADMISSION", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "105 pure Additive task-specific analytic artifacts, no optimization and no performance/CI/P statistics",
        "implementation": ref(Path(__file__), work),
        "shared_transform_implementation": ref(work / "audit_additive_shared_transform.py", work),
        "fixed_launcher": ref(work / "audit_additive_launch.py", work),
        "prior_prepare_only_attempt": "additive/attempt1/protocol.json; absolute single-directory G2 mount, superseded before any new-task admission/artifact by /g2 relative-path routing",
        "authorities": {"task_mapping": ref(work / "authority/task_mapping105.json", work), "model_contract": ref(work / "authority/model_contract.json", work),
                        "G2_manifest_sha256": G2_SHA, "plan_sha256": PLAN_SHA, "R01_manifest_sha256": R01_MANIFEST_SHA,
                        "R01_admission_code_sha256": R01_CODE_SHA, "R01_static_theorem_sha256": R01_STATIC_SHA},
        "precheck_evidence": evidence, "actual_params_from_original_factory": observed,
        "environment": {"sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__},
        "admission_rejection_regressions": regressions,
        "nine_gates": ["binary_labels_and_full_query", "tcr_counts", "complete_query_counts", "class_counts", "effective_weights", "separable_features", "train_only", "actual_l2_and_intercept", "identity"],
        "weights": "All effective weights exactly 1; the target uses no sample_weight and class_weight=None. Recipe allocation weights are not fit weights.",
        "feature_transform": "Original 842+HLA pure marginal builder; per-sequence normalized 1/2mers, raw lengths, train-only HLA one-hot; no scaler or interaction columns.",
        "feature_semantic_checks": "Exact repeated-endpoint sparse block equality on each actual training matrix plus label/metadata stripping invariance and original static source identity.",
        "objective": "J(theta)=sum_i(logaddexp(0,z_i)-y_i*z_i)+||theta||^2/(2C),theta=(w,b/s),z=Xw+s*theta_intercept,finite C>0 and s>0.",
        "proof": "Exact endpoint/class effective-weight cancellation makes gradient zero; Hessian>=I/C in theta, so zero is the unique global minimizer.",
        "construction_policy": "Audit all105 first. Construct independent new artifacts only if every task has all nine PASS gates and extra semantic checks PASS. Any FAIL/UNVERIFIED writes admission evidence but zero artifacts.",
        "prohibited": ["Additive numerical fit", "rounding or clipping scores", "forcing a failed task to zero", "zeroing an interaction model additive block", "performance/effect/CI/P statistics"],
        "artifact_kind": "analytic_balanced_marginal_l2_logistic_v1", "artifact_parent": "new G2 task identity, not any historical checkpoint",
        "HLA_shared_hash": "sha256(json.dumps(mapping,sort_keys=True,separators=(',',':'),allow_nan=False).encode()); root interaction runner must record the same hash per task",
    }
    write_json(work / "additive/protocol.json", protocol)
    return protocol


def run(work, g2, source, r01):
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    mapping, contract, precheck = dependencies(work, g2, source, r01)
    protocol_path = work / "additive/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["implementation"]["sha256"] != sha(Path(__file__)):
        raise ValueError("Additive implementation changed after protocol freeze")
    for name, path in (("task_mapping", work / "authority/task_mapping105.json"), ("model_contract", work / "authority/model_contract.json")):
        if sha(path) != protocol["authorities"][name]["sha256"]:
            raise ValueError("authority changed after Additive protocol freeze")
    api, dimensions_fn, builder, fit_hla_mapping, factory = load_api(source, r01)
    expected_params = model_params_from_contract(contract)
    out = work / "additive"
    for directory in ("admissions", "node_counts", "models"):
        (out / directory).mkdir(exist_ok=False)
    admissions, table, prepared = [], [], {}
    for task in mapping["tasks"]:
        task_id = task_key(task)
        rows, input_audit = read_partition(task, "train", g2)
        hla = fit_hla_mapping(rows)
        matrix = builder(rows, hla)
        actual_params = LogisticRegression(**expected_params).get_params(deep=False)
        params_match = actual_params == protocol["actual_params_from_original_factory"] == expected_params
        stripped = [{k: row[k] for k in ("tcr_cdr3_beta", "epitope", "hla")} for row in rows]
        stripped_matrix = builder(stripped, hla)
        invariance = (matrix != stripped_matrix).nnz == 0
        train_only_hla = hla == fit_hla_mapping(stripped)
        dimensions = dimensions_fn(hla)
        semantic = {"builder_dimension_matches_metadata": matrix.shape == (len(rows), dimensions["total"]),
                    "interaction_feature_count_zero": dimensions["interaction"] == 0,
                    "label_and_pair_metadata_stripping_invariance": bool(invariance),
                    "HLA_schema_fit_on_actual_train_only": bool(train_only_hla),
                    "model_params_match_frozen_original_factory": bool(params_match),
                    "all_training_input_labels_match_roles": True}
        evidence = {"identity": bool(all(precheck["checks"].values()) and all(semantic.values())),
                    "train_only": bool(train_only_hla), "static_feature_contract": bool(all(v["match"] for v in precheck["source_code"])),
                    "actual_weight_rule_verified": actual_params["class_weight"] is None and protocol["weights"].startswith("All effective weights exactly 1"),
                    "actual_fit_call_verified": bool(params_match)}
        admission, nodes = api.admit(rows, [1] * len(rows), matrix, "additive_only", actual_params, evidence)
        nodepath = out / "node_counts" / (task_id + ".csv.gz")
        write_nodes(nodepath, nodes)
        item = {"task_id": task_id, "recipe_id": task["recipe_id"], "fold_id": task["fold_id"], "draw_id": task["draw_id"],
                "admission": admission, "semantic_checks": semantic, "actual_params": actual_params,
                "train_inputs": input_audit, "model_fit_partitions": ["train"], "feature_fit_partitions": ["train"],
                "feature_dimensions": dimensions, "hla_mapping": hla, "hla_mapping_sha256": canonical_sha(hla),
                "complete_input_task_sha256": task["complete_input_task_sha256"],
                "node_counts": ref(nodepath, work), "protocol_sha256": sha(protocol_path),
                "effective_weight_rule": "exact unit weights; no sample_weight; class_weight=None",
                "target_objective": protocol["objective"], "analytic_optimizer_fit_performed": False,
                "not_admission_evidence": "0.5 predictions, small coefficients or approximate floating gradients are never used for admission"}
        admission_path = out / "admissions" / (task_id + ".json")
        write_json(admission_path, item)
        admissions.append({"task_id": task_id, "path": str(admission_path.relative_to(work)), "sha256": sha(admission_path), "status": admission["status"], "semantic_checks_pass": all(semantic.values()), "hla_mapping_sha256": canonical_sha(hla)})
        table.append({"task_id": task_id, "status": admission["status"], **admission["gates"], "train_positive": admission["balance"]["positive_rows"], "train_negative": admission["balance"]["negative_rows"], "TCR_count_violations": admission["balance"]["tcr"]["count_violations"], "full_query_count_violations": admission["balance"]["full_query"]["count_violations"], "TCR_weight_violations": admission["balance"]["tcr"]["weight_violations"], "full_query_weight_violations": admission["balance"]["full_query"]["weight_violations"], "reason": ";".join(admission["reasons"])})
        prepared[task_id] = (task, item)
        print(json.dumps({"task_id": task_id, "admission": admission["status"]}), flush=True)
    all_pass = len(admissions) == 105 and all(v["status"] == "PASS" and v["semantic_checks_pass"] for v in admissions)
    admission_summary = {"status": "PASS_ALL105_NEW_TRAIN_ADDITIVE_ADMISSION" if all_pass else "BLOCKED_NO_ANALYTIC_ARTIFACTS", "task_count": len(admissions), "status_counts": dict(Counter(v["status"] for v in admissions)), "artifacts_constructed_at_admission_freeze": 0, "protocol_sha256": sha(protocol_path), "admissions": admissions}
    write_json(out / "admission_index.json", admission_summary)
    with (out / "admission_table.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    index = []
    if all_pass:
        for task_id, (task, item) in prepared.items():
            # Fresh object, not a historical/interaction checkpoint and no fit.
            parent = {"estimator": LogisticRegression(**expected_params), "method": "additive_only", "fold_id": task["fold_id"], "recipe_id": task["recipe_id"], "draw_id": task["draw_id"], "seed": task["model_seed"], "task_id": task_id,
                      "hla_mapping": item["hla_mapping"], "feature_dimensions": item["feature_dimensions"], "model_fit_partitions": ["train"], "feature_fit_partitions": ["train"], "input_audit": {"train": item["train_inputs"]}, "model_params": expected_params}
            metadata = {"generation": "r04_training_g2_20261009", "parent_kind": "new_G2_task_not_historical_checkpoint", "task_id": task_id, "G2_manifest_sha256": G2_SHA, "plan_sha256": PLAN_SHA, "complete_input_task_sha256": task["complete_input_task_sha256"], "protocol_sha256": sha(protocol_path), "admission_index_sha256": sha(out / "admission_index.json"), "admission_record_sha256": sha(work / next(a["path"] for a in admissions if a["task_id"] == task_id)), "training_inputs": item["train_inputs"], "hla_mapping_sha256": item["hla_mapping_sha256"], "R01_admission_implementation_sha256": R01_CODE_SHA, "R01_static_theorem_sha256": R01_STATIC_SHA, "effective_weight_rule": item["effective_weight_rule"], "target_objective": protocol["objective"]}
            path = out / "models" / (task_id + ".joblib")
            payload = api.analytic_artifact(path, item["admission"], parent, item["feature_dimensions"]["total"], metadata)
            payload = joblib.load(path)
            est = payload["estimator"]
            checks = {"coef_exact_zero": bool(np.all(est.coef_ == 0)), "intercept_exact_zero": bool(np.all(est.intercept_ == 0)), "zero_optimizer_iterations": bool(np.all(est.n_iter_ == 0)), "all_original_params_preserved": est.get_params(deep=False) == expected_params, "analytic_metadata": payload["analytic_solution"] is True and payload["optimizer_fit_performed"] is False, "gate_correspondence": payload["admission_gates"] == item["admission"]["gates"]}
            partition_checks = {}
            for partition in ("train", "validation", "test"):
                rows, inputs = read_partition(task, partition, g2)
                features = builder(rows, payload["hla_mapping"])
                partition_checks[partition] = {"rows": len(rows), "logit_exact_zero": bool(np.all(est.decision_function(features) == 0)), "probability_exact_half": bool(np.all(est.predict_proba(features) == 0.5)), "feature_dimension_matches": features.shape[1] == est.n_features_in_}
            novel = [{"tcr_cdr3_beta": "ACDEFGHIKLMNPQRSTVWY" * 4, "epitope": "YWVTSRQPNMLKIHGFEDCA" * 3, "hla": "HLA-R04*99:99"}]
            novel_features = builder(novel, payload["hla_mapping"])
            checks["fixed_novel_sequences_unknown_HLA_exact_half"] = bool(np.all(est.decision_function(novel_features) == 0) and np.all(est.predict_proba(novel_features) == 0.5))
            buffer = io.BytesIO(); joblib.dump(payload, buffer); cold = joblib.load(io.BytesIO(buffer.getvalue()))
            checks["memory_serialization_reload_exact_parameters"] = bool(np.array_equal(cold["estimator"].coef_, est.coef_) and np.array_equal(cold["estimator"].intercept_, est.intercept_) and cold["hla_mapping"] == payload["hla_mapping"] and np.all(cold["estimator"].predict_proba(novel_features) == 0.5))
            status = "PASS" if all(checks.values()) and all(all(v for k, v in p.items() if k != "rows") for p in partition_checks.values()) else "FAIL"
            record = {"task_id": task_id, "recipe_id": task["recipe_id"], "fold_id": task["fold_id"], "draw_id": task["draw_id"], "model_id": "additive_only", "artifact_file": str(path.relative_to(work)), "artifact_sha256": sha(path), **ref(path, work), "hla_mapping_sha256": item["hla_mapping_sha256"], "admission_record_sha256": metadata["admission_record_sha256"], "status": status, "reload_checks": checks, "partition_checks": partition_checks}
            index.append(record)
    success = all_pass and len(index) == 105 and all(v["status"] == "PASS" for v in index)
    write_json(out / "artifact_index.json", {"status": "PASS105_INDEPENDENT_ANALYTIC_ARTIFACTS" if success else "FAIL_OR_BLOCKED_ANALYTIC_ARTIFACT_RELEASE", "count": len(index), "optimizer_fits": 0, "all105_admission_pass": all_pass, "admission_index_sha256": sha(out / "admission_index.json"), "artifacts": index})
    if not success:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--g2", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--r01", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    (prepare if args.prepare else run)(args.work, args.g2, args.source, args.r01)


if __name__ == "__main__":
    main()
