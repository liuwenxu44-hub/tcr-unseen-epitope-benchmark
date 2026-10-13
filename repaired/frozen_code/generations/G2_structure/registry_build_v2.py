#!/usr/bin/env python3
"""Freeze a conservative exclusion registry; never infer positive labels or match."""
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

ANOMALY_SHA = "789c267ebe1ca40204e030e04f47824ef584db4dbde043f953f002eaf90634e3"
R03_SHA = "a1d2801b9b6437b2d8126e9eedfe0c8144d761b053c33a1f453d0e801dc0015c"
MAP_SHA = "9e8c5f9a9d74b232509ddb3b2d941f9f3c29c3080cfd826ea4eb721f51f7f258"
TOOL_SHA = "abe5d0756cd0ad2ce36d86202faf73ac0db00d970e799fcc26693415320a2168"
REMOTE = "/data/lwx/proj/TCR_unseen_epitope"
ANOMALY_REL = "runs/revision_20261008_v1_1/r03_anomaly_review_20261009"
R03_REL = "runs/revision_20261008_v1_1/r03"
SOURCE_REL = "workspaces/codex-tcrue-figure4-r"
NEW_REL = "runs/revision_20261008_v1_1/r04_structure_g2_20261009"
IMAGE = "sha256:d7c8838e67a156a9ed1abe981fee56f55a3637f2af47bae7b256a3c3c7551b59"
REGISTRY_FIELDS = [
    "interaction_key", "tcr_cdr3_beta", "epitope", "hla", "peptide", "full_hla",
    "original_full_vdjdb_forbidden_registry_member", "original_vdjdb_development_registry_member",
    "vdjdb_filter_valid_annotation_count", "iedb_filter_valid_annotation_count",
    "annotation_qualification", "frozen_experimental_qualification", "frozen_assay_outcome",
    "historical_iedb_eligibility_at_original_registration", "historical_availability",
    "complete_receptor_sample_identity", "current_targeted", "current_qualification",
    "current_assay_outcome", "current_assay_positive_support", "r04_frozen_target",
    "evidence_class", "forbid_for_synthetic_negative", "forbid_reason", "rule_status",
    "r03_internal_selected_positive_member", "r03_internal_observed_interaction_member",
    "r03_internal_graph_relevant_support_member", "raw_mapping_detail_record_count",
    "raw_mapping_detail_coverage", "raw_mapping_record_ids_json",
    "qualification_source_file", "qualification_source_sha256", "qualification_field_selector",
    "source_mapping_file", "source_mapping_sha256", "source_mapping_field_selector",
    "decision_source_file", "decision_source_sha256", "decision_field_selector",
    "current_status_source_file", "current_status_source_sha256", "current_status_field_selector",
    "current_evidence_source_file", "current_evidence_source_sha256", "current_evidence_field_selector",
    "policy_file", "policy_sha256",
]


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def csv_read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def csv_gz(path, fields, rows):
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                w = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
                w.writeheader()
                w.writerows(rows)


def json_gz(path, obj):
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            zipped.write((json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode())


def identity(path, canonical):
    return {"source_file": REMOTE + "/" + canonical, "project_relative_path": canonical,
            "bytes": path.stat().st_size, "sha256": sha(path)}


def index_unique(rows, key):
    d = {r[key]: r for r in rows}
    assert len(d) == len(rows), (key, "duplicate")
    return d


def load_inputs(args):
    a, r, s = args.anomaly, args.r03, args.source
    entries = [
        ("anomaly_manifest", a / "anomaly_manifest.json", ANOMALY_REL + "/anomaly_manifest.json"),
        ("r03_manifest", r / "r03_manifest.json", R03_REL + "/r03_manifest.json"),
        ("source_mapping", r / "structure/source_mapping.json.gz", R03_REL + "/structure/source_mapping.json.gz"),
        ("qualification", a / "qualification/full_registry_qualification.csv", ANOMALY_REL + "/qualification/full_registry_qualification.csv"),
        ("decisions", a / "decision/all_candidate_decisions.csv", ANOMALY_REL + "/decision/all_candidate_decisions.csv"),
        ("current_status", a / "assay_transport_supplement/candidate_current_status.csv", ANOMALY_REL + "/assay_transport_supplement/candidate_current_status.csv"),
        ("current_evidence", a / "assay_transport_supplement/candidate_current_evidence.json", ANOMALY_REL + "/assay_transport_supplement/candidate_current_evidence.json"),
        ("current_transport_contract", a / "assay_transport_supplement/contract.json", ANOMALY_REL + "/assay_transport_supplement/contract.json"),
        ("current_transport_summary", a / "assay_transport_supplement/summary.json", ANOMALY_REL + "/assay_transport_supplement/summary.json"),
        ("qualification_input_identities", a / "qualification/qualification_input_identities.json", ANOMALY_REL + "/qualification/qualification_input_identities.json"),
        ("freeze_tool", a / "freeze_manifest.py", ANOMALY_REL + "/freeze_manifest.py"),
    ]
    inputs = {role: identity(path, canonical) for role, path, canonical in entries}
    assert inputs["anomaly_manifest"]["sha256"] == ANOMALY_SHA
    assert inputs["r03_manifest"]["sha256"] == R03_SHA
    assert inputs["source_mapping"]["sha256"] == MAP_SHA
    assert inputs["freeze_tool"]["sha256"] == TOOL_SHA
    mapping = json.load(gzip.open(r / "structure/source_mapping.json.gz", "rt"))
    # Verify every authoritative raw/source file already declared by frozen R03.
    for item in mapping["source_audit"]["input_identities"]:
        actual = identity(s / item["path"], SOURCE_REL + "/" + item["path"])
        assert actual["sha256"] == item["sha256"] and actual["bytes"] == item["bytes"]
        inputs["r03_source:" + item["path"]] = actual
    qrows = csv_read(a / "qualification/full_registry_qualification.csv")
    q = index_unique(qrows, "interaction_key")
    drows = csv_read(a / "decision/all_candidate_decisions.csv")
    d = index_unique(drows, "interaction_key")
    c = index_unique(csv_read(a / "assay_transport_supplement/candidate_current_status.csv"), "interaction_key")
    eroot = json.loads((a / "assay_transport_supplement/candidate_current_evidence.json").read_text())
    e = index_unique(eroot["keys"], "interaction_key")
    assert len(q) == 135555 and set(q) == set(mapping["registry_positive_interaction_ids"])
    assert sum(x["original_full_vdjdb_forbidden_registry_member"] == "True" for x in qrows) == 117615
    assert set(d) == set(c) == set(e) and len(d) == 196 and set(d) <= set(q)
    assert sum(x["current_positive"] == "True" for x in drows) == 141
    assert all(x["original_forbidden_registry_member"] == "False" for x in drows)
    for key, decision in d.items():
        assert decision["current_status"] == c[key]["current_status"]
        assert decision["current_positive"] == c[key]["current_positive"]
        assert decision["historical_availability"] == "UNKNOWN"
        assert decision["complete_receptor_sample_identity"] == "UNKNOWN"
        assert decision["frozen_assay_outcome"] == "UNVERIFIED_NOT_EXPORTED"
        assert decision["rule_status"] == "POSTHOC"
        support = any(x.get("current_positive") is True for x in e[key]["evidence"])
        assert support == (decision["current_positive"] == "True")
    r04 = {k for k, x in d.items() if "r04_frozen" in x["scopes"].split(";")}
    unknown25 = sorted(k for k in r04 if d[k]["current_positive"] != "True")
    assert len(r04) == 115 and len(unknown25) == 25
    return inputs, mapping, q, d, c, eroot, e, r04, unknown25


def freeze(args, target, file_roles, notes):
    cmd = [sys.executable, str(args.anomaly / "freeze_manifest.py"), "create", "--root", str(args.output),
           "--output", "registry/" + target, "--analysis-id", "R04_G2_" + target.removesuffix(".json").upper(),
           "--status", "frozen"]
    for role, path in file_roles:
        cmd.extend(["--file", role + "=" + path])
    for note in notes:
        cmd.extend(["--note", note])
    subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, str(args.anomaly / "freeze_manifest.py"), "check", "--manifest",
                    str(args.output / "registry" / target), "--root", str(args.output), "--strict",
                    "--report", str(args.output / "registry" / target.replace("manifest", "manifest_check"))],
                   check=True, stdout=subprocess.DEVNULL)


def policy(args, inputs, mapping, q, d, c, eroot, e, r04, unknown25):
    out = args.output / "registry"
    existing = sorted(x.name for x in out.iterdir())
    assert existing == [], ("Expected empty pre-policy registry", existing)
    dump(out / "registry_input_identities.json", inputs)
    rule = {
        "schema_version": 1, "artifact": "R04_G2_PREMATCH_CONSERVATIVE_FORBIDDEN_POLICY",
        "status": "FROZEN_POSTHOC_BEFORE_G2_MATCHING", "rule_status": "POSTHOC",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "User-authorized R04 g2 structural repair; parent anomaly release immutable",
        "parent_anomaly_manifest_sha256": ANOMALY_SHA, "parent_r03_manifest_sha256": R03_SHA,
        "frozen_union": {"keys": 135555, "original_vdjdb_retained": 117615, "iedb_only_added": 17940},
        "key_contract": "exact tcr_cdr3_beta|peptide|full_hla; no HLA coarsening or receptor merging",
        "registry_semantics": "Conservative synthetic-negative exclusion universe, not a universe of experimentally established positive truths.",
        "original_registry_rule": "Retain all117615 previously registered VDJdb keys without changing historical membership or asserting experimental validation.",
        "current_positive_rule": "Use only immutable anomaly supplement: exact typed T-cell assay/receptorID link, peptide and full HLA, current qualitative positive. Current model-key support does not establish original ingestion time or complete biological receptor/sample identity.",
        "unknown_rule": "Every targeted key without current positive support remains UNKNOWN; exclude conservatively because its exact frozen annotation collides. False current_positive is absence of positive support, never evidence of a negative assay.",
        "untargeted_rule": "All untargeted supplementary filter-valid annotation keys are CONSERVATIVE_ANNOTATION_UNQUALIFIED and excluded conservatively, without experimental-positive inference.",
        "unknown25_fixed_keys": unknown25,
        "unknown25_fixed_key_sha256": hashlib.sha256(("\n".join(unknown25) + "\n").encode()).hexdigest(),
        "unknown25_disposition": "EXCLUDE_FROM_G2_SYNTHETIC_NEGATIVE_CANDIDATE_UNIVERSE; retain frozen assay/historical UNKNOWN; no relabeling",
        "targeted_count": 196, "current_positive_support_count": 141, "targeted_unknown_count": 55,
        "r04_targeted_count": 115, "r04_current_positive_support_count": 90, "r04_unknown_count": 25,
        "exclusive_evidence_class_precedence": ["ORIGINAL_REGISTERED_VDJDB", "CURRENT_ASSAY_POSITIVE_SUPPORT", "CONSERVATIVE_RISK_EXCLUSION_UNKNOWN", "CONSERVATIVE_ANNOTATION_UNQUALIFIED"],
        "orthogonal_statuses_preserved": ["original registration", "frozen experimental qualification", "frozen assay outcome", "current qualification", "historical availability", "complete receptor/sample identity", "forbid reason"],
        "source_record_detail_scope": mapping["source_record_detail_scope"],
        "prohibitions": ["current network", "model training", "prediction", "performance", "threshold tuning", "frozen parent mutation", "historical UNKNOWN relabeling", "positive truth inference for annotation universe"],
        "before_matching": True, "matching_run_by_this_code": False, "performance_examined_by_this_code": False,
        "compression_contract": "UTF8 CSV; LF; sorted interaction keys; gzip filename empty and mtime0",
        "generation": {"script_sha256": sha(pathlib.Path(__file__)), "science_image": IMAGE, "network": "none", "uid": 1001, "parents": "read_only", "output": "new_root_read_write"},
    }
    dump(out / "registry_policy.json", rule)
    fields = list(next(iter(d.values()))) + ["current_assay_outcome", "evidence_class", "forbid_reason",
        "current_status_source_file", "current_status_source_sha256", "current_status_field_selector",
        "current_evidence_source_file", "current_evidence_source_sha256", "current_evidence_field_selector", "current_evidence_json"]
    epos = {x["interaction_key"]: n for n, x in enumerate(eroot["keys"])}
    rows = []
    for key in unknown25:
        row = dict(d[key]); row.update({"current_assay_outcome": "UNKNOWN", "evidence_class": "CONSERVATIVE_RISK_EXCLUSION_UNKNOWN",
            "forbid_reason": "POSTHOC_CONSERVATIVE_EXACT_ANNOTATION_RISK_CURRENT_OUTCOME_UNKNOWN"})
        for role, prefix, selector in [("current_status", "current_status", "interaction_key=" + key),
                                      ("current_evidence", "current_evidence", "/keys/" + str(epos[key]))]:
            row[prefix + "_source_file"] = inputs[role]["source_file"]
            row[prefix + "_source_sha256"] = inputs[role]["sha256"]
            row[prefix + "_field_selector"] = selector
        row["current_evidence_json"] = json.dumps(e[key], sort_keys=True, separators=(",", ":"))
        rows.append(row)
    csv_gz(out / "unknown25_crosswalk.csv.gz", fields, rows)
    dump(out / "registry_code_identity.json", {"artifact": "PREEXECUTION_REGISTRY_CODE_IDENTITY", "script_sha256": sha(pathlib.Path(__file__)), "science_image": IMAGE, "network": "none", "uid": os.getuid(), "parents": "RO", "output": "NEW_ONLY", "freeze_tool_sha256": TOOL_SHA})
    freeze(args, "policy_manifest.json", [
        ("POLICY", "registry/registry_policy.json"), ("FIXED_UNKNOWN25", "registry/unknown25_crosswalk.csv.gz"),
        ("INPUT_IDENTITIES", "registry/registry_input_identities.json"), ("CODE_IDENTITY", "registry/registry_code_identity.json"),
        ("REGISTRY_BUILD_CODE", "registry_build.py")], ["POSTHOC conservative amendment; policy and25 UNKNOWN fixed before g2 matching. No performance or fitting authorized."])
    print(json.dumps({"policy_manifest_sha256": sha(out / "policy_manifest.json"), "policy_sha256": sha(out / "registry_policy.json")}))


def build(args, inputs, mapping, q, d, c, eroot, e, r04, unknown25):
    out = args.output / "registry"
    dump(out / "registry_build_amendment_v2.json", {"status": "CODE_ONLY_SCHEMA_REPAIR_BEFORE_MATCHING", "predecessor_code_sha256": sha(args.output / "registry_build.py"), "active_code_sha256": sha(pathlib.Path(__file__)), "policy_manifest_sha256": sha(out / "policy_manifest.json"), "reason": "Initial writer emitted source_mapping_source_file and source_mapping_source_sha256 names but declared source_mapping_file/source_mapping_sha256. Version2 fixes these two provenance-column keys only.", "failed_attempt": "CSV writer rejected first data row before accepted registry; own incomplete header-only gzip preserved under registry/attempt1/forbidden_registry_partial.csv.gz after inventory; no matching or performance occurred", "scientific_policy_or_keys_changed": False, "historical_or_current_qualification_changed": False, "failed_partial_file": "registry/attempt1/forbidden_registry_partial.csv.gz", "failed_partial_sha256": sha(out / "attempt1/forbidden_registry_partial.csv.gz"), "error": "ValueError: dict contains fields not in fieldnames: source_mapping_source_file, source_mapping_source_sha256"})
    freeze(args, "build_code_manifest_v2.json", [("ACTIVE_REGISTRY_BUILD_CODE", "registry_build_v2.py"), ("CODE_ONLY_AMENDMENT", "registry/registry_build_amendment_v2.json"), ("UNCHANGED_POLICY_FREEZE", "registry/policy_manifest.json")], ["Code-only CSV provenance schema repair; policy/UNKNOWN25 fixed before any matching and unchanged."])
    subprocess.run([sys.executable, str(args.anomaly / "freeze_manifest.py"), "check", "--manifest", str(out / "policy_manifest.json"), "--root", str(args.output), "--strict"], check=True, stdout=subprocess.DEVNULL)
    assert json.loads((out / "registry_input_identities.json").read_text()) == inputs
    rule = json.loads((out / "registry_policy.json").read_text()); assert rule["unknown25_fixed_keys"] == unknown25
    policy_sha = sha(out / "registry_policy.json")
    mapped = collections.defaultdict(list)
    records = mapping["source_records"]
    for rid, item in records.items():
        assert rid == item["source_record_id"] and item["interaction_key"] in q
        mapped[item["interaction_key"]].append(rid)
    allkeys = sorted(q)
    source_index = {k: n for n, k in enumerate(mapping["registry_positive_interaction_ids"])}
    epos = {x["interaction_key"]: n for n, x in enumerate(eroot["keys"])}
    rows, classes = [], collections.Counter()
    for key in allkeys:
        raw = q[key]; target = d.get(key); original = raw["original_full_vdjdb_forbidden_registry_member"] == "True"
        positive = bool(target and target["current_positive"] == "True")
        if original:
            cls, reason = "ORIGINAL_REGISTERED_VDJDB", "RETAIN_ORIGINAL_REGISTERED_VDJDB_FORBIDDEN_KEY"
        elif positive:
            cls, reason = "CURRENT_ASSAY_POSITIVE_SUPPORT", "POSTHOC_CURRENT_EXACT_ASSAY_MODEL_KEY_POSITIVE_SUPPORT"
        elif target:
            cls, reason = "CONSERVATIVE_RISK_EXCLUSION_UNKNOWN", "POSTHOC_CONSERVATIVE_EXACT_ANNOTATION_RISK_CURRENT_OUTCOME_UNKNOWN"
        else:
            cls, reason = "CONSERVATIVE_ANNOTATION_UNQUALIFIED", "POSTHOC_CONSERVATIVE_UNTARGETED_FILTER_VALID_ANNOTATION_EXPERIMENTAL_OUTCOME_UNQUALIFIED"
        classes[cls] += 1
        tcr, pep, hla = key.split("|")
        ids = sorted(mapped.get(key, []))
        total = int(raw["vdjdb_filter_valid_annotation_count"]) + int(raw["iedb_filter_valid_annotation_count"])
        assert len(ids) <= total
        row = {x: raw[x] for x in raw if x in REGISTRY_FIELDS}
        row.update({"interaction_key": key, "tcr_cdr3_beta": tcr, "epitope": pep, "hla": hla, "peptide": pep, "full_hla": hla,
            "frozen_experimental_qualification": raw["experimental_qualification"],
            "frozen_assay_outcome": target["frozen_assay_outcome"] if target else raw["experimental_qualification"],
            "historical_availability": target["historical_availability"] if target else "UNVERIFIED_NOT_INFERRED",
            "complete_receptor_sample_identity": target["complete_receptor_sample_identity"] if target else "UNKNOWN",
            "current_targeted": bool(target), "current_qualification": c[key]["current_status"] if target else "UNTARGETED_NOT_ASSAY_VALIDATED",
            "current_assay_outcome": "POSITIVE_SUPPORT" if positive else ("UNKNOWN" if target else "NOT_TARGETED"),
            "current_assay_positive_support": positive, "r04_frozen_target": key in r04,
            "evidence_class": cls, "forbid_for_synthetic_negative": True, "forbid_reason": reason,
            "rule_status": "ORIGINAL_REGISTRATION_RETAINED" if original else "POSTHOC",
            "raw_mapping_detail_record_count": len(ids), "raw_mapping_detail_coverage": "COMPLETE_FOR_KEY" if len(ids) == total else "R03_RELEVANT_DETAIL_SUBSET_FULL_KEY_RETAINED",
            "raw_mapping_record_ids_json": json.dumps(ids, separators=(",", ":")),
            "policy_file": REMOTE + "/" + NEW_REL + "/registry/registry_policy.json", "policy_sha256": policy_sha})
        for role, prefix, selector in [("qualification", "qualification", "interaction_key=" + key),
                ("source_mapping", "source_mapping", "/registry_positive_interaction_ids/" + str(source_index[key])),
                ("decisions", "decision", "interaction_key=" + key),
                ("current_status", "current_status", "interaction_key=" + key),
                ("current_evidence", "current_evidence", "/keys/" + str(epos[key]) if target else "")]:
            is_used = role in ("qualification", "source_mapping") or target is not None
            row[prefix + ("_file" if prefix == "source_mapping" else "_source_file")] = inputs[role]["source_file"] if is_used else ""
            row[prefix + ("_sha256" if prefix == "source_mapping" else "_source_sha256")] = inputs[role]["sha256"] if is_used else ""
            row[prefix + "_field_selector"] = selector if is_used else ""
        rows.append(row)
    assert dict(classes) == {"ORIGINAL_REGISTERED_VDJDB": 117615, "CURRENT_ASSAY_POSITIVE_SUPPORT": 141, "CONSERVATIVE_RISK_EXCLUSION_UNKNOWN": 55, "CONSERVATIVE_ANNOTATION_UNQUALIFIED": 17744}
    csv_gz(out / "forbidden_registry.csv.gz", REGISTRY_FIELDS, rows)
    fields = ["interaction_key", "source_record_id", "source_database", "source_path", "raw_source_file", "raw_source_sha256", "field_selector", "source_mapping_file", "source_mapping_sha256", "source_mapping_field_selector", "source_record_json"]
    source_hash = {x["path"]: x["sha256"] for x in mapping["source_audit"]["input_identities"]}
    def raw_rows():
        for key in allkeys:
            for rid in sorted(mapped.get(key, [])):
                x = records[rid]
                yield {"interaction_key": key, "source_record_id": rid, "source_database": x["source_database"],
                    "source_path": x["source_path"], "raw_source_file": REMOTE + "/" + SOURCE_REL + "/" + x["source_path"],
                    "raw_source_sha256": source_hash[x["source_path"]], "field_selector": x["field_selector"],
                    "source_mapping_file": inputs["source_mapping"]["source_file"], "source_mapping_sha256": inputs["source_mapping"]["sha256"],
                    "source_mapping_field_selector": "/source_records/" + rid.replace("~", "~0").replace("/", "~1"),
                    "source_record_json": json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)}
    csv_gz(out / "raw_mapping_crosswalk.csv.gz", fields, raw_rows())
    json_gz(out / "current_evidence_preserved.json.gz", eroot)
    csv_gz(out / "targeted196_decision_crosswalk.csv.gz", list(next(iter(d.values()))), [d[k] for k in sorted(d)])
    keytext = "\n".join(allkeys) + "\n"
    dump(out / "registry_key_identity.json", {"key_count": len(allkeys), "key_encoding": "UTF8 sorted exact interaction_key LF including trailing LF", "key_sha256": hashlib.sha256(keytext.encode()).hexdigest(), "original_count": 117615, "supplementary_count": 17940, "original_key_sha256": hashlib.sha256(("\n".join(k for k in allkeys if q[k]["original_full_vdjdb_forbidden_registry_member"] == "True") + "\n").encode()).hexdigest()})
    dump(out / "registry_schema.json", {"registry_fields": REGISTRY_FIELDS, "raw_mapping_fields": fields, "current_assay_outcome_allowed": ["POSITIVE_SUPPORT", "UNKNOWN", "NOT_TARGETED"], "evidence_classes": dict(classes), "matching_key": "interaction_key", "matching_action": "exclude every row; forbid_for_synthetic_negative is always True", "raw_source_record_json": "Exact source-mapping record preserved with all native metadata, IDs, raw fields, and original selector; no biological reinterpretation"})
    dump(out / "registry_summary.json", {"status": "PASS_PREMATCH_REGISTRY_WITH_EXPLICIT_EVIDENCE_LIMITATIONS", "registry_keys": 135555, "original_vdjdb_retained": 117615, "iedb_only_added": 17940, "evidence_class_counts": dict(classes), "targeted": 196, "current_positive_support": 141, "targeted_unknown": 55, "r04_targeted": 115, "r04_current_positive_support": 90, "r04_unknown": 25, "raw_mapping_records_preserved": len(records), "full_raw_annotation_record_count": mapping["full_registry_positive_record_count"], "keys_with_complete_raw_mapping_detail": sum(x["raw_mapping_detail_coverage"] == "COMPLETE_FOR_KEY" for x in rows), "source_record_detail_scope": mapping["source_record_detail_scope"], "annotation_universe_is_not_positive_truth": True, "historical_unknown_preserved": True, "current_network_requests": 0, "fits": 0, "performance_read": False, "matching_performed": False, "policy_manifest_sha256": sha(out / "policy_manifest.json"), "limitations": ["Original registration is retained without inferring assay validation", "Current positive support is model-key evidence; historical ingestion and complete biological receptor/sample identity remain UNKNOWN", "Untargeted annotation records have no current assay outcome qualification", "Only R03 relevant source-record details were archived; every full registry key and full raw per-key counts retained"]})
    roles = [("FAILED_NONADMITTED_PARTIAL_HEADER_ONLY", "registry/attempt1/forbidden_registry_partial.csv.gz"), ("ORIGINAL_POLICY_GENERATION_CODE", "registry_build.py"), ("ACTIVE_REGISTRY_BUILD_CODE", "registry_build_v2.py"), ("CODE_ONLY_AMENDMENT", "registry/registry_build_amendment_v2.json"), ("BUILD_CODE_FREEZE", "registry/build_code_manifest_v2.json"), ("BUILD_CODE_FREEZE_CHECK", "registry/build_code_manifest_check_v2.json"), ("POLICY_FREEZE", "registry/policy_manifest.json"), ("POLICY_FREEZE_CHECK", "registry/policy_manifest_check.json"), ("POLICY", "registry/registry_policy.json"), ("FIXED_UNKNOWN25", "registry/unknown25_crosswalk.csv.gz"), ("INPUT_IDENTITIES", "registry/registry_input_identities.json"), ("CODE_IDENTITY", "registry/registry_code_identity.json"), ("CONSERVATIVE_FORBIDDEN_REGISTRY", "registry/forbidden_registry.csv.gz"), ("RAW_MAPPING_CROSSWALK", "registry/raw_mapping_crosswalk.csv.gz"), ("CURRENT_EVIDENCE_PRESERVED", "registry/current_evidence_preserved.json.gz"), ("TARGETED_DECISIONS_PRESERVED", "registry/targeted196_decision_crosswalk.csv.gz"), ("KEY_IDENTITY", "registry/registry_key_identity.json"), ("SCHEMA", "registry/registry_schema.json"), ("SUMMARY", "registry/registry_summary.json")]
    catalog = {"artifact": "EXPLICIT_PREMATCH_REGISTRY_CATALOG", "self_reference": "Excluded from own entries; enclosing registry_manifest freezes catalog", "files": [{"role": role, "path": path, "bytes": (args.output / path).stat().st_size, "sha256": sha(args.output / path)} for role, path in roles]}
    dump(out / "registry_catalog.json", catalog)
    roles.append(("EXPLICIT_CATALOG", "registry/registry_catalog.json"))
    freeze(args, "registry_manifest.json", roles, ["Complete135555 conservative forbidden registry frozen before g2 matching. Neither annotation eligibility nor UNKNOWN supports a positive label.", "Source map detailed records are219833 of242822 full eligible annotation records; completeness of135555 interaction keys is verified independently from detail availability."])
    print(json.dumps({"pre_match_manifest_sha256": sha(out / "registry_manifest.json"), "registry_sha256": sha(out / "forbidden_registry.csv.gz"), "classes": dict(classes)}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["policy", "build"])
    for name in ["anomaly", "r03", "source", "output"]:
        p.add_argument("--" + name, type=pathlib.Path, required=True)
    args = p.parse_args()
    assert os.getuid() == 1001, "Require frozen science container uid1001"
    data = load_inputs(args)
    (policy if args.stage == "policy" else build)(args, *data)


if __name__ == "__main__":
    main()
