#!/usr/bin/env python3
"""Portable, read-only checks of the repaired H1/G2 frozen materials.

No training, sampling, tuning or inferential analysis is implemented. Only the
explicit ``score-smoke`` command evaluates a few existing prediction records.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
H1 = "generations/H1"
G2 = "generations/G2_training"
COMM = "generations/R07_selected_communication/source_data"


class UnresolvedMemberError(ValueError):
    """A provenance source is not present in the controlled package inventory."""


def portable_root(root):
    path = Path(root).resolve()
    # Python on Windows may still inherit MAX_PATH for ordinary path strings.
    # Keep root and member paths in the same namespace for containment checks.
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        text = str(path)
        path = Path("\\\\?\\UNC\\" + text[2:] if text.startswith("\\\\") else "\\\\?\\" + text)
    return path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class Materials:
    """Resolve relocated files by controlled package path or frozen SHA-256.

    Historical absolute paths are never opened. Hash lookup also finds the seven
    H1 reused checkpoints and frozen source-table copies without their old tree.
    """

    def __init__(self, root, manifest=None):
        self.root = portable_root(root)
        if manifest is None:
            candidates = [self.root / n for n in (
                "MATERIALS_MANIFEST.json", "DERIVED_FILES.json", "manifest.json")]
            manifest = next((p for p in candidates if p.is_file()), None)
        require(manifest is not None, "No material inventory; supply --manifest")
        self.manifest = Path(manifest).resolve()
        data = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.entries = {r["path"]: r for r in data["files"]}
        require(len(self.entries) == len(data["files"]), "Duplicate inventory paths")
        self.by_digest = defaultdict(list)
        for record in self.entries.values():
            for key in ("sha256", "upstream_sha256", "source_sha256"):
                if record.get(key):
                    self.by_digest[record[key]].append(record["path"])
        self.checked = set()
        self.resolved = {}
        self.aliases = {}
        if (self.root / "PUBLIC_PATH_ALIASES.json").is_file():
            alias_file = self.path("PUBLIC_PATH_ALIASES.json")
            self.aliases = json.loads(alias_file.read_text(encoding="utf-8")).get("map", {})

    def path(self, rel, expected=None):
        # Never follow an old filesystem path, '..', or a symlink out of root.
        rel = str(rel).replace("\\", "/")
        cache_key = (rel, expected)
        if cache_key in self.resolved:
            return self.resolved[cache_key]
        candidates = [rel]
        alias = "upstream-project:/" + rel.lstrip("/")
        for prefix in ("/data/lwx/proj/TCR_unseen_epitope/",):
            if rel.startswith(prefix):
                alias = "upstream-project:/" + rel[len(prefix):]
        if alias in self.aliases:
            candidates.append(self.aliases[alias]["package_path"])
        if expected:
            candidates.extend(self.by_digest.get(expected, ()))
        selected = next((r for r in candidates if r in self.entries and (not expected or expected in (
            self.entries[r].get("sha256"), self.entries[r].get("upstream_sha256"),
            self.entries[r].get("source_sha256")))), None)
        if selected is None:
            raise UnresolvedMemberError("Unresolved frozen member: " + rel)
        record = self.entries[selected]
        if expected:
            require(expected in (record.get("sha256"), record.get("upstream_sha256"),
                                 record.get("source_sha256")), "Frozen source identity mismatch: " + selected)
        path = (self.root / selected).resolve()
        path.relative_to(self.root)
        require(path.is_file(), "Missing material: " + selected)
        if selected not in self.checked:
            require(path.stat().st_size == record["bytes"], "Byte-size drift: " + selected)
            require(sha(path) == record["sha256"], "SHA-256 drift: " + selected)
            self.checked.add(selected)
        self.resolved[cache_key] = path
        return path

    def json(self, rel, expected=None):
        return json.loads(self.path(rel, expected).read_text(encoding="utf-8"))

    def csv(self, rel, expected=None):
        return csv_rows(self.path(rel, expected))

    def module(self, rel, name):
        path = self.path(rel)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def same(actual, expected, location, atol=2e-12):
    if isinstance(actual, dict):
        for key, value in actual.items():
            require(key in expected, location + ": missing " + key)
            same(value, expected[key], location + "/" + key, atol)
    elif isinstance(actual, (float, int)) and not isinstance(actual, bool):
        require(math.isfinite(float(expected)), location + ": nonfinite reference")
        require(math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=atol),
                location + ": numeric mismatch")
    elif isinstance(actual, bool):
        require(str(actual).lower() == str(expected).lower(), location + ": Boolean mismatch")
    else:
        require(str(actual) == str(expected), location + ": value mismatch")


def keyed(rows, fields):
    result = {tuple(str(r[f]) for f in fields): r for r in rows}
    require(len(result) == len(rows), "Duplicate result selector: " + ",".join(fields))
    return result


def compare_rows(actual, expected, fields, name):
    a, b = keyed(actual, fields), keyed(expected, fields)
    require(a.keys() == b.keys(), name + ": result universe differs")
    for key in a:
        same(a[key], b[key], name + ":" + "|".join(key))


def verify_h1(m):
    api = m.module(H1 + "/metrics/build_metrics.py", "frozen_h1_metrics")
    index = m.json(H1 + "/scoring/prediction_index.json")
    require(index["status"] == "PASS_H1_EXACT125_INTERNAL_AND25_EXTERNAL_PREDICTIONS",
            "H1 closed prediction gate failed")
    require(len(index["internal"]) == 125 and len(index["external"]) == 25,
            "H1 prediction budget differs")
    internal, external = defaultdict(list), {}
    expected_slots = {(a, b, f) for a in api.RESAMPLES for b in api.RESAMPLES for f in api.FOLDS}
    seen = set()
    for entry in index["internal"]:
        slot = (entry["train_resample_id"], entry["test_resample_id"], entry["fold_id"])
        require(slot in expected_slots and slot not in seen, "H1 internal duplicate/unexpected task")
        seen.add(slot)
        rows = m.csv(entry["path"], entry["sha256"])
        require(len(rows) == entry["rows"], "H1 internal row count mismatch")
        require(all(r["fold_id"] == entry["fold_id"] for r in rows), "H1 fold mismatch")
        internal[slot[:2]].extend(rows)
    require(seen == expected_slots, "H1 missing internal task")
    for entry in index["external"]:
        require(entry["member_id"] not in external, "H1 duplicate external member")
        external[entry["member_id"]] = m.csv(entry["path"], entry["sha256"])
        require(len(external[entry["member_id"]]) == entry["rows"], "H1 external row count mismatch")
    ensemble = api.ensemble(external)
    contract = m.json(H1 + "/metrics/calculation_contract.json")
    m.path(H1 + "/metrics/calculation_contract.json", index["metric_contract_sha256"])
    masks = m.csv(contract["fixed_masks"]["path"], contract["fixed_masks"]["sha256"])
    removed = {(mask, scope): {r["query_id"] for r in masks if r["mask_id"] == mask
                and r["scope"] == scope and r["remove_complete_query"] == "True"}
               for mask in api.MASK_IDS for scope in ("internal", "external")}
    cells, queries, aggregates = [], [], []
    for set_id in ("full", *api.MASK_IDS):
        local = []
        for (train, test), full in sorted(internal.items()):
            require((len(full), sum(int(r["label"]) for r in full), len({r["query_id"] for r in full}))
                    == (20578, 10289, 168), "H1 internal coverage differs")
            qfold = defaultdict(set)
            for r in full:
                qfold[r["query_id"]].add(r["fold_id"])
            require(all(len(v) == 1 for v in qfold.values()), "H1 query occurs in multiple folds")
            rows = full if set_id == "full" else [r for r in full if r["query_id"] not in removed[set_id, "internal"]]
            summary, q = api.summarize(rows, "internal", set_id, train, test)
            local.append(summary); cells.append(summary); queries.extend(q)
        aggregates.append(api.mean_cells(local, set_id))
        rows = ensemble if set_id == "full" else [r for r in ensemble if r["query_id"] not in removed[set_id, "external"]]
        summary, q = api.summarize(rows, "external", set_id)
        aggregates.append({**summary, "cell_count": 1,
                           "aggregation": "math.fsum/25 member probabilities per record before query-equal metrics"})
        queries.extend(q)
    compare_rows(cells, m.csv(H1 + "/metrics/results/cell_metrics.csv"),
                 ("set_id", "train_resample_id", "test_resample_id"), "H1 cells")
    compare_rows(queries, m.csv(H1 + "/metrics/results/per_query_metrics.csv"),
                 ("scope", "set_id", "train_resample_id", "test_resample_id", "query_id"), "H1 query metrics")
    compare_rows([r for r in aggregates if r["set_id"] == "full"],
                 m.json(H1 + "/metrics/results/primary_summary.json"), ("scope",), "H1 primary")
    compare_rows([r for r in aggregates if r["set_id"] != "full"],
                 m.csv(H1 + "/metrics/results/six_fixed_source_sensitivity_summaries.csv"),
                 ("scope", "set_id"), "H1 fixed masks")
    compare_rows(ensemble, m.csv(H1 + "/metrics/results/external_25_member_ensemble_predictions.csv"),
                 ("interaction_key",), "H1 external ensemble")
    return {"status": "PASS", "internal_prediction_files": 125, "external_prediction_files": 25,
            "cells": len(cells), "query_results": len(queries), "source_sensitivity_summaries": 6,
            "model_scoring_performed": False, "inferential_analysis_performed": False}


def verify_g2(m):
    api = m.module(G2 + "/summarize_r04_g2.py", "frozen_g2_metrics")
    ledger = m.json(G2 + "/execution/fit_ledger.json")
    ledger = ledger["tasks"] if isinstance(ledger, dict) else ledger
    expected = {(s, f, d) for s in api.RECIPES for f in api.FOLDS for d in api.DRAWS}
    require({(r["recipe_id"], r["fold_id"], r["draw_id"]) for r in ledger} == expected
            and len(ledger) == 105, "G2 closed 105-task grid differs")
    common = m.csv("generations/G2_structure/execution/common_positives.csv")
    common_keys = Counter(api.key(r) for r in common)
    groups = defaultdict(list)
    for entry in ledger:
        require(entry["status"] == "PASS" and entry["fit_attempts"] == 1, "G2 ledger status differs")
        rows = m.csv(G2 + "/" + entry["prediction_file"], entry["prediction_sha256"])
        require(len(rows) == entry["test_rows"], "G2 prediction count differs")
        offset = 0
        for kind in ("positive", "negative"):
            source = entry["source_inputs"]["test"]
            source_rel, source_sha = source[kind + "_file"], source[kind + "_sha256"]
            source_rows = m.csv("generations/G2_structure/" + source_rel, source_sha)
            require(len(source_rows) == source[kind + "_rows"], "G2 frozen input count differs")
            for number, original in enumerate(source_rows):
                require(offset < len(rows), "G2 missing prediction source row")
                row = rows[offset]; offset += 1
                require(api.key(row) == api.key(original) and row["pmhc_id"] == original["pmhc_id"],
                        "G2 prediction/source key differs")
                require(row["input_source_file"] == source_rel and row["input_source_sha256"] == source_sha
                        and row["input_source_row_index"] == str(number), "G2 source selector differs")
        require(offset == len(rows), "G2 extra prediction source rows")
        for row in rows:
            require(all(row[k] == entry[k] for k in ("recipe_id", "fold_id", "draw_id", "model_id", "model_sha256")),
                    "G2 prediction model/task identity differs")
            prob, logit = float(row["probability"]), float(row["logit"])
            require(math.isfinite(prob) and math.isfinite(logit) and 0 <= prob <= 1, "G2 invalid score")
            logistic = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
            require(abs(prob - logistic) <= 1e-12, "G2 probability/logit mismatch")
        groups[entry["recipe_id"], entry["draw_id"]].extend(rows)
    queries, draws = [], []
    for (recipe, draw), rows in sorted(groups.items()):
        require(Counter(api.key(r) for r in rows if int(r["label"]) == 1) == common_keys,
                "G2 common-positive universe differs")
        qgroups = defaultdict(list)
        for row in rows:
            qgroups[api.query(row)].append(row)
        local = []
        for (peptide, hla), rr in sorted(qgroups.items()):
            require(len({r["fold_id"] for r in rr}) == 1, "G2 query occurs in multiple test folds")
            labels = [int(r["label"]) for r in rr]
            scores = [float(r["probability"]) for r in rr]
            p, n = sum(labels), len(labels)
            require(p > 0 and n == 2 * p, "G2 query no longer balanced")
            counts = Counter(scores); tied = [v for v in counts.values() if v >= 2]
            ap = float(api.average_precision_score(labels, scores))
            auc = float(api.roc_auc_score(labels, scores)); add = p/n; ref = api.finite_ap(p, n)
            local.append({"recipe_id": recipe, "draw_id": draw, "fold_id": rr[0]["fold_id"],
                "epitope": peptide, "hla": hla, "pmhc_id": peptide + "|" + hla,
                "npositive": p, "nnegative": n-p, "sample_count": n,
                "unique_score_count": len(counts), "tied_group_count": len(tied),
                "records_in_ties": sum(tied), "max_tie_size": max(counts.values()),
                "interaction_AP": ap, "interaction_AUROC": auc, "additive_AP": add,
                "additive_AUROC": 0.5, "finite_no_ties_expected_AP": ref,
                "finite_no_ties_expected_AUROC": 0.5, "delta_Additive_AP": ap-add,
                "delta_Additive_AUROC": auc-0.5, "delta_finite_no_ties_AP": ap-ref,
                "delta_finite_no_ties_AUROC": auc-0.5, "status": "COMPLETE_DESCRIPTIVE"})
        queries.extend(local)
        draws.append({"recipe_id": recipe, "draw_id": draw, "query_count": len(local),
                      "row_count": len(rows), "npositive": sum(int(r["label"]) for r in rows),
                      "nnegative": sum(1-int(r["label"]) for r in rows),
                      **api.means(local, api.BASE_METRICS), "status": "COMPLETE_DESCRIPTIVE"})
    recipes = [{"recipe_id": s, "fixed_draw_count": 3, "complete_draw_count": 3, "sd_ddof": 0,
                **api.describe([r for r in draws if r["recipe_id"] == s], api.BASE_METRICS),
                "status": "COMPLETE_DESCRIPTIVE"} for s in api.RECIPES]
    compare_rows(queries, m.csv(G2 + "/metrics/per_query_oof.csv"),
                 ("recipe_id", "draw_id", "epitope", "hla"), "G2 query metrics")
    compare_rows(draws, m.csv(G2 + "/metrics/draw_oof_summary.csv"),
                 ("recipe_id", "draw_id"), "G2 draws")
    compare_rows(recipes, m.csv(G2 + "/metrics/recipe_summary.csv"), ("recipe_id",), "G2 recipes")
    baseline = {(r["draw_id"], r["epitope"], r["hla"]): r for r in queries if r["recipe_id"] == "S0"}
    vsq = []
    for r in queries:
        b = baseline[r["draw_id"], r["epitope"], r["hla"]]
        vsq.append({"recipe_id": r["recipe_id"], "draw_id": r["draw_id"], "epitope": r["epitope"],
                    "hla": r["hla"], "pmhc_id": r["pmhc_id"], "recipe_fold_id": r["fold_id"],
                    "S0_fold_id": b["fold_id"], "npositive": r["npositive"], "nnegative": r["nnegative"],
                    **{"delta_G2_S0_" + k: r[k]-b[k] for k in api.BASE_METRICS}, "status": "COMPLETE_DESCRIPTIVE"})
    vsd = [{"recipe_id": s, "draw_id": d, "paired_query_count": len(
            [r for r in vsq if r["recipe_id"] == s and r["draw_id"] == d]),
            **api.means([r for r in vsq if r["recipe_id"] == s and r["draw_id"] == d], api.DELTA_COLS),
            "status": "COMPLETE_DESCRIPTIVE"} for s in api.RECIPES for d in api.DRAWS]
    vsr = [{"recipe_id": s, "fixed_draw_count": 3, "complete_draw_count": 3, "sd_ddof": 0,
            **api.describe([r for r in vsd if r["recipe_id"] == s], api.DELTA_COLS),
            "status": "COMPLETE_DESCRIPTIVE"} for s in api.RECIPES]
    for actual, filename, keys in ((vsq, "vs_s0_per_query.csv", ("recipe_id", "draw_id", "epitope", "hla")),
                                    (vsd, "vs_s0_draw_summary.csv", ("recipe_id", "draw_id")),
                                    (vsr, "vs_s0_recipe_summary.csv", ("recipe_id",))):
        compare_rows(actual, m.csv(G2 + "/metrics/" + filename), keys, "G2 " + filename)
    return {"status": "PASS", "prediction_files": 105, "query_results": len(queries),
            "draws": len(draws), "recipes": len(recipes), "sd_ddof": 0,
            "model_scoring_performed": False, "inferential_analysis_performed": False}


def model_specs(m, group):
    if group in ("H1", "all"):
        for i in range(1, 6):
            for f in range(1, 6):
                task = f"resample_{i}__fold_{f}"
                for kind in ("interaction", "analytic_additive"):
                    yield "H1", task, kind, f"active_models/H1/{kind}/{task}.joblib", None
    if group in ("G2", "all"):
        ledger = m.json(G2 + "/execution/fit_ledger.json")
        ledger = ledger["tasks"] if isinstance(ledger, dict) else ledger
        for r in ledger:
            yield "G2", r["task_id"], "interaction", G2 + "/" + r["model_file"], r["model_sha256"]
        for r in m.json(G2 + "/additive/artifact_index.json")["artifacts"]:
            yield "G2", r["task_id"], "analytic_additive", G2 + "/" + r["artifact_file"], r["artifact_sha256"]


def load_artifact(m, rel, digest=None):
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    artifact = joblib.load(m.path(rel, digest))
    require(isinstance(artifact, dict) and isinstance(artifact["estimator"], LogisticRegression),
            "Unexpected artifact type: " + rel)
    estimator = artifact["estimator"]
    require(estimator.classes_.tolist() == [0, 1], "Unexpected classifier labels")
    require(np.isfinite(estimator.coef_).all() and np.isfinite(estimator.intercept_).all(),
            "Nonfinite artifact parameters")
    require(estimator.coef_.shape[1] == artifact["feature_dimensions"]["total"], "Feature dimension mismatch")
    if artifact.get("analytic_solution") is True:
        require(artifact.get("optimizer_fit_performed") is False, "Analytic artifact fit flag differs")
        require(np.all(estimator.coef_ == 0) and np.all(estimator.intercept_ == 0), "Analytic artifact not exact zero")
        require(np.all(estimator.n_iter_ == 0), "Analytic artifact has optimizer iterations")
    return artifact


def verify_models(m, group):
    counts = Counter()
    for branch, task, kind, rel, digest in model_specs(m, group):
        artifact = load_artifact(m, rel, digest)
        if kind == "analytic_additive":
            require(artifact.get("analytic_solution") is True, "Missing analytic solution flag: " + task)
        counts[branch + "/" + kind] += 1
    expected = {"H1/interaction": 25, "H1/analytic_additive": 25,
                "G2/interaction": 105, "G2/analytic_additive": 105}
    require(dict(counts) == {k:v for k,v in expected.items() if group == "all" or k.startswith(group + "/")},
            "Model artifact budget differs")
    return {"status": "PASS", "loaded": dict(counts), "training_performed": False, "model_scoring_performed": False}


def feature_api(m):
    """Compile a closed whitelist of *unchanged* frozen numerical definitions.

    Importing the whole legacy module pulls in Torch/CNN training. This adapter
    executes only feature constants/functions, with no stubbed legacy modules.
    """
    import numpy as np
    from scipy.sparse import csr_matrix, hstack
    namespace = {"np": np, "csr_matrix": csr_matrix, "hstack": hstack, "Counter": Counter}
    selections = {
        "vocab.py": {"STANDARD_AMINO_ACIDS"},
        "kmer_logistic.py": {"UNK_HLA", "normalized_kmer_frequencies"},
        "degree_matched_orthogonalized_logistic.py": {
            "ALPHABET_SIZE", "CENTERING_MATRIX", "INTERACTION_FEATURE_COUNT", "ORTHOGONALITY_TOLERANCE",
            "SEQUENCE_MAIN_EFFECT_COUNT", "BASE_FEATURE_COUNT", "double_centered_outer",
            "orthogonalized_feature_dimensions", "_build_feature_matrix_and_audit", "build_orthogonalized_feature_matrix"},
        "degree_matched_additive_logistic.py": {
            "ADDITIVE_SEQUENCE_FEATURE_COUNT", "additive_feature_dimensions", "build_additive_feature_matrix"},
    }
    for filename, names in selections.items():
        path = m.path("active_source/src/" + filename)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        selected, seen = [], set()
        for node in tree.body:
            node_names = {node.name} if isinstance(node, ast.FunctionDef) else {
                target.id for target in getattr(node, "targets", ()) if isinstance(target, ast.Name)}
            if node_names & names:
                require(node_names <= names, "Unexpected shared assignment in frozen builder")
                selected.append(node); seen.update(node_names)
        require(seen == names, "Incomplete frozen feature API: " + filename)
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def score_smoke(m, group, records):
    import numpy as np
    api = feature_api(m)
    checked = []
    for branch in ("H1", "G2"):
        if group not in ("all", branch):
            continue
        if branch == "H1":
            # One reused and one newly fitted frozen member cover both paths.
            index = m.json(H1 + "/scoring/prediction_index.json")
            selection = [next(r for r in index["external"] if r["action"] == action)
                         for action in ("VERIFIED_SCORE_REUSE", "NEW_SCORE")]
            tasks = [(r["member_id"], m.csv(r["path"], r["sha256"]),
                      "active_models/H1/interaction/" + r["member_id"] + ".joblib",
                      "active_models/H1/analytic_additive/" + r["member_id"] + ".joblib",
                      "interaction_probability") for r in selection]
        else:
            ledger = m.json(G2 + "/execution/fit_ledger.json")
            ledger = ledger["tasks"] if isinstance(ledger, dict) else ledger
            r = ledger[0]
            tasks = [(r["task_id"], m.csv(G2 + "/" + r["prediction_file"], r["prediction_sha256"]),
                      G2 + "/" + r["model_file"], G2 + "/additive/models/" + r["task_id"] + ".joblib", "probability")]
        for task, full, model_rel, additive_rel, probability_field in tasks:
            # Include both labels with deterministic source-order selection.
            by_label = [[r for r in full if int(r["label"]) == label] for label in (1, 0)]
            rows = [r for pair in zip(*by_label) for r in pair][:records]
            require(len(rows) == records, "Insufficient frozen records for smoke check")
            model, additive = load_artifact(m, model_rel), load_artifact(m, additive_rel)
            matrix = api["build_orthogonalized_feature_matrix"](rows, model["hla_mapping"])
            probability = model["estimator"].predict_proba(matrix)[:, 1]
            reference = np.asarray([float(r[probability_field]) for r in rows])
            maximum = float(np.max(np.abs(probability-reference)))
            require(maximum <= 1e-12, "Checkpoint roundtrip differs: " + task)
            add_matrix = api["build_additive_feature_matrix"](rows, additive["hla_mapping"])
            require(np.all(additive["estimator"].predict_proba(add_matrix)[:, 1] == 0.5),
                    "Analytic checkpoint roundtrip differs: " + task)
            checked.append({"group": branch, "task_id": task, "records": len(rows),
                            "maximum_probability_error": maximum, "analytic_probability": 0.5})
    return {"status": "PASS", "checks": checked, "training_performed": False,
            "scope": "Small fixed-record checkpoint roundtrip; not full rescoring or a new analysis"}


def pointer(value, selector):
    for part in selector.strip("/").split("/") if selector else ():
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def verify_provenance(m):
    cache, counts, unavailable = {}, Counter(), []
    for table in ("1", "2", "3", "S1", "S2", "S3", "S4"):
        rel = COMM + "/Table_" + table + "_provenance.json"
        provenance = m.json(rel)
        displayed = m.csv(COMM + "/Table_" + table + ".csv")
        for cell in provenance["cells"]:
            selected_rows = [r for r in displayed if r.get("section", cell["panel_or_section"]) == cell["panel_or_section"]]
            printed = selected_rows[cell["table_row_index_zero_based"]][cell["metric"]]
            same(printed, cell["plotted_or_printed_value"], "Table_" + table + " displayed cell")
            counts["verified_display_cell"] += 1
            fmt = cell["field_selector"]["format"]
            require(all(k in cell for k in ("source_file", "source_sha256", "source_commit", "field_selector", "confirmatory_status")),
                    "Incomplete provenance record")
            selector = cell["field_selector"]
            if fmt == "TEXT":
                same(selector["literal"], cell["value_full_precision"], "Editorial literal")
                counts["editorial_literal"] += 1
                continue
            try:
                path = m.path(cell["source_file"], cell["source_sha256"])
            except UnresolvedMemberError:
                unavailable.append({"table": table, "source_file": cell["source_file"],
                                    "source_sha256": cell["source_sha256"], "format": fmt})
                continue
            key = str(path)
            if key not in cache:
                cache[key] = m.json(path.relative_to(m.root).as_posix()) if fmt == "JSON" else (
                    csv_rows(path) if fmt == "CSV" else path.read_text(encoding="utf-8"))
            if fmt == "JSON":
                value = pointer(cache[key], selector["pointer"])
            elif fmt == "CSV":
                value = cache[key][selector["data_row_index_zero_based"]][selector["column"]]
            elif fmt == "MARKDOWN_TABLE":
                require(selector["source_literal"] in cache[key], "Missing source literal in Markdown")
                counts["editorial_translation_source_literal"] += 1
                continue
            else:
                raise ValueError("Unsupported provenance selector: " + fmt)
            # Full precision means semantic equality; CSV represents scalars as text.
            target = cell["value_full_precision"]
            if isinstance(value, (dict, list)):
                require(value == target, "Structured provenance value mismatch")
            else:
                try:
                    same(float(value), float(target), "Table_" + table + " source selector", atol=1e-12)
                except (TypeError, ValueError):
                    same(str(value), str(target), "Table_" + table + " source selector")
            counts["verified_" + fmt.lower() + "_selector"] += 1
    unique = {json.dumps(r, sort_keys=True): r for r in unavailable}
    return {"status": "PASS" if not unavailable else "INCOMPLETE_MISSING_FROZEN_PROVENANCE_SOURCES",
            "counts": dict(counts), "unavailable_cell_count": len(unavailable),
            "unavailable_sources": list(unique.values()),
            "editorial_translation_equivalence": "Requires human judgment; only source literal checked"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path, default=Path(__file__).resolve().parent.parent / "materials")
    parser.add_argument("--manifest", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-results", "load-models", "score-smoke"):
        p = sub.add_parser(name)
        p.add_argument("--group", choices=("H1", "G2", "all"), default="all")
        if name == "score-smoke":
            p.add_argument("--records", type=int, default=4, choices=range(2, 21))
    sub.add_parser("verify-provenance")
    sub.add_parser("verify-integrity")
    args = parser.parse_args()
    m = Materials(args.materials, args.manifest)
    if args.command == "verify-results":
        result = {branch: fn(m) for branch, fn in (("H1", verify_h1), ("G2", verify_g2)) if args.group in ("all", branch)}
    elif args.command == "load-models":
        result = verify_models(m, args.group)
    elif args.command == "score-smoke":
        result = score_smoke(m, args.group, args.records)
    elif args.command == "verify-provenance":
        result = verify_provenance(m)
    else:
        for rel in m.entries:
            m.path(rel)
        result = {"status": "PASS", "inventory_files": len(m.entries)}
    print(json.dumps({"command": args.command, "verified_material_files": len(m.checked), "result": result},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 2 if isinstance(result, dict) and str(result.get("status", "")).startswith("INCOMPLETE") else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, FileNotFoundError, ImportError) as exc:
        print(json.dumps({"status": "FAIL", "error": type(exc).__name__ + ": " + str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
