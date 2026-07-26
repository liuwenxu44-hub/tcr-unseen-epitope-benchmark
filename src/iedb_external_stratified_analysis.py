from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .five_fold_summary import file_sha256
from .iedb_external_evaluation import (
    ALL_METHODS,
    COMPARATOR_METHOD,
    ENDPOINTS,
    FULL_METHODS,
    PRIMARY_METHOD,
    SIMILARITY_METHOD,
    _absolute_metrics,
    _csv_bytes,
    _format_float,
    _json_bytes,
    _load_json,
    _load_yaml,
    _query_metric_rows,
    _quantile_interval,
    _read_csv,
    _relative,
    _resolve,
    _weighted_pooled_replicates,
    stratified_query_bootstrap_counts,
)
from .metrics import binary_classification_metrics
from .tcrue.iedb_external_audit import _source_category


STRATIFICATIONS = (
    "hla_seen_vs_unseen",
    "tcr_seen_vs_unseen",
    "antigen_source",
)
SOURCE_CATEGORY_ORDER = (
    "tumor_associated",
    "self_antigen",
    "pathogen_or_nonself",
    "other_nonself",
    "unknown_source",
)
PRIOR_OUTPUT_FILES = {
    "evaluation_manifest.json",
    "external_evaluation_report.md",
    "external_metrics.csv",
    "external_per_query_metrics.csv",
    "primary_comparison.csv",
    "primary_comparison.json",
}
NEW_OUTPUT_FILES = {
    "stratified_metrics.csv",
    "stratified_comparisons.csv",
    "stratified_analysis.json",
    "stratified_analysis_report.md",
    "stratified_analysis_manifest.json",
}

METRIC_COLUMNS = (
    "stratification",
    "stratum",
    "population",
    "method",
    "query_count",
    "positive_count",
    "negative_count",
    "publication_count",
    "macro_evaluable_query_count",
    "macro_excluded_query_count",
    "similarity_available_query_count",
    "similarity_coverage_fraction",
    "small_query_stratum",
    "query_level_macro_auprc",
    "query_level_macro_auroc",
    "pooled_auprc",
    "pooled_auroc",
)
COMPARISON_COLUMNS = (
    "stratification",
    "stratum",
    "endpoint",
    "orthogonalized",
    "additive_only",
    "difference",
    "ci_lower",
    "ci_upper",
    "query_count",
    "positive_count",
    "negative_count",
    "publication_count",
    "macro_evaluable_query_count",
    "macro_excluded_query_count",
    "bootstrap_query_count",
    "bootstrap_iterations",
    "bootstrap_seed",
    "confidence_level",
    "exploratory_only",
    "small_query_stratum",
)


def _verified_path(
    project_dir: Path, source: Mapping[str, Any], description: str
) -> Path:
    path = _resolve(project_dir, str(source["path"]))
    observed = file_sha256(path)
    if observed != source["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {description}: {path}")
    return path


def _parse_bool(value: str, description: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{description} must be an explicit True/False annotation")


def _publication_tokens(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        token
        for row in rows
        for token in str(row["reference_id"]).split("|")
        if token
    }


def _eligible_queries(
    labels: np.ndarray, query_ids: Sequence[str]
) -> tuple[set[str], set[str]]:
    classes: dict[str, set[int]] = defaultdict(set)
    for query_id, label in zip(query_ids, labels, strict=True):
        classes[str(query_id)].add(int(label))
    eligible = {query_id for query_id, values in classes.items() if values == {0, 1}}
    return eligible, set(classes) - eligible


def paired_stratum_bootstrap(
    labels: np.ndarray,
    query_ids: Sequence[str],
    orthogonalized_scores: np.ndarray,
    additive_scores: np.ndarray,
    *,
    seed: int,
    iterations: int,
    confidence_level: float,
    batch_size: int = 128,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    orthogonalized_scores = np.asarray(orthogonalized_scores, dtype=np.float64)
    additive_scores = np.asarray(additive_scores, dtype=np.float64)
    query_array = np.asarray(query_ids, dtype=object)
    if not (
        labels.shape
        == orthogonalized_scores.shape
        == additive_scores.shape
        == query_array.shape
    ):
        raise ValueError("stratum bootstrap arrays differ in shape")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("stratum bootstrap requires both classes")

    all_queries = sorted(set(query_ids))
    all_query_index = {
        query_id: index for index, query_id in enumerate(all_queries)
    }
    row_query_indices = np.asarray(
        [all_query_index[str(query_id)] for query_id in query_ids], dtype=np.int64
    )
    all_counts, _ = stratified_query_bootstrap_counts(
        ["stratum"] * len(all_queries), seed=seed, iterations=iterations
    )
    orthogonalized_pooled = _weighted_pooled_replicates(
        all_counts,
        row_query_indices,
        labels.astype(np.float64),
        orthogonalized_scores,
        batch_size=batch_size,
    )
    additive_pooled = _weighted_pooled_replicates(
        all_counts,
        row_query_indices,
        labels.astype(np.float64),
        additive_scores,
        batch_size=batch_size,
    )
    pooled_replicates = {
        "pooled_auprc": orthogonalized_pooled["auprc"]
        - additive_pooled["auprc"],
        "pooled_auroc": orthogonalized_pooled["auroc"]
        - additive_pooled["auroc"],
    }
    if not all(np.isfinite(values).all() for values in pooled_replicates.values()):
        raise ValueError("a pooled query-cluster bootstrap replicate was undefined")

    eligible, excluded = _eligible_queries(labels, query_ids)
    if not eligible:
        raise ValueError("stratum has no query containing both classes")
    eligible_queries = sorted(eligible)
    macro_counts, _ = stratified_query_bootstrap_counts(
        ["eligible"] * len(eligible_queries), seed=seed, iterations=iterations
    )
    per_query_difference = {"auprc": [], "auroc": []}
    for query_id in eligible_queries:
        selected = query_array == query_id
        orthogonalized = binary_classification_metrics(
            labels[selected], orthogonalized_scores[selected]
        )
        additive = binary_classification_metrics(
            labels[selected], additive_scores[selected]
        )
        for metric in ("auprc", "auroc"):
            per_query_difference[metric].append(
                float(orthogonalized[metric] - additive[metric])
            )
    macro_replicates = {
        "query_level_macro_auprc": macro_counts
        @ np.asarray(per_query_difference["auprc"])
        / len(eligible_queries),
        "query_level_macro_auroc": macro_counts
        @ np.asarray(per_query_difference["auroc"])
        / len(eligible_queries),
    }
    replicates = {**macro_replicates, **pooled_replicates}
    return {
        "confidence_intervals": {
            endpoint: _quantile_interval(values, confidence_level)
            for endpoint, values in replicates.items()
        },
        "replicates": replicates,
        "all_query_count": len(all_queries),
        "macro_evaluable_query_count": len(eligible_queries),
        "macro_excluded_query_count": len(excluded),
        "iterations": iterations,
        "seed": seed,
        "confidence_level": confidence_level,
        "unit": "query_cluster",
    }


def _audit_and_load(project_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "evaluation_config": project_dir / "configs/external_iedb_evaluation.yaml",
        "evaluation_protocol": project_dir
        / "results/external/iedb/evaluation_protocol.json",
        "evaluation_protocol_manifest": project_dir
        / "results/external/iedb/evaluation_protocol_manifest.json",
        "cohort_manifest": project_dir
        / "data/external/iedb_strict_cap100/manifest.json",
        "cohort": project_dir / "data/external/iedb_strict_cap100/test.csv",
        "scoring_manifest": project_dir
        / "results/external/iedb/scoring/scoring_manifest.json",
        "ensemble_predictions": project_dir
        / "results/external/iedb/scoring/ensemble_predictions.csv",
        "evaluation_manifest": project_dir
        / "results/external/iedb/evaluation/evaluation_manifest.json",
        "external_per_query_metrics": project_dir
        / "results/external/iedb/evaluation/external_per_query_metrics.csv",
    }
    config = _load_yaml(paths["evaluation_config"])
    protocol = _load_json(paths["evaluation_protocol"])
    protocol_manifest = _load_json(paths["evaluation_protocol_manifest"])
    cohort_manifest = _load_json(paths["cohort_manifest"])
    scoring_manifest = _load_json(paths["scoring_manifest"])
    evaluation_manifest = _load_json(paths["evaluation_manifest"])

    if config.get("protocol_status") != "frozen" or config.get(
        "freeze_timing"
    ) != "pre-scoring":
        raise ValueError("evaluation protocol is not frozen pre-scoring")
    if tuple(config.get("exploratory_stratification", ())) != STRATIFICATIONS:
        raise ValueError("frozen exploratory stratifications differ from the analysis")
    if tuple(protocol.get("exploratory_stratification", ())) != STRATIFICATIONS:
        raise ValueError("protocol JSON exploratory stratifications differ")
    if config.get("stratification_rule") != (
        "exploratory_only_and_never_replaces_full_cohort_primary_analysis"
    ):
        raise ValueError("frozen exploratory-only rule is missing")
    if file_sha256(paths["evaluation_config"]) != protocol_manifest["sources"][
        "configuration"
    ]["sha256"]:
        raise ValueError("evaluation configuration differs from protocol manifest")
    if file_sha256(paths["evaluation_protocol"]) != protocol_manifest["outputs"][
        "protocol_json"
    ]["sha256"]:
        raise ValueError("evaluation protocol JSON differs from protocol manifest")
    if file_sha256(paths["cohort_manifest"]) != config["cohort"]["manifest"][
        "sha256"
    ]:
        raise ValueError("cohort manifest differs from frozen evaluation protocol")
    if file_sha256(paths["cohort"]) != cohort_manifest["outputs"]["test"][
        "sha256"
    ]:
        raise ValueError("cohort differs from cohort manifest")
    if file_sha256(paths["cohort"]) != scoring_manifest["inputs"]["cohort_test"][
        "source_file_sha256"
    ]:
        raise ValueError("cohort differs from scoring manifest")
    if file_sha256(paths["ensemble_predictions"]) != scoring_manifest["outputs"][
        "ensemble_predictions"
    ]["sha256"]:
        raise ValueError("ensemble predictions differ from scoring manifest")
    if file_sha256(paths["external_per_query_metrics"]) != evaluation_manifest[
        "outputs"
    ]["external_per_query_metrics.csv"]["sha256"]:
        raise ValueError("per-query metrics differ from evaluation manifest")
    for name in (
        "cohort_test",
        "ensemble_predictions",
        "scoring_manifest",
        "evaluation_protocol_manifest",
        "cohort_manifest",
    ):
        source = evaluation_manifest["inputs"][name]
        if file_sha256(_resolve(project_dir, source["path"])) != source["sha256"]:
            raise ValueError(f"prior evaluation input hash changed: {name}")

    audit_config_source = cohort_manifest["source_provenance"]["audit_config"]
    audit_config_path = _verified_path(
        project_dir, audit_config_source, "frozen antigen-source mapping"
    )
    audit_config = _load_yaml(audit_config_path)
    categories = audit_config.get("antigen_source_categories")
    if not isinstance(categories, dict) or tuple(categories) != SOURCE_CATEGORY_ORDER[:-1]:
        raise ValueError("antigen-source category mapping was not explicitly frozen")

    cohort_columns, cohort_rows = _read_csv(paths["cohort"])
    required_cohort_columns = {
        "interaction_key",
        "query_id",
        "reference_id",
        "label",
        "tcr_seen_in_vdjdb",
        "hla_seen_in_vdjdb",
        "antigen_source",
    }
    if not required_cohort_columns.issubset(cohort_columns):
        raise ValueError("cohort is missing a frozen stratification annotation")
    if len(cohort_rows) != 1612:
        raise ValueError("external cohort row count changed")

    cohort_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    positive_categories = Counter()
    hla_seen_by_query: dict[str, set[bool]] = defaultdict(set)
    category_by_query: dict[str, set[str]] = defaultdict(set)
    for source in cohort_rows:
        key = (source["interaction_key"], source["query_id"])
        if key in cohort_by_key:
            raise ValueError(f"duplicate cohort key: {key}")
        label = int(source["label"])
        if label not in (0, 1):
            raise ValueError("cohort label is not binary")
        tcr_seen = _parse_bool(
            source["tcr_seen_in_vdjdb"], "TCR seen-in-VDJdb"
        )
        hla_seen = _parse_bool(
            source["hla_seen_in_vdjdb"], "HLA seen-in-VDJdb"
        )
        antigen_sources = tuple(
            value for value in source["antigen_source"].split("|") if value
        )
        category = _source_category(
            {"antigen_sources": antigen_sources}, categories
        )
        if category not in SOURCE_CATEGORY_ORDER:
            raise ValueError(f"unfrozen antigen-source category: {category}")
        row = {
            **source,
            "label": label,
            "tcr_seen": tcr_seen,
            "hla_seen": hla_seen,
            "antigen_source_category": category,
        }
        cohort_by_key[key] = row
        hla_seen_by_query[source["query_id"]].add(hla_seen)
        category_by_query[source["query_id"]].add(category)
        if label == 1:
            positive_categories[category] += 1
    if any(len(values) != 1 for values in hla_seen_by_query.values()):
        raise ValueError("HLA seen annotation varies within a query")
    if any(len(values) != 1 for values in category_by_query.values()):
        raise ValueError("antigen-source category varies within a query")
    expected_category_counts = Counter(
        cohort_manifest["summary"]["positive_antigen_source_distribution"]
    )
    if positive_categories != expected_category_counts:
        raise ValueError("frozen antigen-source mapping no longer reproduces manifest")

    ensemble_columns, ensemble_rows = _read_csv(paths["ensemble_predictions"])
    if "label" in ensemble_columns:
        raise ValueError("ensemble predictions unexpectedly contain labels")
    predictions: dict[str, dict[tuple[str, str], float | None]] = {
        method: {} for method in ALL_METHODS
    }
    for row in ensemble_rows:
        method = row["method"]
        if method not in predictions:
            raise ValueError(f"unexpected method: {method}")
        key = (row["interaction_key"], row["query_id"])
        if key not in cohort_by_key or key in predictions[method]:
            raise ValueError(f"invalid or duplicate ensemble key for {method}: {key}")
        raw = row["ensemble_score"].strip()
        score = None if raw == "" else float(raw)
        if score is not None and (not math.isfinite(score) or not 0.0 <= score <= 1.0):
            raise ValueError(f"invalid score for {method}: {score}")
        predictions[method][key] = score
    cohort_keys = set(cohort_by_key)
    for method, values in predictions.items():
        if set(values) != cohort_keys:
            raise ValueError(f"ensemble keys differ from cohort for {method}")
        if method in FULL_METHODS and any(value is None for value in values.values()):
            raise ValueError(f"full-coverage method is incomplete: {method}")

    per_query_columns, per_query_rows = _read_csv(paths["external_per_query_metrics"])
    required_per_query = {"method", "query_id", "status", "auprc", "auroc"}
    if not required_per_query.issubset(per_query_columns):
        raise ValueError("prior per-query metrics are missing required columns")
    per_query_index = {
        (row["method"], row["query_id"]): row for row in per_query_rows
    }
    if len(per_query_index) != len(per_query_rows) or len(per_query_rows) != 705:
        raise ValueError("prior per-query metric keys are duplicated or incomplete")
    expected_queries = {row["query_id"] for row in cohort_rows}
    if set(per_query_index) != {
        (method, query_id) for method in ALL_METHODS for query_id in expected_queries
    }:
        raise ValueError("prior per-query metric method/query keys differ")

    ordered_keys = sorted(cohort_keys, key=lambda key: (key[1], key[0]))
    joined = []
    for key in ordered_keys:
        row = cohort_by_key[key]
        joined.append(
            {
                **row,
                "scores": {method: predictions[method][key] for method in ALL_METHODS},
            }
        )

    labels = np.asarray([row["label"] for row in joined], dtype=np.int8)
    query_ids = [str(row["query_id"]) for row in joined]
    for method in FULL_METHODS:
        scores = np.asarray([row["scores"][method] for row in joined], dtype=np.float64)
        _, recomputed = _query_metric_rows(
            labels, scores, query_ids, method=method
        )
        for query_id, metrics in recomputed.items():
            prior = per_query_index[(method, query_id)]
            if prior["status"] != "evaluable":
                raise ValueError("prior full-method query was not evaluable")
            for metric in ("auprc", "auroc"):
                if not math.isclose(
                    metrics[metric],
                    float(prior[metric]),
                    rel_tol=0.0,
                    abs_tol=5e-12,
                ):
                    raise ValueError(
                        f"prior per-query {metric} differs for {method}/{query_id}"
                    )
    for query_id in expected_queries:
        available = all(
            row["scores"][SIMILARITY_METHOD] is not None
            for row in joined
            if row["query_id"] == query_id
        )
        expected_status = "evaluable" if available else "unavailable"
        if per_query_index[(SIMILARITY_METHOD, query_id)]["status"] != expected_status:
            raise ValueError("Similarity availability differs from prior per-query metrics")

    sources = {
        name: {"path": _relative(project_dir, path), "sha256": file_sha256(path)}
        for name, path in paths.items()
    }
    sources["antigen_source_mapping"] = {
        "path": _relative(project_dir, audit_config_path),
        "sha256": file_sha256(audit_config_path),
    }
    audit = {
        "all_input_sha256_match": True,
        "cohort_prediction_keys_paired": True,
        "prior_per_query_metrics_reproduced": True,
        "stratifications_frozen_pre_scoring": list(STRATIFICATIONS),
        "definitions": {
            "hla_seen_vs_unseen": (
                "record hla_seen_in_vdjdb annotation; HLA membership in the frozen "
                "complete VDJdb positive registry"
            ),
            "tcr_seen_vs_unseen": (
                "record tcr_seen_in_vdjdb annotation; TCR membership in the frozen "
                "complete VDJdb positive registry"
            ),
            "antigen_source": (
                "pre-scoring external_iedb_audit.yaml keyword mapping applied by the "
                "existing _source_category implementation"
            ),
        },
        "antigen_source_categories": list(SOURCE_CATEGORY_ORDER),
        "antigen_source_mapping_reproduces_cohort_manifest": True,
        "cohort_query_count": len(expected_queries),
        "cohort_record_count": len(joined),
        "sources": sources,
    }
    return {
        "config": config,
        "joined": joined,
        "source_categories": categories,
    }, audit


def _strata(joined: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    strata = []
    for stratification, field in (
        ("hla_seen_vs_unseen", "hla_seen"),
        ("tcr_seen_vs_unseen", "tcr_seen"),
    ):
        for value, name in ((True, "seen"), (False, "unseen")):
            indices = [
                index for index, row in enumerate(joined) if row[field] is value
            ]
            if not indices:
                raise ValueError(f"frozen stratum is empty: {stratification}/{name}")
            strata.append(
                {
                    "stratification": stratification,
                    "stratum": name,
                    "indices": indices,
                }
            )
    observed_categories = {
        str(row["antigen_source_category"]) for row in joined
    }
    for category in SOURCE_CATEGORY_ORDER:
        indices = [
            index
            for index, row in enumerate(joined)
            if row["antigen_source_category"] == category
        ]
        if indices:
            strata.append(
                {
                    "stratification": "antigen_source",
                    "stratum": category,
                    "indices": indices,
                }
            )
    if observed_categories != {
        row["stratum"]
        for row in strata
        if row["stratification"] == "antigen_source"
    }:
        raise ValueError("observed antigen-source categories were altered")
    return strata


def _evaluate_once(loaded: Mapping[str, Any]) -> dict[str, Any]:
    joined = loaded["joined"]
    config = loaded["config"]
    all_labels = np.asarray([row["label"] for row in joined], dtype=np.int8)
    all_queries = np.asarray([row["query_id"] for row in joined], dtype=object)
    all_scores = {
        method: np.asarray(
            [
                np.nan if row["scores"][method] is None else row["scores"][method]
                for row in joined
            ],
            dtype=np.float64,
        )
        for method in ALL_METHODS
    }
    similarity_available_queries = {
        str(query_id)
        for query_id in sorted(set(all_queries))
        if np.isfinite(
            all_scores[SIMILARITY_METHOD][all_queries == query_id]
        ).all()
    }
    bootstrap_config = config["statistics"]["bootstrap"]
    metric_rows = []
    comparison_rows = []
    strata_json = []
    for stratum_spec in _strata(joined):
        indices = np.asarray(stratum_spec["indices"], dtype=np.int64)
        labels = all_labels[indices]
        query_ids = all_queries[indices].tolist()
        stratum_rows = [joined[index] for index in indices]
        query_count = len(set(query_ids))
        positive_count = int(labels.sum())
        negative_count = int(labels.size - positive_count)
        publication_count = len(_publication_tokens(stratum_rows))
        eligible, excluded = _eligible_queries(labels, query_ids)
        small = query_count < 5
        full_metrics = {}
        for method in FULL_METHODS:
            metrics, _, method_eligible = _absolute_metrics(
                labels,
                all_scores[method][indices],
                query_ids,
                method=method,
            )
            if set(method_eligible) != eligible:
                raise ValueError("macro query eligibility differs by full method")
            full_metrics[method] = metrics
            metric_rows.append(
                {
                    "stratification": stratum_spec["stratification"],
                    "stratum": stratum_spec["stratum"],
                    "population": "stratum_all_records",
                    "method": method,
                    "query_count": query_count,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "publication_count": publication_count,
                    "macro_evaluable_query_count": len(eligible),
                    "macro_excluded_query_count": len(excluded),
                    "similarity_available_query_count": "",
                    "similarity_coverage_fraction": "",
                    "small_query_stratum": str(small).lower(),
                    **{
                        endpoint: _format_float(metrics[endpoint])
                        for endpoint in ENDPOINTS
                    },
                }
            )

        available_in_stratum = set(query_ids) & similarity_available_queries
        common_queries = eligible & similarity_available_queries
        common_local_mask = np.asarray(
            [query_id in common_queries for query_id in query_ids], dtype=np.bool_
        )
        common_metrics = {}
        if common_queries:
            common_indices = indices[common_local_mask]
            common_labels = all_labels[common_indices]
            common_query_ids = all_queries[common_indices].tolist()
            common_rows = [joined[index] for index in common_indices]
            for method in ALL_METHODS:
                metrics, _, method_eligible = _absolute_metrics(
                    common_labels,
                    all_scores[method][common_indices],
                    common_query_ids,
                    method=method,
                )
                if set(method_eligible) != common_queries:
                    raise ValueError("Similarity common-query population is inconsistent")
                common_metrics[method] = metrics
                metric_rows.append(
                    {
                        "stratification": stratum_spec["stratification"],
                        "stratum": stratum_spec["stratum"],
                        "population": "similarity_common_evaluable_queries",
                        "method": method,
                        "query_count": len(common_queries),
                        "positive_count": int(common_labels.sum()),
                        "negative_count": int(
                            common_labels.size - common_labels.sum()
                        ),
                        "publication_count": len(_publication_tokens(common_rows)),
                        "macro_evaluable_query_count": len(common_queries),
                        "macro_excluded_query_count": 0,
                        "similarity_available_query_count": len(
                            available_in_stratum
                        ),
                        "similarity_coverage_fraction": _format_float(
                            len(available_in_stratum) / query_count
                        ),
                        "small_query_stratum": str(small).lower(),
                        **{
                            endpoint: _format_float(metrics[endpoint])
                            for endpoint in ENDPOINTS
                        },
                    }
                )
        else:
            for method in ALL_METHODS:
                metric_rows.append(
                    {
                        "stratification": stratum_spec["stratification"],
                        "stratum": stratum_spec["stratum"],
                        "population": "similarity_common_evaluable_queries",
                        "method": method,
                        "query_count": 0,
                        "positive_count": 0,
                        "negative_count": 0,
                        "publication_count": 0,
                        "macro_evaluable_query_count": 0,
                        "macro_excluded_query_count": 0,
                        "similarity_available_query_count": len(
                            available_in_stratum
                        ),
                        "similarity_coverage_fraction": _format_float(
                            len(available_in_stratum) / query_count
                        ),
                        "small_query_stratum": str(small).lower(),
                        **{endpoint: "" for endpoint in ENDPOINTS},
                    }
                )

        differences = {
            endpoint: full_metrics[PRIMARY_METHOD][endpoint]
            - full_metrics[COMPARATOR_METHOD][endpoint]
            for endpoint in ENDPOINTS
        }
        bootstrap = paired_stratum_bootstrap(
            labels,
            query_ids,
            all_scores[PRIMARY_METHOD][indices],
            all_scores[COMPARATOR_METHOD][indices],
            seed=int(bootstrap_config["seed"]),
            iterations=int(bootstrap_config["iterations"]),
            confidence_level=float(bootstrap_config["confidence_level"]),
        )
        comparison_json = {}
        for endpoint in ENDPOINTS:
            lower, upper = bootstrap["confidence_intervals"][endpoint]
            bootstrap_query_count = (
                bootstrap["macro_evaluable_query_count"]
                if endpoint.startswith("query_level_macro")
                else bootstrap["all_query_count"]
            )
            comparison_rows.append(
                {
                    "stratification": stratum_spec["stratification"],
                    "stratum": stratum_spec["stratum"],
                    "endpoint": endpoint,
                    "orthogonalized": _format_float(
                        full_metrics[PRIMARY_METHOD][endpoint]
                    ),
                    "additive_only": _format_float(
                        full_metrics[COMPARATOR_METHOD][endpoint]
                    ),
                    "difference": _format_float(differences[endpoint]),
                    "ci_lower": _format_float(lower),
                    "ci_upper": _format_float(upper),
                    "query_count": query_count,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "publication_count": publication_count,
                    "macro_evaluable_query_count": len(eligible),
                    "macro_excluded_query_count": len(excluded),
                    "bootstrap_query_count": bootstrap_query_count,
                    "bootstrap_iterations": bootstrap["iterations"],
                    "bootstrap_seed": bootstrap["seed"],
                    "confidence_level": _format_float(
                        bootstrap["confidence_level"]
                    ),
                    "exploratory_only": "true",
                    "small_query_stratum": str(small).lower(),
                }
            )
            comparison_json[endpoint] = {
                "orthogonalized": full_metrics[PRIMARY_METHOD][endpoint],
                "additive_only": full_metrics[COMPARATOR_METHOD][endpoint],
                "difference": differences[endpoint],
                "confidence_interval_95": [lower, upper],
                "bootstrap_query_count": bootstrap_query_count,
            }
        strata_json.append(
            {
                "stratification": stratum_spec["stratification"],
                "stratum": stratum_spec["stratum"],
                "sample_size": {
                    "query_count": query_count,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "publication_count": publication_count,
                    "macro_evaluable_query_count": len(eligible),
                    "macro_excluded_query_count": len(excluded),
                },
                "small_query_stratum": small,
                "full_method_metrics": full_metrics,
                "orthogonalized_minus_additive_only": comparison_json,
                "similarity": {
                    "available_query_count": len(available_in_stratum),
                    "common_evaluable_query_count": len(common_queries),
                    "total_stratum_query_count": query_count,
                    "coverage_fraction": len(available_in_stratum) / query_count,
                    "common_population_metrics": common_metrics,
                },
            }
        )
    return {
        "metric_rows": metric_rows,
        "comparison_rows": comparison_rows,
        "analysis_json": {
            "schema_version": 1,
            "status": "exploratory_pre_specified_stratified_analysis",
            "never_replaces_full_141_query_primary_analysis": True,
            "stratifications": strata_json,
            "statistics": {
                "bootstrap_type": "paired_query_cluster",
                "iterations": int(bootstrap_config["iterations"]),
                "seed": int(bootstrap_config["seed"]),
                "confidence_level": float(bootstrap_config["confidence_level"]),
                "p_values_computed": False,
            },
            "restrictions": {
                "small_antigen_source_query_threshold": 5,
                "small_strata_descriptive_only": True,
                "category_merging": False,
                "seen_unseen_definition_changes": False,
                "antigen_source_mapping_changes": False,
            },
        },
    }


def _report(result: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    lines = [
        "# IEDB external exploratory stratified analysis",
        "",
        "This analysis is restricted to the three exploratory stratifications frozen before external scoring. It does not replace the primary analysis over all 141 queries.",
        "",
        "No model training, rescoring, tuning, calibration, threshold selection, plotting, or significance testing was performed.",
        "",
        "## Frozen definitions",
        "",
        f"- HLA seen/unseen: {audit['definitions']['hla_seen_vs_unseen']}.",
        f"- TCR seen/unseen: {audit['definitions']['tcr_seen_vs_unseen']}.",
        f"- Antigen source: {audit['definitions']['antigen_source']}.",
        "",
        "## Sample sizes and Macro AUPRC comparison",
        "",
        "| Stratification | Stratum | Queries | Positive | Negative | Publications | Macro eligible | Excluded | Orthogonalized - Additive-only Macro AUPRC (95% CI) | Similarity common eligible / coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stratum in result["analysis_json"]["stratifications"]:
        size = stratum["sample_size"]
        comparison = stratum["orthogonalized_minus_additive_only"][
            "query_level_macro_auprc"
        ]
        similarity = stratum["similarity"]
        lower, upper = comparison["confidence_interval_95"]
        lines.append(
            "| {dimension} | {stratum} | {query_count} | {positive_count} | "
            "{negative_count} | {publication_count} | {eligible} | {excluded} | "
            "{difference:.6f} [{lower:.6f}, {upper:.6f}] | {common} / {coverage:.6f} |".format(
                dimension=stratum["stratification"],
                stratum=stratum["stratum"],
                query_count=size["query_count"],
                positive_count=size["positive_count"],
                negative_count=size["negative_count"],
                publication_count=size["publication_count"],
                eligible=size["macro_evaluable_query_count"],
                excluded=size["macro_excluded_query_count"],
                difference=comparison["difference"],
                lower=lower,
                upper=upper,
                common=similarity["common_evaluable_query_count"],
                coverage=similarity["coverage_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "All four endpoint differences and confidence intervals, all absolute method metrics, and Similarity common-query descriptive metrics are stored in the CSV and JSON outputs.",
            "",
            "Antigen-source strata with fewer than five queries are designated descriptive-only and are not interpreted. Categories were not merged after scoring.",
            "",
            "No p values were calculated or reported.",
            "",
        ]
    )
    return "\n".join(lines)


def _current_commit(project_dir: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def analyze_iedb_external_strata(
    project_dir: Path,
    *,
    output_dir: Path | None = None,
    analysis_source_commit: str | None = None,
) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    target = (
        output_dir.resolve()
        if output_dir is not None
        else project_dir / "results/external/iedb/evaluation"
    )
    source_commit = analysis_source_commit or _current_commit(project_dir)
    if target.exists():
        unexpected = {
            path.name for path in target.iterdir() if path.is_file()
        } - PRIOR_OUTPUT_FILES - NEW_OUTPUT_FILES
        if unexpected:
            raise ValueError(
                "Evaluation directory contains unexpected user-owned files: "
                + ", ".join(sorted(unexpected))
            )

    loaded, audit = _audit_and_load(project_dir)
    first = _evaluate_once(loaded)
    second = _evaluate_once(loaded)
    payloads = {
        "stratified_metrics.csv": _csv_bytes(
            METRIC_COLUMNS, first["metric_rows"]
        ),
        "stratified_comparisons.csv": _csv_bytes(
            COMPARISON_COLUMNS, first["comparison_rows"]
        ),
        "stratified_analysis.json": _json_bytes(first["analysis_json"]),
        "stratified_analysis_report.md": _report(first, audit).encode("utf-8"),
    }
    repeated = {
        "stratified_metrics.csv": _csv_bytes(
            METRIC_COLUMNS, second["metric_rows"]
        ),
        "stratified_comparisons.csv": _csv_bytes(
            COMPARISON_COLUMNS, second["comparison_rows"]
        ),
        "stratified_analysis.json": _json_bytes(second["analysis_json"]),
        "stratified_analysis_report.md": _report(second, audit).encode("utf-8"),
    }
    if payloads != repeated:
        raise ValueError("repeated stratified analysis outputs differ")

    target.mkdir(parents=True, exist_ok=True)
    for filename, payload in payloads.items():
        (target / filename).write_bytes(payload)
    output_records = {
        filename: {
            "path": _relative(project_dir, target / filename),
            "sha256": file_sha256(target / filename),
            "rows": (
                len(first["metric_rows"])
                if filename == "stratified_metrics.csv"
                else len(first["comparison_rows"])
                if filename == "stratified_comparisons.csv"
                else None
            ),
        }
        for filename in payloads
    }
    manifest = {
        "schema_version": 1,
        "analysis_source_commit": source_commit,
        "mode": "pre_specified_exploratory_external_stratified_analysis",
        "inputs": audit.pop("sources"),
        "audit": audit,
        "stratifications": list(STRATIFICATIONS),
        "statistics": first["analysis_json"]["statistics"],
        "reproducibility": {
            "complete_analysis_runs_compared": 2,
            "all_deterministic_output_bytes_identical": True,
        },
        "operations": {
            "training": False,
            "rescoring": False,
            "hyperparameter_tuning": False,
            "calibration": False,
            "threshold_selection": False,
            "significance_testing": False,
            "plotting": False,
            "figure_1_updated": False,
        },
        "outputs": output_records,
    }
    manifest_path = target / "stratified_analysis_manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    return {
        "output_dir": _relative(project_dir, target),
        "strata": first["analysis_json"]["stratifications"],
        "outputs": {
            **output_records,
            "stratified_analysis_manifest.json": {
                "path": _relative(project_dir, manifest_path),
                "sha256": file_sha256(manifest_path),
            },
        },
    }
