from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .vdjdb import file_sha256


REGISTRY_COLUMNS = [
    "method",
    "role",
    "artifact_type",
    "resample_id",
    "fold_id",
    "path",
    "expected_sha256",
    "observed_sha256",
    "size_bytes",
    "hash_match",
    "aggregation",
]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _validate_source(project_dir: Path, source: Mapping[str, Any]) -> Path:
    path = project_dir / str(source["path"])
    if not path.is_file():
        raise FileNotFoundError(f"required protocol source is missing: {path}")
    observed = file_sha256(path)
    if observed != source["sha256"]:
        raise ValueError(
            f"protocol source SHA-256 mismatch for {source['path']}: "
            f"expected {source['sha256']}, observed {observed}"
        )
    return path


def _find_artifacts(
    value: Any, suffix: str
) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            path = item.get("path")
            sha256 = item.get("sha256")
            if (
                isinstance(path, str)
                and path.endswith(suffix)
                and isinstance(sha256, str)
            ):
                found.append((path, sha256))
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return found


def _registry_row(
    project_dir: Path,
    *,
    method: str,
    role: str,
    artifact_type: str,
    resample_id: str,
    fold_id: str,
    path: str,
    expected_sha256: str,
    aggregation: str,
) -> dict[str, str | int]:
    artifact = project_dir / path
    if not artifact.is_file():
        raise FileNotFoundError(f"required checkpoint/reference is missing: {artifact}")
    observed_sha256 = file_sha256(artifact)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"checkpoint/reference SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    return {
        "method": method,
        "role": role,
        "artifact_type": artifact_type,
        "resample_id": resample_id,
        "fold_id": fold_id,
        "path": path,
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha256,
        "size_bytes": artifact.stat().st_size,
        "hash_match": "true",
        "aggregation": aggregation,
    }


def _build_checkpoint_registry(
    project_dir: Path, config: Mapping[str, Any]
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    fold_ids = list(config["fold_ids"])
    resample_ids = list(config["resample_ids"])

    primary_source = config["primary_models"]["checkpoint_source"]
    primary_summary = _read_json(_validate_source(project_dir, primary_source))
    primary_artifacts = _find_artifacts(primary_summary, "model.joblib")
    pattern = re.compile(
        r"/(resample_[1-9][0-9]*)/"
        r"(additive_only|orthogonalized_interaction)/"
        r"(fold_[1-9][0-9]*)_model\.joblib$"
    )
    seen_primary: set[tuple[str, str, str]] = set()
    for path, sha256 in primary_artifacts:
        match = pattern.search(path)
        if not match:
            continue
        resample_id, method, fold_id = match.groups()
        if resample_id not in resample_ids or fold_id not in fold_ids:
            continue
        key = (method, resample_id, fold_id)
        if key in seen_primary:
            raise ValueError(f"duplicate primary checkpoint registration: {key}")
        seen_primary.add(key)
        method_config = config["primary_models"][method]
        rows.append(
            _registry_row(
                project_dir,
                method=method,
                role=str(method_config["role"]),
                artifact_type="model_checkpoint",
                resample_id=resample_id,
                fold_id=fold_id,
                path=path,
                expected_sha256=sha256,
                aggregation=str(method_config["probability_aggregation"]),
            )
        )
    expected_primary = {
        (method, resample_id, fold_id)
        for method in ("orthogonalized_interaction", "additive_only")
        for resample_id in resample_ids
        for fold_id in fold_ids
    }
    if seen_primary != expected_primary:
        missing = sorted(expected_primary - seen_primary)
        extra = sorted(seen_primary - expected_primary)
        raise ValueError(f"primary checkpoint grid mismatch; missing={missing}, extra={extra}")
    for method in ("orthogonalized_interaction", "additive_only"):
        configured_count = int(config["primary_models"][method]["checkpoint_count"])
        observed_count = sum(row["method"] == method for row in rows)
        if observed_count != configured_count:
            raise ValueError(
                f"{method} checkpoint count mismatch: "
                f"expected {configured_count}, observed {observed_count}"
            )

    for method, section_name, suffix in (
        ("degree_matched_cnn", "degree_matched_cnn", ".pt"),
        (
            "degree_matched_tcr_only_logistic",
            "degree_matched_tcr_only_logistic",
            "model.joblib",
        ),
    ):
        method_config = config["secondary_controls"][section_name]
        source = method_config["checkpoint_source"]
        summary = _read_json(_validate_source(project_dir, source))
        artifacts = _find_artifacts(summary, suffix)
        by_fold: dict[str, tuple[str, str]] = {}
        for path, sha256 in artifacts:
            fold_match = re.search(r"/(fold_[1-9][0-9]*)/", path)
            if not fold_match:
                continue
            fold_id = fold_match.group(1)
            if fold_id in fold_ids:
                if fold_id in by_fold:
                    raise ValueError(f"duplicate {method} checkpoint for {fold_id}")
                by_fold[fold_id] = (path, sha256)
        if set(by_fold) != set(fold_ids):
            raise ValueError(f"{method} checkpoint folds are incomplete")
        if len(by_fold) != int(method_config["checkpoint_count"]):
            raise ValueError(f"{method} checkpoint count mismatch")
        for fold_id in fold_ids:
            path, sha256 = by_fold[fold_id]
            rows.append(
                _registry_row(
                    project_dir,
                    method=method,
                    role="secondary_control",
                    artifact_type="model_checkpoint",
                    resample_id="",
                    fold_id=fold_id,
                    path=path,
                    expected_sha256=sha256,
                    aggregation=str(method_config["probability_aggregation"]),
                )
            )

    similarity = config["secondary_controls"]["sequence_similarity"]
    similarity_summary = _read_json(
        _validate_source(project_dir, similarity["reference_source"])
    )
    references = _find_artifacts(
        similarity_summary, "train_query_balanced_cap100.csv"
    )
    reference_by_fold: dict[str, tuple[str, str]] = {}
    for path, sha256 in references:
        fold_match = re.search(r"_(fold_[1-9][0-9]*)_train_", path)
        if fold_match and fold_match.group(1) in fold_ids:
            reference_by_fold[fold_match.group(1)] = (path, sha256)
    if set(reference_by_fold) != set(fold_ids):
        raise ValueError("sequence-similarity train references are incomplete")
    if len(reference_by_fold) != int(similarity["fold_reference_count"]):
        raise ValueError("sequence-similarity reference count mismatch")
    for fold_id in fold_ids:
        path, sha256 = reference_by_fold[fold_id]
        rows.append(
            _registry_row(
                project_dir,
                method="sequence_similarity",
                role="secondary_control",
                artifact_type="train_positive_reference",
                resample_id="",
                fold_id=fold_id,
                path=path,
                expected_sha256=sha256,
                aggregation=str(similarity["score_aggregation"]),
            )
        )

    rows.sort(
        key=lambda row: (
            str(row["method"]),
            str(row["resample_id"]),
            str(row["fold_id"]),
        )
    )
    return rows


def _cohort_summary(manifest: Mapping[str, Any]) -> dict[str, int | float | bool]:
    summary = manifest["summary"]
    degree = manifest["degree_matching_audit"]
    cohort = {
        "query_count": int(summary["query_count"]),
        "positive_count": int(summary["positive_count"]),
        "negative_count": int(summary["negative_count"]),
        "degree_matching_rate": float(degree["matching_rate"]),
        "tcr_degree_exact": bool(degree["tcr_degree_exact"]),
        "query_degree_exact": bool(degree["query_degree_exact"]),
    }
    if cohort["query_count"] <= 0 or cohort["positive_count"] <= 0:
        raise ValueError("formal external cohort is empty")
    if cohort["positive_count"] != cohort["negative_count"]:
        raise ValueError("formal external cohort is not class-balanced")
    if (
        cohort["degree_matching_rate"] != 1.0
        or cohort["tcr_degree_exact"] is not True
        or cohort["query_degree_exact"] is not True
    ):
        raise ValueError("formal external cohort is not exactly degree matched")
    return cohort


def _write_registry(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _render_protocol(protocol: Mapping[str, Any]) -> str:
    cohort = protocol["cohort"]
    counts = protocol["artifact_counts"]
    return "\n".join(
        [
            "# Frozen IEDB external evaluation protocol",
            "",
            "## Pre-scoring declaration",
            "",
            "This protocol was frozen before any external model scoring. No external model performance has been viewed, and no external prediction was generated during protocol freezing.",
            "",
            "Figure 1 remains a draft and must not be formally updated until external validation is complete.",
            "",
            "## Formal cohort",
            "",
            f"- Queries: {cohort['query_count']}",
            f"- Positives: {cohort['positive_count']}",
            f"- Negatives: {cohort['negative_count']}",
            f"- Exact degree matching: {cohort['degree_matching_rate']:.6f}",
            "",
            "All cohort quantities above are read from the frozen formal-cohort manifest.",
            "",
            "## Frozen methods",
            "",
            f"- Orthogonalized-interaction Logistic: {counts['orthogonalized_interaction_checkpoints']} checkpoints; arithmetic mean of all checkpoint probabilities.",
            f"- Additive-only Logistic: {counts['additive_only_checkpoints']} checkpoints; arithmetic mean of all checkpoint probabilities.",
            f"- Degree-matched CNN: {counts['degree_matched_cnn_checkpoints']} checkpoints; arithmetic mean across folds.",
            f"- Degree-matched TCR-only Logistic: {counts['degree_matched_tcr_only_logistic_checkpoints']} checkpoints; arithmetic mean across folds.",
            f"- Sequence Similarity: {counts['sequence_similarity_fold_references']} fold-specific positive training references; same-HLA only and no cross-HLA fallback. Available-fold scores are averaged.",
            "",
            "Any missing artifact or SHA-256 mismatch is a hard stop.",
            "",
            "## Endpoints and comparison",
            "",
            "The sole primary endpoint is query-level Macro AUPRC. The primary comparison is Orthogonalized-interaction Logistic minus Additive-only Logistic over all formal-cohort queries.",
            "",
            "Secondary endpoints are Macro AUROC, pooled AUROC, and pooled AUPRC. Similarity-related comparisons are restricted to common evaluable queries and must report coverage separately.",
            "",
            "## Statistical analysis",
            "",
            "The query is the independent sampling unit. The paired analysis uses 10,000 query-cluster bootstrap replicates for 95% confidence intervals and 10,000 query-level paired permutations for two-sided p values. The three secondary endpoints use Holm correction; the pre-specified Macro AUPRC primary endpoint is reported as the sole primary test.",
            "",
            "No external data may be used for model selection, calibration, threshold selection, or hyperparameter tuning.",
            "",
            "## Exploratory stratification",
            "",
            "HLA seen/unseen, TCR seen/unseen, and antigen source analyses are exploratory. They cannot replace the primary analysis over the complete formal cohort.",
            "",
            "## Operations performed",
            "",
            "No training, scoring, prediction, resampling, statistical testing, or plotting was performed while freezing this protocol.",
            "",
        ]
    )


def freeze_iedb_evaluation_protocol(
    project_dir: Path,
    config_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["protocol_status"] != "frozen" or config["freeze_timing"] != "pre-scoring":
        raise ValueError("IEDB evaluation protocol is not marked frozen pre-scoring")
    declaration = config["pre_scoring_declaration"]
    if declaration["frozen_before_any_external_scoring"] is not True:
        raise ValueError("protocol was not frozen before external scoring")
    if declaration["external_model_performance_viewed"] is not False:
        raise ValueError("external model performance was viewed before protocol freeze")
    if any(bool(value) for value in config["restrictions"].values()):
        raise ValueError("a prohibited operation is enabled in the evaluation protocol")

    cohort_source = config["cohort"]["manifest"]
    cohort_manifest = _read_json(_validate_source(project_dir, cohort_source))
    cohort = _cohort_summary(cohort_manifest)
    registry = _build_checkpoint_registry(project_dir, config)

    checkpoint_rows = [row for row in registry if row["artifact_type"] == "model_checkpoint"]
    artifact_counts = {
        "evaluation_methods": 5,
        "model_checkpoints": len(checkpoint_rows),
        "orthogonalized_interaction_checkpoints": sum(
            row["method"] == "orthogonalized_interaction" for row in checkpoint_rows
        ),
        "additive_only_checkpoints": sum(
            row["method"] == "additive_only" for row in checkpoint_rows
        ),
        "degree_matched_cnn_checkpoints": sum(
            row["method"] == "degree_matched_cnn" for row in checkpoint_rows
        ),
        "degree_matched_tcr_only_logistic_checkpoints": sum(
            row["method"] == "degree_matched_tcr_only_logistic"
            for row in checkpoint_rows
        ),
        "sequence_similarity_fold_references": sum(
            row["method"] == "sequence_similarity" for row in registry
        ),
        "registered_artifacts": len(registry),
    }

    protocol = {
        "schema_version": int(config["schema_version"]),
        "source_commit": config["source_commit"],
        "protocol_status": config["protocol_status"],
        "freeze_date": str(config["freeze_date"]),
        "freeze_timing": config["freeze_timing"],
        "pre_scoring_declaration": declaration,
        "cohort": cohort,
        "artifact_counts": artifact_counts,
        "primary_models": config["primary_models"],
        "secondary_controls": config["secondary_controls"],
        "endpoints": config["endpoints"],
        "statistics": config["statistics"],
        "exploratory_stratification": config["exploratory_stratification"],
        "stratification_rule": config["stratification_rule"],
        "similarity_comparison": config["similarity_comparison"],
        "restrictions": config["restrictions"],
    }

    configured_output = project_dir / config["outputs"]["directory"]
    destination = output_dir or configured_output
    destination.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "protocol_json": destination / config["outputs"]["protocol_json"],
        "protocol_md": destination / config["outputs"]["protocol_md"],
        "checkpoint_registry_csv": destination
        / config["outputs"]["checkpoint_registry_csv"],
        "manifest_json": destination / config["outputs"]["manifest_json"],
    }
    output_paths["protocol_json"].write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_paths["protocol_md"].write_text(
        _render_protocol(protocol), encoding="utf-8"
    )
    _write_registry(output_paths["checkpoint_registry_csv"], registry)

    source_entries = {
        "configuration": {
            "path": str(config_path.relative_to(project_dir)),
            "sha256": file_sha256(config_path),
        },
        "cohort_manifest": dict(cohort_source),
        "primary_checkpoint_summary": dict(
            config["primary_models"]["checkpoint_source"]
        ),
        "cnn_checkpoint_summary": dict(
            config["secondary_controls"]["degree_matched_cnn"]["checkpoint_source"]
        ),
        "tcr_only_checkpoint_summary": dict(
            config["secondary_controls"]["degree_matched_tcr_only_logistic"][
                "checkpoint_source"
            ]
        ),
        "similarity_reference_summary": dict(
            config["secondary_controls"]["sequence_similarity"]["reference_source"]
        ),
        "similarity_algorithm": dict(
            config["secondary_controls"]["sequence_similarity"]["algorithm"]
        ),
        "similarity_config": dict(
            config["secondary_controls"]["sequence_similarity"]["config"]
        ),
    }
    for source in source_entries.values():
        _validate_source(project_dir, source)

    manifest = {
        "schema_version": 1,
        "source_commit": config["source_commit"],
        "protocol_status": "frozen",
        "freeze_timing": "pre-scoring",
        "external_model_performance_viewed": False,
        "cohort": cohort,
        "artifact_counts": artifact_counts,
        "checkpoint_and_reference_hashes_all_match": all(
            row["hash_match"] == "true" for row in registry
        ),
        "sources": source_entries,
        "outputs": {
            key: {
                "path": str(
                    (configured_output / path.name).relative_to(project_dir)
                ),
                "sha256": file_sha256(path),
            }
            for key, path in output_paths.items()
            if key != "manifest_json"
        },
        "operations_performed": {
            "training": False,
            "scoring": False,
            "prediction_generation": False,
            "resampling": False,
            "statistical_testing": False,
            "plotting": False,
        },
        "figure_1_status": "draft_until_external_validation_complete",
    }
    output_paths["manifest_json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
