from __future__ import annotations

import csv
import io
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .five_fold_summary import file_sha256
from .iedb_external_scoring import (
    _load_feature_rows,
    _load_registry,
    _logistic_scores,
    _read_json,
    _read_yaml,
    _verify_sha256,
)
from .tcrue.iedb_external_audit import (
    _canonical_tcr,
    _standardized_hla,
    _valid_epitope,
)


BASE_COMMIT = "932a6353324701319599b188f8f4114477c3d2fd"
METHOD = "orthogonalized_interaction"
EXPECTED_RESAMPLE_IDS = tuple(f"resample_{index}" for index in range(1, 6))
EXPECTED_FOLD_IDS = tuple(f"fold_{index}" for index in range(1, 6))
EXPECTED_MODEL_COUNT = len(EXPECTED_RESAMPLE_IDS) * len(EXPECTED_FOLD_IDS)
OUTPUT_COLUMNS = (
    "tcr_cdr3_beta",
    "epitope",
    "hla",
    "score",
    "rank",
    "contributing_model_count",
)
EVALUATION_CONFIG_PATH = Path("configs/external_iedb_evaluation.yaml")
PROTOCOL_MANIFEST_PATH = Path(
    "results/external/iedb/evaluation_protocol_manifest.json"
)
AUDIT_CONFIG_PATH = Path("configs/external_iedb_audit.yaml")
FEASIBILITY_MANIFEST_PATH = Path(
    "results/external/iedb/feasibility_manifest.json"
)
COHORT_TEST_PATH = Path("data/external/iedb_strict_cap100/test.csv")
FROZEN_ENSEMBLE_PATH = Path(
    "results/external/iedb/scoring/ensemble_predictions.csv"
)


def _git_commit(project_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        git_path = project_dir / ".git"
        if not git_path.exists():
            return BASE_COMMIT
        if git_path.is_file():
            marker = git_path.read_text(encoding="utf-8").strip()
            if not marker.startswith("gitdir: "):
                raise RuntimeError("could not locate the Git metadata directory")
            git_path = (project_dir / marker.removeprefix("gitdir: ")).resolve()
        head = (git_path / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            reference = head.removeprefix("ref: ")
            loose_reference = git_path / reference
            if loose_reference.is_file():
                commit = loose_reference.read_text(encoding="utf-8").strip()
            else:
                commit = ""
                packed_refs = git_path / "packed-refs"
                if packed_refs.is_file():
                    for line in packed_refs.read_text(encoding="utf-8").splitlines():
                        if line.endswith(f" {reference}"):
                            commit = line.split(" ", maxsplit=1)[0]
                            break
        else:
            commit = head
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("could not determine the code commit")
    return commit


def _load_validation_scope(project_dir: Path) -> dict[str, Any]:
    manifest = _read_json(project_dir / FEASIBILITY_MANIFEST_PATH)
    source = manifest["audit_pipeline"]["config"]
    if Path(source["path"]) != AUDIT_CONFIG_PATH:
        raise ValueError("frozen amino-acid/HLA validation config path changed")
    config_path = project_dir / AUDIT_CONFIG_PATH
    _verify_sha256(config_path, source["sha256"], "IEDB validation scope config")
    config = _read_yaml(config_path)
    scope = config.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("IEDB validation scope is missing")
    if scope.get("classical_full_resolution_hla_only") is not True:
        raise ValueError("frozen HLA validation no longer requires full resolution")
    return scope


def normalize_query(
    project_dir: Path, epitope: str, hla: str
) -> tuple[str, str, dict[str, Any]]:
    scope = _load_validation_scope(project_dir)
    normalized_epitope = _valid_epitope(epitope, scope)
    if not normalized_epitope:
        raise ValueError(
            "invalid epitope: expected 8-15 standard amino-acid characters"
        )
    normalized_hla = _standardized_hla(hla)
    if not normalized_hla:
        raise ValueError(
            "invalid HLA: expected one classical full-resolution value such as "
            "HLA-A*02:01"
        )
    return normalized_epitope, normalized_hla, scope


def load_candidate_tcrs(
    tcr_file: Path, scope: Mapping[str, Any]
) -> tuple[list[str], int, int]:
    if not tcr_file.is_file():
        raise FileNotFoundError(f"candidate TCR file is missing: {tcr_file}")
    tcrs: list[str] = []
    input_rows = 0
    with tcr_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"candidate TCR CSV has no header: {tcr_file}")
        if "tcr_cdr3_beta" not in reader.fieldnames:
            raise ValueError(
                "candidate TCR CSV must contain column: tcr_cdr3_beta"
            )
        for line_number, row in enumerate(reader, start=2):
            input_rows += 1
            normalized = _canonical_tcr(row.get("tcr_cdr3_beta", ""), scope)
            if not normalized:
                raise ValueError(
                    f"invalid TCR at CSV line {line_number}: expected a canonical "
                    "5-30 aa sequence using standard amino acids, starting with C "
                    "and ending with F or W"
                )
            tcrs.append(normalized)
    if not tcrs:
        raise ValueError("candidate TCR CSV contains no data rows")
    unique_tcrs = sorted(set(tcrs))
    return unique_tcrs, input_rows, input_rows - len(unique_tcrs)


def load_orthogonalized_registry(
    project_dir: Path,
    *,
    registry_path: Path | None = None,
    registry_sha256: str | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    config_path = project_dir / EVALUATION_CONFIG_PATH
    protocol_manifest = _read_json(project_dir / PROTOCOL_MANIFEST_PATH)
    config_source = protocol_manifest["sources"]["configuration"]
    _verify_sha256(config_path, config_source["sha256"], "frozen evaluation config")
    config = _read_yaml(config_path)
    if config.get("protocol_status") != "frozen":
        raise ValueError("external evaluation protocol is not frozen")
    expected_count = int(
        config["primary_models"][METHOD]["checkpoint_count"]
    )
    if expected_count != EXPECTED_MODEL_COUNT:
        raise ValueError("frozen Orthogonalized checkpoint count is not 25")

    registry_source = protocol_manifest["outputs"]["checkpoint_registry_csv"]
    selected_path = registry_path or project_dir / registry_source["path"]
    selected_sha = registry_sha256 or registry_source["sha256"]
    registry = _load_registry(project_dir, selected_path, selected_sha)
    rows = sorted(
        (
            dict(row)
            for row in registry
            if row["method"] == METHOD
            and row["artifact_type"] == "model_checkpoint"
        ),
        key=lambda row: (row["resample_id"], row["fold_id"]),
    )
    observed_pairs = {(row["resample_id"], row["fold_id"]) for row in rows}
    expected_pairs = {
        (resample_id, fold_id)
        for resample_id in EXPECTED_RESAMPLE_IDS
        for fold_id in EXPECTED_FOLD_IDS
    }
    if len(rows) != expected_count or observed_pairs != expected_pairs:
        raise ValueError(
            "checkpoint registry does not contain the complete resample_1-5 x "
            "fold_1-5 Orthogonalized ensemble"
        )
    if any(
        row["aggregation"] != "arithmetic_mean_across_all_25_checkpoints"
        for row in rows
    ):
        raise ValueError("Orthogonalized checkpoint aggregation rule changed")
    return rows, config


def score_feature_rows(
    project_dir: Path,
    feature_rows: Sequence[Mapping[str, str]],
    registry_rows: Sequence[Mapping[str, str]],
) -> tuple[list[float], list[list[float]]]:
    if len(registry_rows) != EXPECTED_MODEL_COUNT:
        raise ValueError("exactly 25 Orthogonalized checkpoints are required")
    per_record: list[list[float]] = [[] for _ in feature_rows]
    for registry_row in registry_rows:
        _, probabilities = _logistic_scores(
            project_dir, registry_row, feature_rows
        )
        for index, probability in enumerate(probabilities):
            value = float(probability)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"invalid score from checkpoint: {registry_row['path']}"
                )
            per_record[index].append(value)
    if any(len(values) != EXPECTED_MODEL_COUNT for values in per_record):
        raise ValueError("one or more records were not scored by all 25 models")
    scores = [math.fsum(values) / EXPECTED_MODEL_COUNT for values in per_record]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores):
        raise ValueError("ensemble score is outside [0,1] or non-finite")
    return scores, per_record


def build_ranked_rows(
    tcrs: Sequence[str], epitope: str, hla: str, scores: Sequence[float]
) -> list[dict[str, Any]]:
    if len(tcrs) != len(scores):
        raise ValueError("TCR and score counts differ")
    ordered = sorted(zip(tcrs, scores, strict=True), key=lambda item: (-item[1], item[0]))
    return [
        {
            "tcr_cdr3_beta": tcr,
            "epitope": epitope,
            "hla": hla,
            "score": format(float(score), ".17g"),
            "rank": rank,
            "contributing_model_count": EXPECTED_MODEL_COUNT,
        }
        for rank, (tcr, score) in enumerate(ordered, start=1)
    ]


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(OUTPUT_COLUMNS), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _metadata_path(out_path: Path) -> Path:
    return Path(f"{out_path}.metadata.json")


def rank_candidate_tcrs(
    project_dir: Path,
    *,
    epitope: str,
    hla: str,
    tcr_file: Path,
    out_path: Path,
) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    if tcr_file.resolve() == out_path.resolve():
        raise ValueError("--out must differ from --tcr-file")
    metadata_path = _metadata_path(out_path)
    if metadata_path.resolve() == tcr_file.resolve():
        raise ValueError("metadata path would overwrite --tcr-file")

    normalized_epitope, normalized_hla, scope = normalize_query(
        project_dir, epitope, hla
    )
    tcrs, input_rows, duplicate_count = load_candidate_tcrs(tcr_file, scope)
    registry_rows, _ = load_orthogonalized_registry(project_dir)
    query_id = f"{normalized_epitope}|{normalized_hla}"
    feature_rows = [
        {
            "record_id": f"cli:{index}",
            "interaction_key": f"{tcr}|{normalized_epitope}|{normalized_hla}",
            "query_id": query_id,
            "tcr_cdr3_beta": tcr,
            "epitope": normalized_epitope,
            "hla": normalized_hla,
        }
        for index, tcr in enumerate(tcrs, start=1)
    ]
    scores, per_record = score_feature_rows(project_dir, feature_rows, registry_rows)
    ranked_rows = build_ranked_rows(
        tcrs, normalized_epitope, normalized_hla, scores
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = _csv_bytes(ranked_rows)
    out_path.write_bytes(output_payload)
    metadata = {
        "schema_version": 1,
        "code_commit": _git_commit(project_dir),
        "method": METHOD,
        "score_semantics": "research_ranking_score_not_calibrated_probability",
        "clinical_use": "prohibited",
        "command_parameters": {
            "epitope": epitope,
            "hla": hla,
            "tcr_file": str(tcr_file),
            "out": str(out_path),
        },
        "normalized_query": {
            "epitope": normalized_epitope,
            "hla": normalized_hla,
        },
        "input": {
            "path": str(tcr_file),
            "sha256": file_sha256(tcr_file),
            "rows": input_rows,
            "label_column_read": False,
        },
        "checkpoints": [
            {
                "resample_id": row["resample_id"],
                "fold_id": row["fold_id"],
                "path": row["path"],
                "sha256": row["expected_sha256"],
            }
            for row in registry_rows
        ],
        "model_count": len(registry_rows),
        "aggregation": "arithmetic_mean_of_25_model_probabilities",
        "counts": {
            "input_rows": input_rows,
            "exact_duplicate_tcr_rows_removed_after_normalization": duplicate_count,
            "output_rows": len(ranked_rows),
        },
        "output": {
            "path": str(out_path),
            "sha256": file_sha256(out_path),
            "rows": len(ranked_rows),
        },
        "validations": {
            "all_records_scored_by_25_models": all(
                len(values) == EXPECTED_MODEL_COUNT for values in per_record
            ),
            "scores_finite_and_in_unit_interval": True,
            "checkpoint_paths_sizes_and_sha256_match": True,
            "deterministic_sort": "score_descending_then_tcr_lexicographic",
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": "passed",
        "output": str(out_path),
        "metadata": str(metadata_path),
        "input_rows": input_rows,
        "output_rows": len(ranked_rows),
        "duplicates_removed": duplicate_count,
        "contributing_model_count": len(registry_rows),
        "score_semantics": metadata["score_semantics"],
    }


def frozen_fixture_consistency(project_dir: Path) -> dict[str, Any]:
    feature_rows = _load_feature_rows(project_dir / COHORT_TEST_PATH)
    fixture = feature_rows[0]
    if "label" in fixture:
        raise ValueError("label entered the CLI consistency fixture")
    registry_rows, _ = load_orthogonalized_registry(project_dir)
    scores, per_record = score_feature_rows(project_dir, [fixture], registry_rows)
    expected: float | None = None
    with (project_dir / FROZEN_ENSEMBLE_PATH).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if (
                row["method"] == METHOD
                and row["interaction_key"] == fixture["interaction_key"]
            ):
                expected = float(row["ensemble_score"])
                break
    if expected is None:
        raise ValueError("frozen IEDB ensemble fixture score is missing")
    difference = abs(scores[0] - expected)
    if difference > 1e-12:
        raise ValueError(
            f"CLI/frozen ensemble score mismatch: absolute error {difference}"
        )
    return {
        "interaction_key": fixture["interaction_key"],
        "query_id": fixture["query_id"],
        "label_loaded": False,
        "contributing_model_count": len(per_record[0]),
        "cli_score": scores[0],
        "frozen_ensemble_score": expected,
        "absolute_error": difference,
        "tolerance": 1e-12,
        "status": "passed",
    }


def _source(project_dir: Path, relative_path: str) -> dict[str, str]:
    path = project_dir / relative_path
    return {
        "path": relative_path,
        "source_commit": BASE_COMMIT,
        "sha256": file_sha256(path),
    }


def build_prediction_cli_manifest(project_dir: Path, output_path: Path) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    registry_rows, _ = load_orthogonalized_registry(project_dir)
    fixture = frozen_fixture_consistency(project_dir)
    output_paths = (
        "predict.py",
        "src/predict_cli.py",
        "docs/prediction_cli.md",
        "examples/candidate_tcrs.csv",
    )
    manifest = {
        "schema_version": 1,
        "generation_base_commit": BASE_COMMIT,
        "method": METHOD,
        "checkpoint_design": {
            "resample_ids": list(EXPECTED_RESAMPLE_IDS),
            "fold_ids": list(EXPECTED_FOLD_IDS),
            "model_count": len(registry_rows),
            "aggregation": "arithmetic_mean_of_25_model_probabilities",
            "registry": _source(
                project_dir,
                "results/external/iedb/external_checkpoint_registry.csv",
            ),
            "checkpoints": [
                {
                    "resample_id": row["resample_id"],
                    "fold_id": row["fold_id"],
                    "path": row["path"],
                    "sha256": row["expected_sha256"],
                }
                for row in registry_rows
            ],
        },
        "normalization_sources": [
            _source(project_dir, "src/tcrue/vdjdb.py"),
            _source(project_dir, "src/tcrue/iedb_external_audit.py"),
            _source(project_dir, "configs/external_iedb_audit.yaml"),
        ],
        "feature_source": _source(
            project_dir, "src/degree_matched_orthogonalized_logistic.py"
        ),
        "frozen_score_consistency": fixture,
        "score_interpretation": {
            "name": "research ranking score",
            "calibrated_probability": False,
            "clinical_decision_use": False,
        },
        "validations": {
            "checkpoint_count_is_25": len(registry_rows) == EXPECTED_MODEL_COUNT,
            "checkpoint_hashes_match": True,
            "all_records_require_all_models": True,
            "scores_finite_and_in_unit_interval": True,
            "label_column_read": False,
            "frozen_ensemble_absolute_error_at_most_1e_12": fixture[
                "absolute_error"
            ]
            <= 1e-12,
        },
        "operations": {
            "training": False,
            "hyperparameter_tuning": False,
            "performance_evaluation": False,
            "statistical_testing": False,
            "plotting": False,
            "fixture_consistency_scoring": True,
        },
        "outputs": {
            Path(path).name: _source(project_dir, path) for path in output_paths
        },
        "completion_matrix_update": {
            "deliverable_id": "command_line_prediction_tool",
            "new_status": "COMPLETE",
            "other_statuses_modified": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def update_prediction_completion_matrix(project_dir: Path, manifest_path: Path) -> None:
    matrix_path = project_dir / "results/final/project_completion_matrix.csv"
    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    targets = [
        row for row in rows if row["deliverable_id"] == "command_line_prediction_tool"
    ]
    if len(targets) != 1:
        raise ValueError("completion matrix prediction-tool row is missing or duplicated")
    evidence = [
        "predict.py",
        "src/predict_cli.py",
        "docs/prediction_cli.md",
        "examples/candidate_tcrs.csv",
        str(manifest_path.relative_to(project_dir)),
    ]
    targets[0].update(
        {
            "status": "COMPLETE",
            "remaining_bucket": "NONE",
            "formal_plot_blocker": "False",
            "evidence_count": str(len(evidence)),
            "evidence_paths": json.dumps(evidence),
            "evidence_source_commits": json.dumps([BASE_COMMIT] * len(evidence)),
            "evidence_sha256": json.dumps(
                [file_sha256(project_dir / path) for path in evidence]
            ),
            "gap": "",
        }
    )
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
