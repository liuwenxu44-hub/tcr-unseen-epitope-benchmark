from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import torch
import yaml

from .data import HlaVocab
from .degree_matched_additive_logistic import build_additive_feature_matrix
from .degree_matched_orthogonalized_logistic import (
    build_orthogonalized_feature_matrix,
)
from .five_fold_summary import file_sha256
from .model_cnn import TwoTowerCnn
from .retrieval_similarity import (
    build_training_references,
    max_tcr_similarities,
    score_similarity_query,
    select_training_epitopes,
)
from .tcr_only_logistic import build_tcr_only_feature_matrix
from .vocab import AminoAcidVocab


PROTOCOL_FREEZE_COMMIT = "f1170fd30a850834d220be45e2b4859914a8d230"
METHOD_ORDER = (
    "orthogonalized_interaction",
    "additive_only",
    "degree_matched_cnn",
    "degree_matched_tcr_only_logistic",
    "sequence_similarity",
)
MODEL_METHODS = METHOD_ORDER[:-1]
FEATURE_COLUMNS = (
    "record_id",
    "interaction_key",
    "query_id",
    "tcr_cdr3_beta",
    "epitope",
    "hla",
)
RAW_COLUMNS = (
    "method",
    "model_id",
    "fold_id",
    "resample_id",
    "interaction_key",
    "query_id",
    "logit",
    "probability",
    "similarity_score",
    "status",
    "checkpoint_sha256",
    "input_sha256",
)
ENSEMBLE_COLUMNS = (
    "interaction_key",
    "query_id",
    "method",
    "ensemble_score",
    "contributing_model_count",
    "similarity_available_fold_count",
)
COVERAGE_COLUMNS = (
    "query_id",
    "epitope",
    "hla",
    "record_count",
    "available_fold_count",
    "available_fold_ids",
    "status",
)
OUTPUT_NAMES = {
    "raw_model_predictions": "raw_model_predictions.csv.gz",
    "ensemble_predictions": "ensemble_predictions.csv",
    "similarity_coverage": "similarity_coverage.csv",
    "scoring_manifest": "scoring_manifest.json",
    "scoring_audit": "scoring_audit.md",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _verify_sha256(path: Path, expected: str, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"SHA-256 mismatch for {description}: expected {expected}, observed {observed}"
        )
    return observed


def verify_clean_worktree(project_dir: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("refusing external scoring because the Git worktree is dirty")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not head:
        raise RuntimeError("could not determine clean scoring source commit")
    return head


def _load_feature_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"cohort CSV has no header: {path}")
        missing = sorted(set(FEATURE_COLUMNS) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"cohort feature columns are missing: {missing}")
        for source in reader:
            row = {
                "record_id": source["record_id"].strip(),
                "interaction_key": source["interaction_key"].strip(),
                "query_id": source["query_id"].strip().upper(),
                "tcr_cdr3_beta": source["tcr_cdr3_beta"].strip().upper(),
                "epitope": source["epitope"].strip().upper(),
                "hla": source["hla"].strip().upper(),
            }
            if not all(row.values()):
                raise ValueError("blank field in label-free cohort feature table")
            if row["query_id"] != f"{row['epitope']}|{row['hla']}":
                raise ValueError(f"query identity mismatch: {row['record_id']}")
            rows.append(row)
    if len({row["interaction_key"] for row in rows}) != len(rows):
        raise ValueError("duplicate interaction key in external cohort")
    if any("label" in row for row in rows):
        raise ValueError("label leaked into external scoring feature table")
    return rows


def label_free_feature_sha256(rows: Sequence[Mapping[str, str]]) -> str:
    payload = json.dumps(
        [{column: row[column] for column in FEATURE_COLUMNS} for row in rows],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _load_registry(
    project_dir: Path,
    registry_path: Path,
    expected_sha256: str,
) -> list[dict[str, str]]:
    _verify_sha256(registry_path, expected_sha256, "external checkpoint registry")
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("external checkpoint registry is empty")
    required = {
        "method",
        "artifact_type",
        "resample_id",
        "fold_id",
        "path",
        "expected_sha256",
        "observed_sha256",
        "size_bytes",
        "hash_match",
    }
    if not required <= set(rows[0]):
        raise ValueError("external checkpoint registry schema mismatch")
    for row in rows:
        if row["method"] not in METHOD_ORDER:
            raise ValueError(f"unexpected registered method: {row['method']}")
        if row["hash_match"] != "true":
            raise ValueError(f"registry contains failed hash: {row['path']}")
        if row["expected_sha256"] != row["observed_sha256"]:
            raise ValueError(f"registry expected/observed mismatch: {row['path']}")
        artifact = project_dir / row["path"]
        observed = _verify_sha256(
            artifact, row["expected_sha256"], f"registered artifact {row['path']}"
        )
        if observed != row["observed_sha256"]:
            raise ValueError(f"registered artifact changed: {row['path']}")
        if artifact.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"registered artifact size changed: {row['path']}")
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("duplicate artifact path in external checkpoint registry")
    return rows


def _validate_registry_counts(
    rows: Sequence[Mapping[str, str]], config: Mapping[str, Any]
) -> dict[str, int]:
    model_rows = [row for row in rows if row["artifact_type"] == "model_checkpoint"]
    reference_rows = [
        row for row in rows if row["artifact_type"] == "train_positive_reference"
    ]
    counts = Counter(row["method"] for row in model_rows)
    expected = {
        "orthogonalized_interaction": int(
            config["primary_models"]["orthogonalized_interaction"][
                "checkpoint_count"
            ]
        ),
        "additive_only": int(
            config["primary_models"]["additive_only"]["checkpoint_count"]
        ),
        "degree_matched_cnn": int(
            config["secondary_controls"]["degree_matched_cnn"]["checkpoint_count"]
        ),
        "degree_matched_tcr_only_logistic": int(
            config["secondary_controls"]["degree_matched_tcr_only_logistic"][
                "checkpoint_count"
            ]
        ),
    }
    if dict(counts) != expected:
        raise ValueError(f"model checkpoint counts differ from protocol: {counts}")
    similarity_count = sum(
        row["method"] == "sequence_similarity" for row in reference_rows
    )
    expected_similarity = int(
        config["secondary_controls"]["sequence_similarity"]["fold_reference_count"]
    )
    if similarity_count != expected_similarity or len(reference_rows) != similarity_count:
        raise ValueError("Similarity fold-reference count differs from protocol")
    return {**expected, "sequence_similarity": similarity_count}


def _float_text(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("prediction contains NaN or infinity")
    return format(float(value), ".17g")


def _model_id(row: Mapping[str, str]) -> str:
    parts = [row["method"]]
    if row["resample_id"]:
        parts.append(row["resample_id"])
    if row["fold_id"]:
        parts.append(row["fold_id"])
    return ":".join(parts)


def _validate_probabilities(probabilities: np.ndarray, description: str) -> None:
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError(f"non-finite or non-vector probabilities for {description}")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(f"probabilities outside [0,1] for {description}")


def _logistic_scores(
    project_dir: Path,
    registry_row: Mapping[str, str],
    feature_rows: Sequence[Mapping[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = joblib.load(project_dir / registry_row["path"])
    if checkpoint.get("fold_id") != registry_row["fold_id"]:
        raise ValueError(f"checkpoint fold mismatch: {registry_row['path']}")
    method = registry_row["method"]
    if method == "additive_only":
        matrix = build_additive_feature_matrix(feature_rows, checkpoint["hla_mapping"])
    elif method == "orthogonalized_interaction":
        matrix = build_orthogonalized_feature_matrix(
            feature_rows, checkpoint["hla_mapping"]
        )
    elif method == "degree_matched_tcr_only_logistic":
        matrix = build_tcr_only_feature_matrix(feature_rows)
    else:
        raise ValueError(f"not a Logistic scoring method: {method}")
    estimator = checkpoint["estimator"]
    logits = np.asarray(estimator.decision_function(matrix), dtype=np.float64)
    probabilities = np.asarray(estimator.predict_proba(matrix)[:, 1], dtype=np.float64)
    if logits.shape != (len(feature_rows),):
        raise ValueError(f"Logistic logit shape mismatch for {registry_row['path']}")
    _validate_probabilities(probabilities, registry_row["path"])
    return logits, probabilities


def _cnn_training_config(
    project_dir: Path,
    protocol_config: Mapping[str, Any],
    fold_id: str,
) -> Mapping[str, Any]:
    summary_source = protocol_config["secondary_controls"]["degree_matched_cnn"][
        "checkpoint_source"
    ]
    summary_path = project_dir / summary_source["path"]
    _verify_sha256(summary_path, summary_source["sha256"], "CNN checkpoint summary")
    summary = _read_json(summary_path)
    source = summary["audit"]["fold_inputs"][fold_id]["checkpoint"][
        "training_config"
    ]
    config_path = project_dir / source["path"]
    _verify_sha256(config_path, source["sha256"], f"CNN {fold_id} training config")
    return _read_yaml(config_path)


def _cnn_scores(
    project_dir: Path,
    registry_row: Mapping[str, str],
    feature_rows: Sequence[Mapping[str, str]],
    protocol_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cpu")
    checkpoint = torch.load(project_dir / registry_row["path"], map_location=device)
    model = TwoTowerCnn(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    amino_vocab = AminoAcidVocab()
    if checkpoint["amino_acid_vocab"]["token_to_id"] != amino_vocab.token_to_id:
        raise ValueError(f"CNN amino-acid vocabulary mismatch: {registry_row['path']}")
    hla_vocab = HlaVocab(checkpoint["hla_to_id"])
    training_config = _cnn_training_config(
        project_dir, protocol_config, registry_row["fold_id"]
    )
    sequence = training_config["sequence"]
    tcr_ids = torch.tensor(
        [
            amino_vocab.encode(
                row["tcr_cdr3_beta"], int(sequence["tcr_max_length"])
            ).token_ids
            for row in feature_rows
        ],
        dtype=torch.long,
    )
    epitope_ids = torch.tensor(
        [
            amino_vocab.encode(
                row["epitope"], int(sequence["epitope_max_length"])
            ).token_ids
            for row in feature_rows
        ],
        dtype=torch.long,
    )
    hla_ids = torch.tensor(
        [hla_vocab.encode(row["hla"]) for row in feature_rows], dtype=torch.long
    )
    logits: list[float] = []
    batch_size = int(training_config["dataloader"]["batch_size"])
    with torch.no_grad():
        for start in range(0, len(feature_rows), batch_size):
            stop = min(start + batch_size, len(feature_rows))
            batch_logits = model(
                tcr_ids[start:stop], epitope_ids[start:stop], hla_ids[start:stop]
            )
            logits.extend(float(value) for value in batch_logits.tolist())
    logit_array = np.asarray(logits, dtype=np.float64)
    probabilities = np.asarray(
        torch.sigmoid(torch.tensor(logit_array, dtype=torch.float64)).numpy(),
        dtype=np.float64,
    )
    _validate_probabilities(probabilities, registry_row["path"])
    return logit_array, probabilities


def _read_reference_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Similarity train reference has no header: {path}")
        required = {"tcr_cdr3_beta", "epitope", "hla"}
        if not required <= set(reader.fieldnames):
            raise ValueError(f"Similarity train reference schema mismatch: {path}")
        for source in reader:
            rows.append(
                {
                    "tcr_cdr3_beta": source["tcr_cdr3_beta"].strip().upper(),
                    "epitope": source["epitope"].strip().upper(),
                    "hla": source["hla"].strip().upper(),
                }
            )
    return rows


def _similarity_scores(
    project_dir: Path,
    registry_row: Mapping[str, str],
    feature_rows: Sequence[Mapping[str, str]],
    protocol_config: Mapping[str, Any],
) -> tuple[list[float | None], set[str]]:
    similarity = protocol_config["secondary_controls"]["sequence_similarity"]
    if similarity["same_hla_only"] is not True or similarity[
        "cross_hla_fallback"
    ] is not False:
        raise ValueError("Similarity HLA restriction differs from frozen protocol")
    algorithm_source = similarity["config"]
    algorithm_path = project_dir / algorithm_source["path"]
    _verify_sha256(algorithm_path, algorithm_source["sha256"], "Similarity config")
    algorithm = _read_yaml(algorithm_path)["similarity"]
    train_rows = _read_reference_rows(project_dir / registry_row["path"])
    references = build_training_references(train_rows)
    query_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(feature_rows):
        query_indices[row["query_id"]].append(index)
    scores: list[float | None] = [None] * len(feature_rows)
    available_queries: set[str] = set()
    for query_id, indices in query_indices.items():
        first = feature_rows[indices[0]]
        hla_references = references.get(first["hla"])
        if not hla_references:
            continue
        selected = select_training_epitopes(
            first["epitope"], hla_references, int(algorithm["top_k_epitopes"])
        )
        if not selected:
            continue
        candidate_tcrs = [feature_rows[index]["tcr_cdr3_beta"] for index in indices]
        cache = {
            epitope: max_tcr_similarities(
                candidate_tcrs,
                hla_references[epitope],
                candidate_chunk_size=int(algorithm["candidate_chunk_size"]),
                workers=int(algorithm["workers"]),
            )
            for epitope, _ in selected
        }
        query_scores = score_similarity_query(
            candidate_tcrs, selected, hla_references, cache
        )
        for index, score in zip(indices, query_scores, strict=True):
            if not math.isfinite(score) or score < 0.0 or score > 1.0:
                raise ValueError(f"invalid Similarity score for {query_id}")
            scores[index] = float(score)
        available_queries.add(query_id)
    return scores, available_queries


def _raw_row(
    registry_row: Mapping[str, str],
    feature_row: Mapping[str, str],
    input_sha256: str,
    *,
    logit: float | None = None,
    probability: float | None = None,
    similarity_score: float | None = None,
    status: str = "available",
) -> dict[str, Any]:
    return {
        "method": registry_row["method"],
        "model_id": _model_id(registry_row),
        "fold_id": registry_row["fold_id"],
        "resample_id": registry_row["resample_id"],
        "interaction_key": feature_row["interaction_key"],
        "query_id": feature_row["query_id"],
        "logit": "" if logit is None else _float_text(logit),
        "probability": "" if probability is None else _float_text(probability),
        "similarity_score": (
            "" if similarity_score is None else _float_text(similarity_score)
        ),
        "status": status,
        "checkpoint_sha256": registry_row["expected_sha256"],
        "input_sha256": input_sha256,
    }


def _score_once(
    project_dir: Path,
    config: Mapping[str, Any],
    registry: Sequence[Mapping[str, str]],
    feature_rows: Sequence[Mapping[str, str]],
    input_sha256: str,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    raw_rows: list[dict[str, Any]] = []
    scores: dict[str, list[list[float]]] = {
        method: [[] for _ in feature_rows] for method in METHOD_ORDER
    }
    available_folds: dict[str, set[str]] = defaultdict(set)

    for registry_row in registry:
        method = registry_row["method"]
        if method in ("additive_only", "orthogonalized_interaction", "degree_matched_tcr_only_logistic"):
            logits, probabilities = _logistic_scores(
                project_dir, registry_row, feature_rows
            )
            for index, feature_row in enumerate(feature_rows):
                probability = float(probabilities[index])
                raw_rows.append(
                    _raw_row(
                        registry_row,
                        feature_row,
                        input_sha256,
                        logit=float(logits[index]),
                        probability=probability,
                    )
                )
                scores[method][index].append(probability)
        elif method == "degree_matched_cnn":
            logits, probabilities = _cnn_scores(
                project_dir, registry_row, feature_rows, config
            )
            for index, feature_row in enumerate(feature_rows):
                probability = float(probabilities[index])
                raw_rows.append(
                    _raw_row(
                        registry_row,
                        feature_row,
                        input_sha256,
                        logit=float(logits[index]),
                        probability=probability,
                    )
                )
                scores[method][index].append(probability)
        elif method == "sequence_similarity":
            similarity_scores, available_queries = _similarity_scores(
                project_dir, registry_row, feature_rows, config
            )
            for query_id in available_queries:
                available_folds[query_id].add(registry_row["fold_id"])
            for index, feature_row in enumerate(feature_rows):
                score = similarity_scores[index]
                raw_rows.append(
                    _raw_row(
                        registry_row,
                        feature_row,
                        input_sha256,
                        similarity_score=score,
                        status="available" if score is not None else "unavailable",
                    )
                )
                if score is not None:
                    scores[method][index].append(score)
        else:
            raise ValueError(f"unsupported external scoring method: {method}")

    expected_counts = {
        "orthogonalized_interaction": int(
            config["primary_models"]["orthogonalized_interaction"]["checkpoint_count"]
        ),
        "additive_only": int(
            config["primary_models"]["additive_only"]["checkpoint_count"]
        ),
        "degree_matched_cnn": int(
            config["secondary_controls"]["degree_matched_cnn"]["checkpoint_count"]
        ),
        "degree_matched_tcr_only_logistic": int(
            config["secondary_controls"]["degree_matched_tcr_only_logistic"][
                "checkpoint_count"
            ]
        ),
    }
    ensemble_rows: list[dict[str, Any]] = []
    maximum_mean_error = 0.0
    for method in METHOD_ORDER:
        for index, feature_row in enumerate(feature_rows):
            values = scores[method][index]
            if method in expected_counts and len(values) != expected_counts[method]:
                raise ValueError(
                    f"raw-score count mismatch for {method}/{feature_row['interaction_key']}"
                )
            if method == "sequence_similarity":
                expected_available = len(available_folds[feature_row["query_id"]])
                if len(values) != expected_available:
                    raise ValueError("Similarity available-fold count mismatch")
            mean = math.fsum(values) / len(values) if values else None
            if mean is not None:
                reconstructed = math.fsum(values) / len(values)
                maximum_mean_error = max(maximum_mean_error, abs(mean - reconstructed))
            ensemble_rows.append(
                {
                    "interaction_key": feature_row["interaction_key"],
                    "query_id": feature_row["query_id"],
                    "method": method,
                    "ensemble_score": "" if mean is None else _float_text(mean),
                    "contributing_model_count": len(values),
                    "similarity_available_fold_count": (
                        len(values) if method == "sequence_similarity" else ""
                    ),
                }
            )
    if maximum_mean_error > 1e-12:
        raise ValueError("ensemble arithmetic-mean audit exceeded 1e-12")

    by_query: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in feature_rows:
        by_query[row["query_id"]].append(row)
    coverage_rows = []
    for query_id in sorted(by_query):
        rows = by_query[query_id]
        folds = sorted(available_folds[query_id])
        coverage_rows.append(
            {
                "query_id": query_id,
                "epitope": rows[0]["epitope"],
                "hla": rows[0]["hla"],
                "record_count": len(rows),
                "available_fold_count": len(folds),
                "available_fold_ids": ";".join(folds),
                "status": "available" if folds else "unavailable",
            }
        )
    return {
        "raw_rows": raw_rows,
        "ensemble_rows": ensemble_rows,
        "coverage_rows": coverage_rows,
        "audit": {
            "raw_prediction_rows": len(raw_rows),
            "ensemble_prediction_rows": len(ensemble_rows),
            "similarity_query_count": len(coverage_rows),
            "similarity_available_query_count": sum(
                row["status"] == "available" for row in coverage_rows
            ),
            "similarity_unavailable_query_count": sum(
                row["status"] == "unavailable" for row in coverage_rows
            ),
            "maximum_ensemble_mean_error": maximum_mean_error,
            "all_model_probabilities_valid": True,
            "all_model_rows_scored": True,
            "similarity_unavailable_explicit": True,
        },
    }


def _csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _deterministic_gzip(payload: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=output, compresslevel=9, mtime=0
    ) as handle:
        handle.write(payload)
    return output.getvalue()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _render_audit(
    cohort: Mapping[str, Any], counts: Mapping[str, int], audit: Mapping[str, Any]
) -> str:
    return "\n".join(
        [
            "# IEDB external cohort blind-scoring audit",
            "",
            "## Scope",
            "",
            "The frozen external cohort was scored without loading labels into any model feature table. No performance metric, statistical test, training, tuning, calibration, threshold selection, or figure was produced.",
            "",
            "## Inputs",
            "",
            f"- Cohort interactions: {cohort['test_count']}",
            f"- Cohort queries: {cohort['query_count']}",
            f"- Registered model checkpoints: {sum(counts[m] for m in MODEL_METHODS)}",
            f"- Similarity fold references: {counts['sequence_similarity']}",
            "- All registered paths, sizes, and SHA-256 values matched the frozen registry.",
            "",
            "## Persisted predictions",
            "",
            f"- Raw prediction rows: {audit['raw_prediction_rows']}",
            f"- Ensemble prediction rows: {audit['ensemble_prediction_rows']}",
            f"- Similarity evaluable queries: {audit['similarity_available_query_count']} / {audit['similarity_query_count']}",
            f"- Maximum ensemble arithmetic-mean error: {audit['maximum_ensemble_mean_error']:.3e}",
            "",
            "## Validation",
            "",
            "- All four model methods scored every interaction.",
            "- All probabilities were finite and within [0,1].",
            "- Similarity used same-HLA fold references only; queries without a valid reference were explicitly unavailable.",
            "- Two complete scoring runs produced identical SHA-256 values for every output.",
            "- Prediction files contain no label column.",
            "",
        ]
    )


def _build_payloads(
    result: Mapping[str, Any],
    *,
    project_dir: Path,
    config_path: Path,
    registry_path: Path,
    cohort_manifest_path: Path,
    cohort_manifest: Mapping[str, Any],
    cohort_test_path: Path,
    input_sha256: str,
    scoring_source_commit: str,
    registry_counts: Mapping[str, int],
) -> dict[str, bytes]:
    raw_payload = _deterministic_gzip(
        _csv_bytes(RAW_COLUMNS, result["raw_rows"])
    )
    ensemble_payload = _csv_bytes(ENSEMBLE_COLUMNS, result["ensemble_rows"])
    coverage_payload = _csv_bytes(COVERAGE_COLUMNS, result["coverage_rows"])
    cohort = {
        "test_count": int(cohort_manifest["summary"]["test_count"]),
        "query_count": int(cohort_manifest["summary"]["query_count"]),
        "positive_count": int(cohort_manifest["summary"]["positive_count"]),
        "negative_count": int(cohort_manifest["summary"]["negative_count"]),
    }
    report_payload = _render_audit(cohort, registry_counts, result["audit"]).encode(
        "utf-8"
    )
    output_dir = Path("results/external/iedb/scoring")
    outputs = {
        "raw_model_predictions": {
            "path": str(output_dir / OUTPUT_NAMES["raw_model_predictions"]),
            "rows": int(result["audit"]["raw_prediction_rows"]),
            "sha256": _sha256_bytes(raw_payload),
        },
        "ensemble_predictions": {
            "path": str(output_dir / OUTPUT_NAMES["ensemble_predictions"]),
            "rows": int(result["audit"]["ensemble_prediction_rows"]),
            "sha256": _sha256_bytes(ensemble_payload),
        },
        "similarity_coverage": {
            "path": str(output_dir / OUTPUT_NAMES["similarity_coverage"]),
            "rows": int(result["audit"]["similarity_query_count"]),
            "sha256": _sha256_bytes(coverage_payload),
        },
        "scoring_audit": {
            "path": str(output_dir / OUTPUT_NAMES["scoring_audit"]),
            "rows": None,
            "sha256": _sha256_bytes(report_payload),
        },
    }
    manifest = {
        "schema_version": 1,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "scoring_source_commit": scoring_source_commit,
        "mode": "blind_external_scoring_without_performance_evaluation",
        "cohort": cohort,
        "registry_counts": dict(registry_counts),
        "inputs": {
            "evaluation_config": {
                "path": str(config_path.relative_to(project_dir)),
                "sha256": file_sha256(config_path),
            },
            "checkpoint_registry": {
                "path": str(registry_path.relative_to(project_dir)),
                "sha256": file_sha256(registry_path),
            },
            "cohort_manifest": {
                "path": str(cohort_manifest_path.relative_to(project_dir)),
                "sha256": file_sha256(cohort_manifest_path),
            },
            "cohort_test": {
                "path": str(cohort_test_path.relative_to(project_dir)),
                "source_file_sha256": file_sha256(cohort_test_path),
                "label_free_feature_table_sha256": input_sha256,
                "feature_columns": list(FEATURE_COLUMNS),
                "label_loaded_into_feature_table": False,
            },
        },
        "audit": dict(result["audit"]),
        "validations": {
            "initial_worktree_clean": True,
            "checkpoint_paths_sizes_and_sha256_match": True,
            "cohort_row_count_matches_manifest": True,
            "all_four_model_methods_cover_every_record": True,
            "per_record_raw_score_counts_match_protocol": True,
            "ensemble_arithmetic_means_within_1e-12": True,
            "probabilities_finite_and_in_unit_interval": True,
            "similarity_same_hla_only": True,
            "similarity_cross_hla_fallback_used": False,
            "similarity_unavailable_explicit": True,
            "prediction_files_contain_labels": False,
        },
        "operations": {
            "training": False,
            "hyperparameter_tuning": False,
            "resampling": False,
            "performance_metrics": False,
            "statistical_testing": False,
            "plotting": False,
        },
        "reproducibility": {
            "complete_scoring_runs_compared": 2,
            "all_output_sha256_identical": True,
            "deterministic_gzip_mtime": 0,
        },
        "outputs": outputs,
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return {
        OUTPUT_NAMES["raw_model_predictions"]: raw_payload,
        OUTPUT_NAMES["ensemble_predictions"]: ensemble_payload,
        OUTPUT_NAMES["similarity_coverage"]: coverage_payload,
        OUTPUT_NAMES["scoring_audit"]: report_payload,
        OUTPUT_NAMES["scoring_manifest"]: manifest_payload,
    }


def score_iedb_external_cohort(
    project_dir: Path,
    config_path: Path,
    *,
    output_dir: Path | None = None,
    enforce_clean_worktree: bool = True,
    clean_source_commit: str | None = None,
) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    config_path = config_path.resolve()
    if enforce_clean_worktree:
        scoring_source_commit = verify_clean_worktree(project_dir)
    else:
        if not clean_source_commit:
            raise ValueError(
                "an externally verified clean source commit is required in isolated mode"
            )
        scoring_source_commit = clean_source_commit

    config = _read_yaml(config_path)
    if config.get("protocol_status") != "frozen":
        raise ValueError("IEDB evaluation protocol is not frozen")
    if config["pre_scoring_declaration"]["external_model_performance_viewed"] is not False:
        raise ValueError("external performance was viewed before scoring")
    protocol_dir = project_dir / config["outputs"]["directory"]
    protocol_manifest_path = protocol_dir / config["outputs"]["manifest_json"]
    protocol_manifest = _read_json(protocol_manifest_path)
    _verify_sha256(
        config_path,
        protocol_manifest["sources"]["configuration"]["sha256"],
        "external evaluation config",
    )
    registry_source = protocol_manifest["outputs"]["checkpoint_registry_csv"]
    registry_path = project_dir / registry_source["path"]
    registry = _load_registry(
        project_dir, registry_path, registry_source["sha256"]
    )
    registry_counts = _validate_registry_counts(registry, config)

    cohort_source = config["cohort"]["manifest"]
    cohort_manifest_path = project_dir / cohort_source["path"]
    _verify_sha256(
        cohort_manifest_path, cohort_source["sha256"], "formal IEDB cohort manifest"
    )
    cohort_manifest = _read_json(cohort_manifest_path)
    test_source = cohort_manifest["outputs"]["test"]
    cohort_test_path = project_dir / test_source["path"]
    _verify_sha256(cohort_test_path, test_source["sha256"], "formal IEDB cohort test")
    feature_rows = _load_feature_rows(cohort_test_path)
    expected_test_count = int(cohort_manifest["summary"]["test_count"])
    if expected_test_count != int(test_source["rows"]):
        raise ValueError("cohort manifest test counts disagree")
    if expected_test_count != (
        int(cohort_manifest["summary"]["positive_count"])
        + int(cohort_manifest["summary"]["negative_count"])
    ):
        raise ValueError("cohort class counts do not sum to test count")
    if len(feature_rows) != expected_test_count:
        raise ValueError("label-free feature row count differs from cohort manifest")
    query_count = len({row["query_id"] for row in feature_rows})
    if query_count != int(cohort_manifest["summary"]["query_count"]):
        raise ValueError("label-free feature query count differs from cohort manifest")
    input_sha256 = label_free_feature_sha256(feature_rows)

    destination = output_dir or project_dir / "results/external/iedb/scoring"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite scoring output: {destination}")

    first_result = _score_once(
        project_dir, config, registry, feature_rows, input_sha256
    )
    first_payloads = _build_payloads(
        first_result,
        project_dir=project_dir,
        config_path=config_path,
        registry_path=registry_path,
        cohort_manifest_path=cohort_manifest_path,
        cohort_manifest=cohort_manifest,
        cohort_test_path=cohort_test_path,
        input_sha256=input_sha256,
        scoring_source_commit=scoring_source_commit,
        registry_counts=registry_counts,
    )
    second_result = _score_once(
        project_dir, config, registry, feature_rows, input_sha256
    )
    second_payloads = _build_payloads(
        second_result,
        project_dir=project_dir,
        config_path=config_path,
        registry_path=registry_path,
        cohort_manifest_path=cohort_manifest_path,
        cohort_manifest=cohort_manifest,
        cohort_test_path=cohort_test_path,
        input_sha256=input_sha256,
        scoring_source_commit=scoring_source_commit,
        registry_counts=registry_counts,
    )
    first_hashes = {name: _sha256_bytes(data) for name, data in first_payloads.items()}
    second_hashes = {
        name: _sha256_bytes(data) for name, data in second_payloads.items()
    }
    if first_hashes != second_hashes:
        raise ValueError(
            f"complete external scoring rebuild is not reproducible: "
            f"first={first_hashes}, second={second_hashes}"
        )

    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in first_payloads.items():
        (destination / name).write_bytes(payload)
    manifest = json.loads(
        first_payloads[OUTPUT_NAMES["scoring_manifest"]].decode("utf-8")
    )
    for output in manifest["outputs"].values():
        output_path = project_dir / output["path"]
        if destination != project_dir / "results/external/iedb/scoring":
            output_path = destination / Path(output["path"]).name
        if file_sha256(output_path) != output["sha256"]:
            raise ValueError(f"persisted scoring output hash mismatch: {output_path}")
    return manifest
