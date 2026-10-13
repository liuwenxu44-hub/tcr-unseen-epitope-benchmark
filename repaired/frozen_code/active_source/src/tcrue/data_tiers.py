from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .vdjdb import CLASSICAL_FULL_RESOLUTION_HLA, file_sha256


PAIR_KEY_COLUMNS = ("tcr_cdr3_beta", "epitope", "hla")
REQUIRED_COLUMNS = {
    *PAIR_KEY_COLUMNS,
    "pmhc_id",
    "confidence_score",
    "label",
}


def _pair_key(record: dict[str, str]) -> tuple[str, str, str]:
    return tuple(record[column] for column in PAIR_KEY_COLUMNS)  # type: ignore[return-value]


def _load_positive_pairs(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input has no header: {path}")
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        records = list(reader)

    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = _pair_key(record)
        if key in seen:
            raise ValueError(f"Duplicate positive pair in source: {key}")
        seen.add(key)
        if record["label"] != "1":
            raise ValueError(f"Non-positive label in positive source: {record['label']}")
        expected_pmhc = f"{record['epitope']}|{record['hla']}"
        if record["pmhc_id"] != expected_pmhc:
            raise ValueError(
                f"Invalid pMHC id for {key}: {record['pmhc_id']} != {expected_pmhc}"
            )
    return list(reader.fieldnames), records


def _confidence_at_least(record: dict[str, str], minimum: float) -> bool:
    try:
        return float(record["confidence_score"]) >= minimum
    except (TypeError, ValueError):
        return False


def _retain_supported_pmhcs(
    records: list[dict[str, str]], minimum: int
) -> list[dict[str, str]]:
    counts = Counter(record["pmhc_id"] for record in records)
    retained = {pmhc_id for pmhc_id, count in counts.items() if count >= minimum}
    output = [record for record in records if record["pmhc_id"] in retained]
    output.sort(
        key=lambda record: (
            record["epitope"],
            record["hla"],
            record["tcr_cdr3_beta"],
        )
    )
    return output


def _write_csv(path: Path, fieldnames: list[str], records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _tier_stats(path: Path, records: list[dict[str, str]]) -> dict[str, Any]:
    pmhc_counts = Counter(record["pmhc_id"] for record in records)
    confidence_counts = Counter(record["confidence_score"] for record in records)
    return {
        "file": path.name,
        "sha256": file_sha256(path),
        "positive_pairs": len(records),
        "unique_tcrs": len({record["tcr_cdr3_beta"] for record in records}),
        "unique_epitopes": len({record["epitope"] for record in records}),
        "unique_hlas": len({record["hla"] for record in records}),
        "unique_pmhcs": len(pmhc_counts),
        "minimum_pmhc_positive_pairs": min(pmhc_counts.values(), default=0),
        "maximum_pmhc_positive_pairs": max(pmhc_counts.values(), default=0),
        "confidence_score_distribution": dict(sorted(confidence_counts.items())),
        "pmhc_positive_pairs": dict(sorted(pmhc_counts.items())),
    }


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("data tier config schema_version must be 1")
    development = config["tiers"]["development"]
    strict = config["tiers"]["strict"]
    if development.get("classical_hla_loci") != ["A", "B", "C"]:
        raise ValueError("development classical_hla_loci must be [A, B, C]")
    if development.get("require_full_resolution_hla") is not True:
        raise ValueError("development must require full-resolution HLA")
    if int(development.get("min_positive_pairs_per_pmhc", 0)) < 1:
        raise ValueError("development minimum pMHC positives must be positive")
    if strict.get("extends") != "development":
        raise ValueError("strict must extend development")
    if float(strict.get("min_confidence_score", -1)) < 0:
        raise ValueError("strict confidence threshold must be non-negative")
    if int(strict.get("min_positive_pairs_per_pmhc", 0)) < 1:
        raise ValueError("strict minimum pMHC positives must be positive")


def render_data_tier_report(manifest: dict[str, Any]) -> str:
    development = manifest["tiers"]["development"]
    strict = manifest["tiers"]["strict"]
    definitions = manifest["definitions"]
    return f"""# Frozen positive data tiers

## Definitions

- Source: `{manifest['source']['file']}`
- Source SHA-256: `{manifest['source']['sha256']}`
- Development: classical full-resolution HLA-A/B/C and at least {definitions['development']['min_positive_pairs_per_pmhc']} positive pairs per pMHC.
- Strict: development plus confidence score >= {definitions['strict']['min_confidence_score']:g}, followed by at least {definitions['strict']['min_positive_pairs_per_pmhc']} positive pairs per pMHC.
- pMHC identity: `epitope|HLA`
- Pair identity: `(CDR3beta, epitope, HLA)`

## Statistics

| Tier | Positive pairs | TCRs | Epitopes | HLAs | pMHCs | Min / max positives per pMHC |
|---|---:|---:|---:|---:|---:|---:|
| Development | {development['positive_pairs']:,} | {development['unique_tcrs']:,} | {development['unique_epitopes']:,} | {development['unique_hlas']:,} | {development['unique_pmhcs']:,} | {development['minimum_pmhc_positive_pairs']:,} / {development['maximum_pmhc_positive_pairs']:,} |
| Strict | {strict['positive_pairs']:,} | {strict['unique_tcrs']:,} | {strict['unique_epitopes']:,} | {strict['unique_hlas']:,} | {strict['unique_pmhcs']:,} | {strict['minimum_pmhc_positive_pairs']:,} / {strict['maximum_pmhc_positive_pairs']:,} |

## Invariants

- Strict pair keys are a subset of development: `{str(manifest['invariants']['strict_is_development_subset']).lower()}`
- Development duplicate pair keys: `{manifest['invariants']['development_duplicate_pair_keys']}`
- Strict duplicate pair keys: `{manifest['invariants']['strict_duplicate_pair_keys']}`
- Deterministic row ordering: epitope, HLA, CDR3beta

This report is generated by `scripts/build_data_tiers.py` from `configs/data_tiers.yaml`.
"""


def build_data_tiers(
    source_path: Path,
    config_path: Path,
    config: dict[str, Any],
    development_path: Path,
    strict_path: Path,
    manifest_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    _validate_config(config)
    fieldnames, source_records = _load_positive_pairs(source_path)
    development_config = config["tiers"]["development"]
    strict_config = config["tiers"]["strict"]

    development_candidates = [
        record
        for record in source_records
        if CLASSICAL_FULL_RESOLUTION_HLA.fullmatch(record["hla"])
    ]
    development = _retain_supported_pmhcs(
        development_candidates,
        int(development_config["min_positive_pairs_per_pmhc"]),
    )

    strict_candidates = [
        record
        for record in development
        if _confidence_at_least(record, float(strict_config["min_confidence_score"]))
    ]
    strict = _retain_supported_pmhcs(
        strict_candidates,
        int(strict_config["min_positive_pairs_per_pmhc"]),
    )

    development_keys = {_pair_key(record) for record in development}
    strict_keys = {_pair_key(record) for record in strict}
    if len(development_keys) != len(development):
        raise ValueError("Development tier contains duplicate pair keys")
    if len(strict_keys) != len(strict):
        raise ValueError("Strict tier contains duplicate pair keys")
    if not strict_keys <= development_keys:
        raise ValueError("Strict tier is not a subset of development")

    _write_csv(development_path, fieldnames, development)
    _write_csv(strict_path, fieldnames, strict)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "file": source_path.name,
            "sha256": file_sha256(source_path),
            "positive_pairs": len(source_records),
        },
        "config": {
            "file": config_path.name,
            "sha256": file_sha256(config_path),
        },
        "definitions": {
            "development": {
                "classical_hla_loci": development_config["classical_hla_loci"],
                "require_full_resolution_hla": True,
                "min_positive_pairs_per_pmhc": int(
                    development_config["min_positive_pairs_per_pmhc"]
                ),
            },
            "strict": {
                "extends": "development",
                "min_confidence_score": float(
                    strict_config["min_confidence_score"]
                ),
                "min_positive_pairs_per_pmhc": int(
                    strict_config["min_positive_pairs_per_pmhc"]
                ),
            },
            "pmhc_id": "epitope|hla",
            "pair_key": list(PAIR_KEY_COLUMNS),
        },
        "tiers": {
            "development": _tier_stats(development_path, development),
            "strict": _tier_stats(strict_path, strict),
        },
        "invariants": {
            "strict_is_development_subset": strict_keys <= development_keys,
            "development_duplicate_pair_keys": len(development) - len(development_keys),
            "strict_duplicate_pair_keys": len(strict) - len(strict_keys),
        },
        "artifacts": {
            "manifest_json": manifest_path.name,
            "report_md": report_path.name,
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_data_tier_report(manifest), encoding="utf-8")
    return manifest
