from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
HUMAN_SPECIES = frozenset({"HOMOSAPIENS", "HOMO SAPIENS", "HUMAN"})
VDJDB_CONTENT_VERSION = "2026-06-03"
CLASSICAL_FULL_RESOLUTION_HLA = re.compile(
    r"^HLA-[ABC]\*\d{2,3}:\d{2,3}(?::\d{2,3})*$"
)

OUTPUT_COLUMNS = [
    "tcr_cdr3_beta",
    "epitope",
    "hla",
    "pmhc_id",
    "mhc_class",
    "species",
    "antigen_source",
    "antigen_gene",
    "reference_id",
    "source_database",
    "assay_type",
    "confidence_score",
    "record_id",
    "source_record_count",
    "label",
]

REQUIRED_COLUMNS = {
    "cdr3.beta",
    "species",
    "mhc.a",
    "mhc.class",
    "antigen.epitope",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_hla(value: str) -> str:
    return "".join(value.upper().split())


def valid_amino_acid_sequence(value: str, minimum: int, maximum: int) -> bool:
    return minimum <= len(value) <= maximum and set(value) <= VALID_AMINO_ACIDS


def _parse_json_object(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clean_text(value: str | None) -> str:
    return (value or "").strip()


def _confidence_value(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_id(tcr: str, epitope: str, hla: str) -> str:
    payload = "\t".join((tcr, epitope, hla)).encode("ascii")
    return f"vdjdb-pair:{hashlib.sha256(payload).hexdigest()[:16]}"


def _join(values: set[str]) -> str:
    return "|".join(sorted(value for value in values if value))


def _threshold_counts(records: list[dict[str, Any]], thresholds: list[int]) -> dict[str, int]:
    counts = Counter(record["pmhc_id"] for record in records)
    return {
        str(threshold): sum(count >= threshold for count in counts.values())
        for threshold in thresholds
    }


def render_feasibility_report(stats: dict[str, Any]) -> str:
    source = stats["source"]
    filters = stats["filters"]
    rows = stats["rows"]
    output = stats["output"]
    quality = stats["quality_audit"]
    classical = quality["classical_full_resolution_hla"]
    strict = quality["confidence_score_at_least_1_and_classical_full_resolution_hla"]
    thresholds = output["pmhcs_by_minimum_positive_pairs"]
    confidence_zero = quality["confidence_score_distribution"].get("0.0", 0)
    top_five_percent = quality["top_5_pmhc_pair_share"] * 100

    return f"""# VDJdb feasibility report

## Scope

- VDJdb content release: `{source['content_version']}`
- Species: `{filters['species']}`
- TCR chain: TRB CDR3
- MHC class: `{filters['mhc_class']}`
- Peptide length: {filters['min_epitope_length']}-{filters['max_epitope_length']} amino acids
- TCR requirement: standard amino acids, length {filters['min_tcr_length']}-{filters['max_tcr_length']}, canonical C/F/W boundaries
- Positive-pair identity: unique `(CDR3beta, peptide, HLA)`

## Main counts

| Quantity | Count |
|---|---:|
| Raw VDJdb rows | {rows['raw']:,} |
| Accepted rows before deduplication | {rows['accepted_before_deduplication']:,} |
| Unique positive pairs | {rows['output_positive_pairs']:,} |
| Unique TCRs | {output['unique_tcrs']:,} |
| Unique peptide epitopes | {output['unique_epitopes']:,} |
| Unique HLA annotations | {output['unique_hlas']:,} |
| Unique pMHCs | {output['unique_pmhcs']:,} |

Available pMHCs by minimum number of positive TCRs:

| Minimum positives | pMHCs |
|---:|---:|
| 5 | {thresholds['5']:,} |
| 10 | {thresholds['10']:,} |
| 20 | {thresholds['20']:,} |
| 30 | {thresholds['30']:,} |
| 50 | {thresholds['50']:,} |
| 100 | {thresholds['100']:,} |

## Quality audit

The dataset is large enough for the MVP, but the raw count overstates its effective diversity.

1. The five largest pMHCs contain {top_five_percent:.2f}% of all cleaned positive pairs. Pooled metrics and unweighted minibatches would therefore be dominated by a few pMHCs.
2. Among classical HLA-A/B/C annotations with at least two-field allele resolution, {classical['positive_pairs']:,} positive pairs and {classical['unique_pmhcs']:,} pMHCs remain. Of these, {classical['pmhcs_by_minimum_positive_pairs']['20']:,} pMHCs have at least 20 positives and {classical['pmhcs_by_minimum_positive_pairs']['50']:,} have at least 50.
3. VDJdb confidence score is 0 for {confidence_zero:,} of {rows['output_positive_pairs']:,} pairs. This does not prove that these pairs are wrong, but it means that a strict evaluation cannot rely only on the full table.
4. Requiring both classical full-resolution HLA and VDJdb score >= 1 leaves {strict['positive_pairs']:,} pairs and {strict['unique_pmhcs']:,} pMHCs. There are still {strict['pmhcs_by_minimum_positive_pairs']['20']:,} pMHCs with at least 20 positives and {strict['pmhcs_by_minimum_positive_pairs']['50']:,} with at least 50.
5. Low-resolution HLA values such as `HLA-A*02` remain distinct from `HLA-A*02:01`. They should not be used in the main strict benchmark.

## Decision

The project passes the first data-feasibility gate.

Use two related data tiers:

- Development tier: classical full-resolution HLA, at least 20 positive TCRs per pMHC. This provides {classical['pmhcs_by_minimum_positive_pairs']['20']:,} pMHCs for split and negative-source experiments.
- Strict evaluation tier: classical full-resolution HLA, VDJdb score >= 1, at least 20 positive TCRs per pMHC. This provides {strict['pmhcs_by_minimum_positive_pairs']['20']:,} pMHCs for higher-confidence sensitivity analysis.

Do not train directly on the full empirical frequency distribution. The next data step should create frozen benchmark subsets with per-pMHC caps or balanced sampling, while retaining the uncapped positives for retrieval evaluation and descriptive statistics.

## Reproduction

```bash
python scripts/download_vdjdb.py
python scripts/preprocess_vdjdb.py \\
  --input data/raw/vdjdb_full_2026-06-03.tsv
```

This report is generated by `scripts/preprocess_vdjdb.py` from `data/processed/positive_pairs_stats.json`. Machine-readable pMHC counts are stored in `data/processed/pmhc_counts.csv`.
"""


def preprocess_vdjdb(
    input_path: Path,
    output_path: Path,
    stats_path: Path,
    pmhc_counts_path: Path,
    report_path: Path | None = None,
    *,
    min_tcr_length: int = 5,
    max_tcr_length: int = 30,
    min_epitope_length: int = 8,
    max_epitope_length: int = 15,
    require_canonical_tcr: bool = True,
    min_pmhc_positives: int = 1,
) -> dict[str, Any]:
    if min_pmhc_positives < 1:
        raise ValueError("min_pmhc_positives must be at least 1")

    rejected: Counter[str] = Counter()
    total_rows = 0
    aggregates: dict[tuple[str, str, str], dict[str, Any]] = {}

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Input has no header: {input_path}")
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Missing required VDJdb columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            species = _clean_text(row.get("species"))
            if species.upper() not in HUMAN_SPECIES:
                rejected["non_human"] += 1
                continue

            mhc_class = _clean_text(row.get("mhc.class")).upper()
            if mhc_class not in {"MHCI", "MHC-I", "MHC1"}:
                rejected["not_mhci"] += 1
                continue

            tcr = _clean_text(row.get("cdr3.beta")).upper()
            if not valid_amino_acid_sequence(tcr, min_tcr_length, max_tcr_length):
                rejected["invalid_tcr"] += 1
                continue
            if require_canonical_tcr and not (tcr.startswith("C") and tcr.endswith(("F", "W"))):
                rejected["noncanonical_tcr"] += 1
                continue

            epitope = _clean_text(row.get("antigen.epitope")).upper()
            if not valid_amino_acid_sequence(epitope, min_epitope_length, max_epitope_length):
                rejected["invalid_epitope"] += 1
                continue

            hla = normalize_hla(_clean_text(row.get("mhc.a")))
            if not hla.startswith("HLA-") or any(separator in hla for separator in (",", ";", "|")):
                rejected["invalid_or_ambiguous_hla"] += 1
                continue

            key = (tcr, epitope, hla)
            method = _parse_json_object(_clean_text(row.get("method")))
            confidence = _confidence_value(_clean_text(row.get("vdjdb.score")))
            reference = _clean_text(row.get("reference.id"))
            antigen_source = _clean_text(row.get("antigen.species"))
            antigen_gene = _clean_text(row.get("antigen.gene"))
            assay = _clean_text(row.get("method.identification")) or _clean_text(
                str(method.get("identification", ""))
            )

            if key not in aggregates:
                aggregates[key] = {
                    "references": set(),
                    "antigen_sources": set(),
                    "antigen_genes": set(),
                    "assays": set(),
                    "confidence_values": [],
                    "source_rows": [],
                }
            aggregate = aggregates[key]
            aggregate["references"].add(reference)
            aggregate["antigen_sources"].add(antigen_source)
            aggregate["antigen_genes"].add(antigen_gene)
            aggregate["assays"].add(assay)
            if confidence is not None:
                aggregate["confidence_values"].append(confidence)
            aggregate["source_rows"].append(row_number)

    cleaned_records: list[dict[str, Any]] = []
    for (tcr, epitope, hla), aggregate in aggregates.items():
        confidence_values = aggregate["confidence_values"]
        confidence = max(confidence_values) if confidence_values else ""
        cleaned_records.append(
            {
                "tcr_cdr3_beta": tcr,
                "epitope": epitope,
                "hla": hla,
                "pmhc_id": f"{epitope}|{hla}",
                "mhc_class": "MHCI",
                "species": "HomoSapiens",
                "antigen_source": _join(aggregate["antigen_sources"]),
                "antigen_gene": _join(aggregate["antigen_genes"]),
                "reference_id": _join(aggregate["references"]),
                "source_database": "VDJdb",
                "assay_type": _join(aggregate["assays"]),
                "confidence_score": confidence,
                "record_id": _record_id(tcr, epitope, hla),
                "source_record_count": len(aggregate["source_rows"]),
                "label": 1,
            }
        )

    all_pmhc_counts = Counter(record["pmhc_id"] for record in cleaned_records)
    eligible_pmhc = {
        pmhc_id for pmhc_id, count in all_pmhc_counts.items() if count >= min_pmhc_positives
    }
    output_records = [
        record for record in cleaned_records if record["pmhc_id"] in eligible_pmhc
    ]
    output_records.sort(key=lambda row: (row["epitope"], row["hla"], row["tcr_cdr3_beta"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_records)

    pmhc_tcrs: dict[str, set[str]] = defaultdict(set)
    pmhc_epitope: dict[str, str] = {}
    pmhc_hla: dict[str, str] = {}
    for record in output_records:
        pmhc_id = record["pmhc_id"]
        pmhc_tcrs[pmhc_id].add(record["tcr_cdr3_beta"])
        pmhc_epitope[pmhc_id] = record["epitope"]
        pmhc_hla[pmhc_id] = record["hla"]

    pmhc_counts_path.parent.mkdir(parents=True, exist_ok=True)
    with pmhc_counts_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["pmhc_id", "epitope", "hla", "positive_pairs", "unique_tcrs"]
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for pmhc_id, count in sorted(
            ((key, len(value)) for key, value in pmhc_tcrs.items()),
            key=lambda item: (-item[1], item[0]),
        ):
            writer.writerow(
                {
                    "pmhc_id": pmhc_id,
                    "epitope": pmhc_epitope[pmhc_id],
                    "hla": pmhc_hla[pmhc_id],
                    "positive_pairs": count,
                    "unique_tcrs": count,
                }
            )

    hla_counts = Counter(record["hla"] for record in output_records)
    source_counts: Counter[str] = Counter()
    for record in output_records:
        sources = record["antigen_source"].split("|") if record["antigen_source"] else ["unknown"]
        source_counts.update(sources)

    thresholds = [1, 5, 10, 20, 30, 50, 100]
    classical_full_resolution = [
        record
        for record in output_records
        if CLASSICAL_FULL_RESOLUTION_HLA.fullmatch(record["hla"])
    ]
    high_confidence_classical = [
        record
        for record in classical_full_resolution
        if record["confidence_score"] != "" and float(record["confidence_score"]) >= 1
    ]
    confidence_counts = Counter(str(record["confidence_score"]) for record in output_records)
    top_five_pairs = sum(count for _, count in all_pmhc_counts.most_common(5))
    stats: dict[str, Any] = {
        "source": {
            "database": "VDJdb",
            "content_version": VDJDB_CONTENT_VERSION,
            "input_file": input_path.name,
            "input_sha256": file_sha256(input_path),
        },
        "filters": {
            "species": "HomoSapiens",
            "mhc_class": "MHCI",
            "min_tcr_length": min_tcr_length,
            "max_tcr_length": max_tcr_length,
            "min_epitope_length": min_epitope_length,
            "max_epitope_length": max_epitope_length,
            "require_canonical_tcr": require_canonical_tcr,
            "min_pmhc_positives": min_pmhc_positives,
        },
        "rows": {
            "raw": total_rows,
            "accepted_before_deduplication": total_rows - sum(rejected.values()),
            "unique_pairs_before_frequency_filter": len(cleaned_records),
            "output_positive_pairs": len(output_records),
            "duplicate_source_rows_collapsed": (
                total_rows - sum(rejected.values()) - len(cleaned_records)
            ),
            "rejected": dict(sorted(rejected.items())),
        },
        "output": {
            "unique_tcrs": len({record["tcr_cdr3_beta"] for record in output_records}),
            "unique_epitopes": len({record["epitope"] for record in output_records}),
            "unique_hlas": len(hla_counts),
            "unique_pmhcs": len(pmhc_tcrs),
            "pmhcs_by_minimum_positive_pairs": _threshold_counts(output_records, thresholds),
            "top_hlas": dict(hla_counts.most_common(20)),
            "top_antigen_sources": dict(source_counts.most_common(20)),
            "top_pmhcs": dict(all_pmhc_counts.most_common(30)),
        },
        "quality_audit": {
            "top_5_pmhc_pair_share": (
                top_five_pairs / len(output_records) if output_records else 0.0
            ),
            "confidence_score_distribution": dict(sorted(confidence_counts.items())),
            "missing_assay_type_pairs": sum(
                not record["assay_type"] for record in output_records
            ),
            "classical_full_resolution_hla": {
                "pattern": CLASSICAL_FULL_RESOLUTION_HLA.pattern,
                "positive_pairs": len(classical_full_resolution),
                "unique_pmhcs": len(
                    {record["pmhc_id"] for record in classical_full_resolution}
                ),
                "pmhcs_by_minimum_positive_pairs": _threshold_counts(
                    classical_full_resolution, thresholds
                ),
            },
            "confidence_score_at_least_1_and_classical_full_resolution_hla": {
                "positive_pairs": len(high_confidence_classical),
                "unique_pmhcs": len(
                    {record["pmhc_id"] for record in high_confidence_classical}
                ),
                "pmhcs_by_minimum_positive_pairs": _threshold_counts(
                    high_confidence_classical, thresholds
                ),
            },
        },
        "artifacts": {
            "positive_pairs_csv": output_path.name,
            "pmhc_counts_csv": pmhc_counts_path.name,
            "feasibility_report": report_path.name if report_path else None,
        },
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_feasibility_report(stats), encoding="utf-8")
    return stats
