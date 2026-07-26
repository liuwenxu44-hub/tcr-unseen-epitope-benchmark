from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .degree_matched_negatives import ExactDegreeMatchError, solve_degree_matched_partition
from .distance_aware_split import normalized_levenshtein_similarity
from .vdjdb import (
    CLASSICAL_FULL_RESOLUTION_HLA,
    file_sha256,
    normalize_hla,
    valid_amino_acid_sequence,
)


PairKey = tuple[str, str, str]
QueryKey = tuple[str, str]
FILTER_FIELDNAMES = (
    "step",
    "filter",
    "input_records",
    "retained_records",
    "removed_records",
    "retained_fraction_of_raw",
)
OVERLAP_FIELDNAMES = (
    "metric",
    "unit",
    "iedb_total",
    "overlap_with_vdjdb",
    "overlap_fraction",
)
TIER_FIELDNAMES = (
    "tier",
    "definition",
    "positive_records",
    "unique_tcrs",
    "unique_peptides",
    "unique_pmhc_queries",
    "unique_hlas",
    "unique_publications",
    "queries_min_1_positive",
    "queries_min_5_positives",
    "queries_min_10_positives",
    "queries_min_20_positives",
    "hla_seen_queries",
    "hla_unseen_queries",
    "pathogen_or_nonself_records",
    "tumor_associated_records",
    "self_antigen_records",
    "other_nonself_records",
    "unknown_source_records",
)
DEGREE_FIELDNAMES = (
    "query_cap",
    "retained_queries",
    "retained_positive_records",
    "exact_solution",
    "maximum_matched_edges",
    "matching_rate",
    "failed_tcr_count",
    "failed_query_count",
    "failure_reason",
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _stable_hash(seed: int, *parts: str) -> str:
    return hashlib.sha256(
        "\0".join((str(seed), *parts)).encode("utf-8")
    ).hexdigest()


def _joined_header(section: str, field: str) -> str:
    return f"{section.strip()}|{field.strip()}"


def _iter_export_rows(zip_path: Path, member: str) -> Iterable[dict[str, str]]:
    with zipfile.ZipFile(zip_path) as archive:
        if member not in archive.namelist():
            raise ValueError(f"missing IEDB export member: {member}")
        with archive.open(member) as binary, io.TextIOWrapper(
            binary, encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.reader(handle)
            try:
                sections = next(reader)
                fields = next(reader)
            except StopIteration as error:
                raise ValueError(f"IEDB export is missing its two-row header: {member}") from error
            if len(sections) != len(fields):
                raise ValueError(f"IEDB export header widths differ: {member}")
            headers = [
                _joined_header(section, field)
                for section, field in zip(sections, fields, strict=True)
            ]
            if len(headers) != len(set(headers)):
                raise ValueError(f"IEDB export contains duplicate joined headers: {member}")
            for row_number, values in enumerate(reader, start=3):
                if len(values) != len(headers):
                    raise ValueError(
                        f"IEDB row {row_number} has {len(values)} values, expected {len(headers)}"
                    )
                yield dict(zip(headers, values, strict=True))


def _reference_number(value: str) -> str:
    match = re.search(r"/reference/(\d+)(?:$|[/?#])", value, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _canonical_publication_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for part in re.split(r"[|;,\s]+", value.strip()):
        cleaned = part.strip().rstrip(".)]")
        if not cleaned:
            continue
        pmid = re.search(r"(?:PMID[:/_-]?|pubmed(?:\.ncbi\.nlm\.nih\.gov)?/)(\d+)", cleaned, re.I)
        if pmid:
            tokens.add(f"PMID:{pmid.group(1)}")
        doi = re.search(r"(?:DOI[:/]|doi\.org/)(10\.\d{4,9}/[^\s?#]+)", cleaned, re.I)
        if doi:
            tokens.add(f"DOI:{doi.group(1).rstrip('/').lower()}")
    if value.strip().isdigit():
        tokens.add(f"PMID:{value.strip()}")
    return tokens


def load_iedb_references(zip_path: Path, member: str) -> tuple[dict[str, dict[str, Any]], int]:
    references: dict[str, dict[str, Any]] = {}
    row_count = 0
    for row in _iter_export_rows(zip_path, member):
        row_count += 1
        iri = _clean(row.get("Reference ID|IEDB IRI"))
        reference_id = _reference_number(iri)
        if not reference_id:
            raise ValueError(f"cannot parse IEDB reference IRI: {iri}")
        pmid = _clean(row.get("Reference|PMID"))
        alternate = _clean(row.get("Reference|Alternate IRIs"))
        publication_tokens = _canonical_publication_tokens(pmid)
        publication_tokens.update(_canonical_publication_tokens(alternate))
        references[reference_id] = {
            "iedb_reference_id": f"IEDB_REF:{reference_id}",
            "reference_iri": iri,
            "reference_type": _clean(row.get("Reference|Type")),
            "pmid": pmid,
            "alternate_iris": alternate,
            "publication_tokens": publication_tokens,
            "title": _clean(row.get("Reference|Title")),
        }
    return references, row_count


def _is_human(row: Mapping[str, str], human_taxon_iri: str) -> bool:
    return any(
        _clean(row.get(f"Chain {chain}|Organism IRI")).rstrip("/")
        == human_taxon_iri.rstrip("/")
        for chain in (1, 2)
    )


def _canonical_tcr(value: str, scope: Mapping[str, Any]) -> str:
    sequence = _clean(value).upper()
    if not valid_amino_acid_sequence(
        sequence, int(scope["tcr_min_length"]), int(scope["tcr_max_length"])
    ):
        return ""
    if scope["require_canonical_tcr"] and not (
        sequence.startswith("C") and sequence.endswith(("F", "W"))
    ):
        return ""
    return sequence


def _extract_trb(row: Mapping[str, str], scope: Mapping[str, Any]) -> tuple[str, str]:
    human_iri = str(scope["species_taxon_iri"]).rstrip("/")
    candidates: list[tuple[int, str, str]] = []
    for chain in (1, 2):
        chain_type = _clean(row.get(f"Chain {chain}|Type")).upper()
        organism = _clean(row.get(f"Chain {chain}|Organism IRI")).rstrip("/")
        if chain_type not in {"BETA", "TRB"} or organism != human_iri:
            continue
        curated = _canonical_tcr(row.get(f"Chain {chain}|CDR3 Curated", ""), scope)
        junction = _canonical_tcr(row.get(f"Chain {chain}|Junction Calculated", ""), scope)
        if curated:
            candidates.append((chain, curated, "CDR3 Curated"))
        elif junction:
            candidates.append((chain, junction, "Junction Calculated"))
    unique = {(sequence, source) for _, sequence, source in candidates}
    if len(unique) != 1:
        return "", ""
    return next(iter(unique))


def _valid_epitope(value: str, scope: Mapping[str, Any]) -> str:
    epitope = _clean(value).upper()
    if valid_amino_acid_sequence(
        epitope,
        int(scope["epitope_min_length"]),
        int(scope["epitope_max_length"]),
    ):
        return epitope
    return ""


def _is_mhc_class_i(value: str) -> bool:
    compact = normalize_hla(value)
    if not compact:
        return False
    if "CLASSII" in compact or re.search(r"HLA-(?:DR|DQ|DP)", compact):
        return False
    return compact in {"HLACLASSI", "MHCCLASSI", "MHCI"} or bool(
        re.match(r"^HLA-[ABCEFG](?:\*|$)", compact)
    )


def _standardized_hla(value: str) -> str:
    hla = normalize_hla(value)
    if any(delimiter in hla for delimiter in (",", ";", "|", "/")):
        return ""
    return hla if CLASSICAL_FULL_RESOLUTION_HLA.fullmatch(hla) else ""


def _filter_row(
    step: int, name: str, input_count: int, retained: int, raw_count: int
) -> dict[str, Any]:
    return {
        "step": step,
        "filter": name,
        "input_records": input_count,
        "retained_records": retained,
        "removed_records": input_count - retained,
        "retained_fraction_of_raw": retained / raw_count if raw_count else 0.0,
    }


def parse_iedb_receptors(
    receptor_zip: Path,
    receptor_member: str,
    references: Mapping[str, Mapping[str, Any]],
    scope: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    raw_count = 0
    stage = Counter()
    aggregates: dict[PairKey, dict[str, Any]] = {}
    mapping_counts = Counter()
    for row in _iter_export_rows(receptor_zip, receptor_member):
        raw_count += 1
        if not _is_human(row, str(scope["species_taxon_iri"])):
            continue
        stage["human"] += 1

        tcr, tcr_source = _extract_trb(row, scope)
        if not tcr:
            continue
        stage["trb"] += 1
        mapping_counts["tcr_cdr3_beta"] += 1

        epitope = _valid_epitope(row.get("Epitope|Name", ""), scope)
        if not epitope:
            continue
        stage["epitope"] += 1
        mapping_counts["epitope"] += 1

        mhc_raw = _clean(row.get("Assay|MHC Allele Names"))
        if not mhc_raw:
            continue
        stage["mhc"] += 1
        mapping_counts["hla_raw"] += 1
        if not _is_mhc_class_i(mhc_raw):
            continue
        stage["mhci"] += 1

        hla = _standardized_hla(mhc_raw)
        if not hla:
            continue
        stage["hla"] += 1
        mapping_counts["hla"] += 1

        key = (tcr, epitope, hla)
        reference_iri = _clean(row.get("Reference|IEDB IRI"))
        reference_number = _reference_number(reference_iri)
        reference = references.get(reference_number, {})
        aggregate = aggregates.setdefault(
            key,
            {
                "tcr_cdr3_beta": tcr,
                "epitope": epitope,
                "hla": hla,
                "pmhc_id": f"{epitope}|{hla}",
                "species": "Homo sapiens",
                "receptor_chain": "TRB",
                "assay_types": set(),
                "iedb_reference_ids": set(),
                "publication_tokens": set(),
                "antigen_sources": set(),
                "source_database": "IEDB receptor_full_v3",
                "receptor_ids": set(),
                "tcr_sequence_sources": set(),
                "source_record_count": 0,
            },
        )
        aggregate["assay_types"].add(_clean(row.get("Assay|Type")))
        if reference_number:
            aggregate["iedb_reference_ids"].add(f"IEDB_REF:{reference_number}")
        aggregate["publication_tokens"].update(reference.get("publication_tokens", set()))
        aggregate["antigen_sources"].update(
            value
            for value in (
                _clean(row.get("Epitope|Source Molecule")),
                _clean(row.get("Epitope|Source Organism")),
            )
            if value
        )
        aggregate["receptor_ids"].add(_clean(row.get("Receptor|IEDB Receptor ID")))
        aggregate["tcr_sequence_sources"].add(tcr_source)
        aggregate["source_record_count"] += 1
        mapping_counts["species"] += 1
        mapping_counts["receptor_chain"] += 1
        mapping_counts["assay_type"] += bool(_clean(row.get("Assay|Type")))
        mapping_counts["reference_id"] += bool(reference_number)
        mapping_counts["publication"] += bool(reference.get("publication_tokens", set()))
        mapping_counts["antigen_source"] += bool(aggregate["antigen_sources"])
        mapping_counts["source_database"] += 1

    records = []
    for key in sorted(aggregates, key=lambda value: (value[1], value[2], value[0])):
        aggregate = aggregates[key]
        records.append(
            {
                **aggregate,
                "assay_types": tuple(sorted(value for value in aggregate["assay_types"] if value)),
                "iedb_reference_ids": tuple(sorted(aggregate["iedb_reference_ids"])),
                "publication_tokens": tuple(sorted(aggregate["publication_tokens"])),
                "antigen_sources": tuple(sorted(aggregate["antigen_sources"])),
                "receptor_ids": tuple(sorted(value for value in aggregate["receptor_ids"] if value)),
                "tcr_sequence_sources": tuple(sorted(aggregate["tcr_sequence_sources"])),
            }
        )

    filters = [
        _filter_row(0, "raw receptor_full v3 records", raw_count, raw_count, raw_count),
        _filter_row(1, "human receptor organism", raw_count, stage["human"], raw_count),
        _filter_row(2, "legal canonical human TRB CDR3", stage["human"], stage["trb"], raw_count),
        _filter_row(3, "legal linear peptide epitope", stage["trb"], stage["epitope"], raw_count),
        _filter_row(4, "MHC annotation present", stage["epitope"], stage["mhc"], raw_count),
        _filter_row(5, "MHC class I annotation", stage["mhc"], stage["mhci"], raw_count),
        _filter_row(6, "classical full-resolution standardized HLA", stage["mhci"], stage["hla"], raw_count),
        _filter_row(7, "unique interaction after deduplication", stage["hla"], len(records), raw_count),
    ]
    return records, filters, dict(mapping_counts)


def load_vdjdb_registry(path: Path) -> dict[str, Any]:
    pairs: set[PairKey] = set()
    tcrs: set[str] = set()
    peptides: set[str] = set()
    queries: set[QueryKey] = set()
    hlas: set[str] = set()
    publications: set[str] = set()
    row_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"tcr_cdr3_beta", "epitope", "hla", "reference_id"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"VDJdb registry lacks required fields: {path}")
        for row in reader:
            row_count += 1
            key = (
                _clean(row["tcr_cdr3_beta"]).upper(),
                _clean(row["epitope"]).upper(),
                normalize_hla(row["hla"]),
            )
            pairs.add(key)
            tcrs.add(key[0])
            peptides.add(key[1])
            queries.add((key[1], key[2]))
            hlas.add(key[2])
            publications.update(_canonical_publication_tokens(row["reference_id"]))
    return {
        "row_count": row_count,
        "pairs": pairs,
        "tcrs": tcrs,
        "peptides": peptides,
        "queries": queries,
        "hlas": hlas,
        "publications": publications,
    }


def build_overlap_audit(
    records: Sequence[Mapping[str, Any]], vdjdb: Mapping[str, Any]
) -> list[dict[str, Any]]:
    pairs = {(row["tcr_cdr3_beta"], row["epitope"], row["hla"]) for row in records}
    tcrs = {row["tcr_cdr3_beta"] for row in records}
    peptides = {row["epitope"] for row in records}
    queries = {(row["epitope"], row["hla"]) for row in records}
    publications = {token for row in records for token in row["publication_tokens"]}
    records_with_publication = [row for row in records if row["publication_tokens"]]
    records_with_shared = [
        row
        for row in records_with_publication
        if set(row["publication_tokens"]) & vdjdb["publications"]
    ]
    values = [
        ("exact_interaction", "unique_interactions", len(pairs), len(pairs & vdjdb["pairs"])),
        ("tcr", "unique_tcrs", len(tcrs), len(tcrs & vdjdb["tcrs"])),
        ("peptide", "unique_peptides", len(peptides), len(peptides & vdjdb["peptides"])),
        ("pmhc", "unique_pmhc_queries", len(queries), len(queries & vdjdb["queries"])),
        (
            "publication_reference",
            "unique_publication_tokens",
            len(publications),
            len(publications & vdjdb["publications"]),
        ),
        (
            "records_with_shared_publication_reference",
            "records_with_mapped_publication",
            len(records_with_publication),
            len(records_with_shared),
        ),
        (
            "records_missing_publication_reference",
            "cleaned_records",
            len(records),
            len(records) - len(records_with_publication),
        ),
    ]
    return [
        {
            "metric": metric,
            "unit": unit,
            "iedb_total": total,
            "overlap_with_vdjdb": overlap,
            "overlap_fraction": overlap / total if total else 0.0,
        }
        for metric, unit, total, overlap in values
    ]


def _maximum_similarity(peptide: str, reference_peptides: Sequence[str]) -> float:
    return max(
        normalized_levenshtein_similarity(peptide, reference)
        for reference in reference_peptides
    )


def build_candidate_tiers(
    records: Sequence[Mapping[str, Any]],
    vdjdb_full: Mapping[str, Any],
    vdjdb_development: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    require_publication = bool(
        config["independence"]["require_mapped_publication_for_tier_a"]
    )
    tier_a = [
        dict(row)
        for row in records
        if (row["publication_tokens"] or not require_publication)
        and not (set(row["publication_tokens"]) & vdjdb_full["publications"])
        and (row["tcr_cdr3_beta"], row["epitope"], row["hla"])
        not in vdjdb_full["pairs"]
    ]
    tier_b = [
        row for row in tier_a if row["epitope"] not in vdjdb_development["peptides"]
    ]
    threshold = float(
        config["independence"]["maximum_development_peptide_similarity"]
    )
    development_peptides = sorted(vdjdb_development["peptides"])
    peptide_similarities = {
        peptide: _maximum_similarity(peptide, development_peptides)
        for peptide in sorted({row["epitope"] for row in tier_b})
    }
    tier_c = [
        row for row in tier_b if peptide_similarities[row["epitope"]] <= threshold
    ]
    return {"A": tier_a, "B": tier_b, "C": tier_c}, peptide_similarities


def _source_category(row: Mapping[str, Any], categories: Mapping[str, Sequence[str]]) -> str:
    source = " ".join(row["antigen_sources"]).lower()
    for category in (
        "tumor_associated",
        "self_antigen",
        "pathogen_or_nonself",
        "other_nonself",
    ):
        if any(keyword.lower() in source for keyword in categories[category]):
            return category
    return "unknown_source"


def summarize_candidate_tiers(
    tiers: Mapping[str, Sequence[Mapping[str, Any]]],
    vdjdb_full: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    definitions = {
        "A": "publication-independent",
        "B": "exact-unseen",
        "C": "distance-aware strict",
    }
    rows = []
    for tier in ("A", "B", "C"):
        records = tiers[tier]
        queries = Counter((row["epitope"], row["hla"]) for row in records)
        sources = Counter(
            _source_category(row, config["antigen_source_categories"])
            for row in records
        )
        rows.append(
            {
                "tier": tier,
                "definition": definitions[tier],
                "positive_records": len(records),
                "unique_tcrs": len({row["tcr_cdr3_beta"] for row in records}),
                "unique_peptides": len({row["epitope"] for row in records}),
                "unique_pmhc_queries": len(queries),
                "unique_hlas": len({row["hla"] for row in records}),
                "unique_publications": len(
                    {token for row in records for token in row["publication_tokens"]}
                ),
                **{
                    f"queries_min_{minimum}_positive{'s' if minimum != 1 else ''}": sum(
                        count >= minimum for count in queries.values()
                    )
                    for minimum in (1, 5, 10, 20)
                },
                "hla_seen_queries": sum(query[1] in vdjdb_full["hlas"] for query in queries),
                "hla_unseen_queries": sum(query[1] not in vdjdb_full["hlas"] for query in queries),
                "pathogen_or_nonself_records": sources["pathogen_or_nonself"],
                "tumor_associated_records": sources["tumor_associated"],
                "self_antigen_records": sources["self_antigen"],
                "other_nonself_records": sources["other_nonself"],
                "unknown_source_records": sources["unknown_source"],
            }
        )
    return rows


def _select_capped_records(
    records: Sequence[Mapping[str, Any]], cap: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[QueryKey, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["epitope"], row["hla"])].append(row)
    selected = []
    for query, values in sorted(grouped.items()):
        ordered = sorted(
            values,
            key=lambda row: (
                _stable_hash(
                    seed,
                    str(cap),
                    query[0],
                    query[1],
                    row["tcr_cdr3_beta"],
                ),
                row["tcr_cdr3_beta"],
            ),
        )
        selected.extend(dict(row) for row in ordered[:cap])
    selected.sort(key=lambda row: (row["epitope"], row["hla"], row["tcr_cdr3_beta"]))
    return selected


def simulate_degree_matching(
    strict_records: Sequence[Mapping[str, Any]],
    positive_registry: set[PairKey],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    seed = int(config["degree_matching"]["seed"])
    rows = []
    for cap in config["degree_matching"]["query_caps"]:
        positives = _select_capped_records(strict_records, int(cap), seed)
        solver_rows = [
            {
                "tcr_cdr3_beta": row["tcr_cdr3_beta"],
                "epitope": row["epitope"],
                "hla": row["hla"],
                "label": 1,
            }
            for row in positives
        ]
        try:
            _, audit = solve_degree_matched_partition(
                solver_rows,
                positive_registry,
                fold_id="fold_1",
                partition="test",
                seed=seed + int(cap),
            )
            row = {
                "query_cap": int(cap),
                "retained_queries": audit["unique_query_count"],
                "retained_positive_records": audit["positive_count"],
                "exact_solution": True,
                "maximum_matched_edges": audit["maximum_flow_value"],
                "matching_rate": audit["matching_rate"],
                "failed_tcr_count": 0,
                "failed_query_count": 0,
                "failure_reason": "",
            }
        except ExactDegreeMatchError as error:
            report = error.report
            row = {
                "query_cap": int(cap),
                "retained_queries": len(
                    {(record["epitope"], record["hla"]) for record in positives}
                ),
                "retained_positive_records": len(positives),
                "exact_solution": False,
                "maximum_matched_edges": report["maximum_matched_edge_count"],
                "matching_rate": report["maximum_matching_ratio"],
                "failed_tcr_count": len(report["failed_tcrs"]),
                "failed_query_count": len(report["failed_queries"]),
                "failure_reason": "bipartite capacity/minimum-cut constraint",
            }
        rows.append(row)
    return rows


def evaluate_decision(
    tier_rows: Sequence[Mapping[str, Any]],
    degree_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[str, list[str]]:
    tiers = {row["tier"]: row for row in tier_rows}
    cap20 = next(row for row in degree_rows if int(row["query_cap"]) == 20)
    pass_rule = config["decision_rule"]["pass"]
    pass_checks = {
        "strict positive records": tiers["C"]["positive_records"]
        >= int(pass_rule["strict_positive_records_at_least"]),
        "strict queries with >=10 positives": tiers["C"]["queries_min_10_positives"]
        >= int(pass_rule["strict_queries_with_at_least_10_positives"]),
        "strict publications": tiers["C"]["unique_publications"]
        >= int(pass_rule["strict_publications_at_least"]),
        "cap=20 exact degree matching": bool(cap20["exact_solution"])
        is bool(pass_rule["cap20_exact_degree_match"]),
        "cap=20 queries": cap20["retained_queries"]
        >= int(pass_rule["cap20_queries_at_least"]),
        "cap=20 positive records": cap20["retained_positive_records"]
        >= int(pass_rule["cap20_positive_records_at_least"]),
    }
    if all(pass_checks.values()):
        return "PASS", [name for name, passed in pass_checks.items() if passed]

    conditional = config["decision_rule"]["conditional"]
    conditional_checks = {
        "exact-unseen positive records": tiers["B"]["positive_records"]
        >= int(conditional["exact_unseen_positive_records_at_least"]),
        "exact-unseen queries with >=10 positives": tiers["B"]["queries_min_10_positives"]
        >= int(conditional["exact_unseen_queries_with_at_least_10_positives"]),
        "exact-unseen publications": tiers["B"]["unique_publications"]
        >= int(conditional["exact_unseen_publications_at_least"]),
    }
    if all(conditional_checks.values()):
        return "CONDITIONAL", [
            f"failed PASS check: {name}" for name, passed in pass_checks.items() if not passed
        ]
    return "FAIL", [
        f"failed PASS check: {name}" for name, passed in pass_checks.items() if not passed
    ] + [
        f"failed CONDITIONAL check: {name}"
        for name, passed in conditional_checks.items()
        if not passed
    ]


def build_schema_mapping(
    mapping_counts: Mapping[str, int],
    filter_rows: Sequence[Mapping[str, Any]],
    reference_row_count: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "tcr_cdr3_beta": {
            "sources": [
                "Chain 1/2|CDR3 Curated",
                "Chain 1/2|Junction Calculated",
            ],
            "transform": "prefer legal canonical curated CDR3, otherwise legal canonical calculated junction",
        },
        "epitope": {
            "sources": ["Epitope|Name"],
            "transform": "uppercase standard amino-acid sequence, length 8-15",
        },
        "hla": {
            "sources": ["Assay|MHC Allele Names"],
            "transform": "tcrue.vdjdb.normalize_hla plus CLASSICAL_FULL_RESOLUTION_HLA",
        },
        "species": {
            "sources": ["Chain 1/2|Organism IRI"],
            "transform": "NCBITaxon_9606 to Homo sapiens",
        },
        "receptor_chain": {
            "sources": ["Chain 1/2|Type"],
            "transform": "BETA/TRB to TRB",
        },
        "assay_type": {"sources": ["Assay|Type"], "transform": "trim and aggregate"},
        "reference_id / PMID / DOI": {
            "sources": [
                "Reference|IEDB IRI",
                "reference_full_v3 Reference|PMID",
                "reference_full_v3 Reference|Alternate IRIs",
            ],
            "transform": "join IEDB reference ID; canonicalize PMID and DOI publication tokens",
            "limitation": "DOI is available only when represented in Alternate IRIs",
        },
        "antigen_source": {
            "sources": ["Epitope|Source Molecule", "Epitope|Source Organism"],
            "transform": "aggregate and classify with predeclared keyword categories",
        },
        "source_database": {
            "sources": [],
            "transform": "constant IEDB receptor_full_v3",
        },
    }
    return {
        "schema_version": 1,
        "export": config["source"]["export_name"],
        "receptor_export_columns": 74,
        "reference_export_columns": 9,
        "reference_export_records": reference_row_count,
        "fields": fields,
        "mapped_record_counts_before_deduplication": dict(sorted(mapping_counts.items())),
        "deduplicated_interaction_count": filter_rows[-1]["retained_records"],
        "normalization_reuse": {
            "module": config["implementation"]["hla_and_sequence_normalization"],
            "functions": ["normalize_hla", "valid_amino_acid_sequence"],
            "hla_pattern": "CLASSICAL_FULL_RESOLUTION_HLA",
        },
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_count(value: Any) -> str:
    return f"{int(value):,}"


def render_report(
    config: Mapping[str, Any],
    filter_rows: Sequence[Mapping[str, Any]],
    overlap_rows: Sequence[Mapping[str, Any]],
    tier_rows: Sequence[Mapping[str, Any]],
    degree_rows: Sequence[Mapping[str, Any]],
    decision: str,
    decision_notes: Sequence[str],
) -> str:
    lines = [
        "# IEDB receptor_full v3 external-data feasibility audit",
        "",
        "## Scope and provenance",
        "",
        f"- Export snapshot/update date: `{config['source']['export_update_date']}`",
        f"- Download date: `{config['source']['download_date']}`",
        f"- Export: `{config['source']['export_name']}` (official IEDB weekly export)",
        f"- License: `{config['source']['license']}`",
        f"- Citation: {config['source']['citation']['text']} DOI `{config['source']['citation']['doi']}`.",
        "- This audit does not train, tune, score, or save a formal external benchmark.",
        "",
        "## Sequential filtering",
        "",
        "| Step | Filter | Input | Retained | Removed |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in filter_rows:
        lines.append(
            f"| {row['step']} | {row['filter']} | {_format_count(row['input_records'])} | "
            f"{_format_count(row['retained_records'])} | {_format_count(row['removed_records'])} |"
        )
    lines.extend(
        [
            "",
            "The linear-peptide operational definition is an unmodified standard amino-acid string of length 8-15. HLA standardization reuses the frozen VDJdb normalization and retains unambiguous classical full-resolution HLA-A/B/C alleles.",
            "",
            "## Overlap with frozen VDJdb",
            "",
            "| Metric | IEDB denominator | Overlap/missing | Fraction |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in overlap_rows:
        lines.append(
            f"| {row['metric']} | {_format_count(row['iedb_total'])} | "
            f"{_format_count(row['overlap_with_vdjdb'])} | {float(row['overlap_fraction']):.6f} |"
        )
    lines.extend(
        [
            "",
            "Reference overlap is computed from canonical PMID/DOI tokens. Tier A conservatively excludes records without a mapped publication, so publication independence is auditable rather than assumed.",
            "",
            "## Candidate audit tiers",
            "",
            "| Tier | Definition | Records | TCRs | Peptides | Queries | HLAs | Publications | >=5 | >=10 | >=20 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tier_rows:
        lines.append(
            f"| {row['tier']} | {row['definition']} | {_format_count(row['positive_records'])} | "
            f"{_format_count(row['unique_tcrs'])} | {_format_count(row['unique_peptides'])} | "
            f"{_format_count(row['unique_pmhc_queries'])} | {_format_count(row['unique_hlas'])} | "
            f"{_format_count(row['unique_publications'])} | {_format_count(row['queries_min_5_positives'])} | "
            f"{_format_count(row['queries_min_10_positives'])} | {_format_count(row['queries_min_20_positives'])} |"
        )
    lines.extend(
        [
            "",
            "Antigen-source categories are a descriptive keyword audit over source molecule and source organism annotations; they are not curated disease labels.",
            "",
            "## Strict-tier degree-matching feasibility",
            "",
            "| Query cap | Queries | Positives | Exact solution | Matching rate | Failure reason |",
            "|---:|---:|---:|---|---:|---|",
        ]
    )
    for row in degree_rows:
        lines.append(
            f"| {row['query_cap']} | {_format_count(row['retained_queries'])} | "
            f"{_format_count(row['retained_positive_records'])} | {row['exact_solution']} | "
            f"{float(row['matching_rate']):.6f} | {row['failure_reason']} |"
        )
    lines.extend(
        [
            "",
            "Only feasibility was solved. No putative-negative edge was persisted.",
            "",
            "## Predeclared decision",
            "",
            f"Decision code: `{decision}`.",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in decision_notes)
    lines.append("")
    final_lines = {
        "PASS": "PASS\uff1a\u9002\u5408\u6784\u5efa\u4e25\u683c\u72ec\u7acb\u5916\u90e8\u6d4b\u8bd5\u96c6",
        "CONDITIONAL": "CONDITIONAL\uff1a\u53ef\u7528\uff0c\u4f46\u9700\u653e\u5bbd\u9884\u5148\u6307\u5b9a\u7684\u67d0\u9879\u6761\u4ef6",
        "FAIL": "FAIL\uff1a\u4e25\u683c\u72ec\u7acb\u6570\u636e\u4e0d\u8db3",
    }
    lines.append(final_lines[decision])
    return "\n".join(lines) + "\n"


def _verify_pinned_inputs(project_dir: Path, config: Mapping[str, Any]) -> None:
    for specification in config["source"]["archives"].values():
        path = project_dir / specification["path"]
        if path.stat().st_size != int(specification["size_bytes"]):
            raise ValueError(f"pinned IEDB archive size mismatch: {path}")
        if file_sha256(path) != specification["sha256"]:
            raise ValueError(f"pinned IEDB archive hash mismatch: {path}")
    for key, path_key, hash_key in (
        ("positive_registry", "positive_registry", "positive_registry_sha256"),
        ("development_registry", "development_registry", "development_registry_sha256"),
        ("data_tiers_manifest", "data_tiers_manifest", "data_tiers_manifest_sha256"),
    ):
        section = config["frozen_vdjdb"]
        if file_sha256(project_dir / section[path_key]) != section[hash_key]:
            raise ValueError(f"frozen VDJdb hash mismatch: {key}")
    for path_key, hash_key in (
        ("hla_and_sequence_normalization", "hla_and_sequence_normalization_sha256"),
        ("peptide_similarity", "peptide_similarity_sha256"),
        ("degree_matching", "degree_matching_sha256"),
    ):
        section = config["implementation"]
        if file_sha256(project_dir / section[path_key]) != section[hash_key]:
            raise ValueError(f"pinned implementation hash mismatch: {section[path_key]}")


def _run_once(
    project_dir: Path, config: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    receptor_spec = config["source"]["archives"]["receptor"]
    reference_spec = config["source"]["archives"]["reference"]
    references, reference_count = load_iedb_references(
        project_dir / reference_spec["path"], reference_spec["member"]
    )
    records, filters, mapping_counts = parse_iedb_receptors(
        project_dir / receptor_spec["path"],
        receptor_spec["member"],
        references,
        config["scope"],
    )
    full = load_vdjdb_registry(project_dir / config["frozen_vdjdb"]["positive_registry"])
    development = load_vdjdb_registry(
        project_dir / config["frozen_vdjdb"]["development_registry"]
    )
    overlaps = build_overlap_audit(records, full)
    tiers, similarities = build_candidate_tiers(records, full, development, config)
    tier_summary = summarize_candidate_tiers(tiers, full, config)
    complete_registry = set(full["pairs"]) | {
        (row["tcr_cdr3_beta"], row["epitope"], row["hla"]) for row in records
    }
    degree_rows = simulate_degree_matching(tiers["C"], complete_registry, config)
    decision, decision_notes = evaluate_decision(tier_summary, degree_rows, config)
    schema = build_schema_mapping(
        mapping_counts, filters, reference_count, config
    )
    schema["strict_tier_maximum_development_peptide_similarity"] = max(
        (similarities[row["epitope"]] for row in tiers["C"]), default=0.0
    )

    names = config["outputs"]
    _write_json(output_dir / names["schema_mapping"], schema)
    _write_csv(output_dir / names["filter_counts"], FILTER_FIELDNAMES, filters)
    _write_csv(output_dir / names["overlap_audit"], OVERLAP_FIELDNAMES, overlaps)
    _write_csv(output_dir / names["candidate_tier_summary"], TIER_FIELDNAMES, tier_summary)
    _write_csv(
        output_dir / names["degree_matching_feasibility"],
        DEGREE_FIELDNAMES,
        degree_rows,
    )
    (output_dir / names["report"]).write_text(
        render_report(
            config,
            filters,
            overlaps,
            tier_summary,
            degree_rows,
            decision,
            decision_notes,
        ),
        encoding="utf-8",
    )
    return {
        "raw_record_count": filters[0]["retained_records"],
        "cleaned_record_count": filters[-1]["retained_records"],
        "filter_counts": filters,
        "overlap_audit": overlaps,
        "tier_summary": tier_summary,
        "degree_matching": degree_rows,
        "decision": decision,
        "decision_notes": decision_notes,
        "strict_max_similarity": schema[
            "strict_tier_maximum_development_peptide_similarity"
        ],
    }


def run_iedb_feasibility_audit(
    project_dir: Path,
    config_path: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _verify_pinned_inputs(project_dir, config)
    final_dir = output_dir or project_dir / config["outputs"]["directory"]
    expected_outputs = {
        config["outputs"][key]
        for key in (
            "schema_mapping",
            "filter_counts",
            "overlap_audit",
            "candidate_tier_summary",
            "degree_matching_feasibility",
            "manifest",
            "report",
        )
    }
    if final_dir.exists():
        unexpected = sorted(
            path.name for path in final_dir.iterdir() if path.name not in expected_outputs
        )
        if unexpected:
            raise ValueError(
                "refusing to write audit directory containing unexpected files: "
                + ", ".join(unexpected)
            )

    with tempfile.TemporaryDirectory(prefix="iedb-audit-rebuild-1-") as first_temp, tempfile.TemporaryDirectory(
        prefix="iedb-audit-rebuild-2-"
    ) as second_temp:
        first_dir = Path(first_temp)
        second_dir = Path(second_temp)
        result = _run_once(project_dir, config, first_dir)
        second_result = _run_once(project_dir, config, second_dir)
        base_names = [
            config["outputs"][key]
            for key in (
                "schema_mapping",
                "filter_counts",
                "overlap_audit",
                "candidate_tier_summary",
                "degree_matching_feasibility",
                "report",
            )
        ]
        first_hashes = {name: file_sha256(first_dir / name) for name in base_names}
        second_hashes = {name: file_sha256(second_dir / name) for name in base_names}
        if first_hashes != second_hashes or result != second_result:
            raise ValueError("IEDB feasibility audit is not reproducible across two rebuilds")
        final_dir.mkdir(parents=True, exist_ok=True)
        for name in base_names:
            shutil.copyfile(first_dir / name, final_dir / name)

    output_entries = {
        name: {
            "path": str((final_dir / name).relative_to(project_dir)),
            "sha256": file_sha256(final_dir / name),
        }
        for name in base_names
    }
    download_manifest_path = project_dir / config["source"]["download_manifest"]
    if not download_manifest_path.is_file():
        raise FileNotFoundError(
            "run scripts/download_iedb_receptors.py before the feasibility audit"
        )
    manifest = {
        "schema_version": 1,
        "frozen_base_commit": config["frozen_base_commit"],
        "audit_pipeline": {
            "config": {
                "path": str(config_path.relative_to(project_dir)),
                "sha256": file_sha256(config_path),
            },
            "download_script": {
                "path": "scripts/download_iedb_receptors.py",
                "sha256": file_sha256(
                    project_dir / "scripts/download_iedb_receptors.py"
                ),
            },
            "audit_entrypoint": {
                "path": "scripts/audit_iedb_external.py",
                "sha256": file_sha256(project_dir / "scripts/audit_iedb_external.py"),
            },
            "audit_module": {
                "path": "src/tcrue/iedb_external_audit.py",
                "sha256": file_sha256(
                    project_dir / "src/tcrue/iedb_external_audit.py"
                ),
            },
        },
        "source": {
            "database": config["source"]["database"],
            "export_name": config["source"]["export_name"],
            "export_update_date": str(config["source"]["export_update_date"]),
            "download_date": str(config["source"]["download_date"]),
            "download_manifest": {
                "path": config["source"]["download_manifest"],
                "sha256": file_sha256(download_manifest_path),
            },
            "archives": config["source"]["archives"],
        },
        "frozen_vdjdb": config["frozen_vdjdb"],
        "implementation": config["implementation"],
        "predeclared_decision_rule": config["decision_rule"],
        "outputs": output_entries,
        "reproducibility": {
            "complete_rebuilds_compared": 2,
            "all_output_sha256_identical": True,
            "sha256_by_file": first_hashes,
        },
        "audit": {
            "formal_external_dataset_generated": False,
            "negative_samples_persisted": False,
            "training_performed": False,
            "tuning_performed": False,
            "model_scoring_performed": False,
            "publication_independent_shared_reference_count": 0,
            "exact_unseen_shared_development_peptide_count": 0,
            "strict_maximum_development_peptide_similarity": result[
                "strict_max_similarity"
            ],
            "decision": result["decision"],
        },
        "statistics": result,
    }
    manifest_path = final_dir / config["outputs"]["manifest"]
    _write_json(manifest_path, manifest)
    result["manifest_path"] = str(manifest_path.relative_to(project_dir))
    result["reproducibility_passed"] = True
    return result
