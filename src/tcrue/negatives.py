from __future__ import annotations

import random

import pandas as pd


def make_random_negatives(
    positives: pd.DataFrame,
    ratio: int,
    tcr_col: str = "tcr_cdr3_beta",
    epitope_col: str = "epitope",
    hla_col: str = "hla",
    seed: int = 13,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = positives[[tcr_col, epitope_col, hla_col]].drop_duplicates().to_dict("records")
    epitopes = positives[epitope_col].dropna().unique().tolist()
    hla_by_epitope = positives.drop_duplicates(epitope_col).set_index(epitope_col)[hla_col].to_dict()
    positive_pairs = {(row[tcr_col], row[epitope_col], row[hla_col]) for row in rows}
    negatives = []

    for row in rows:
        made = 0
        attempts = 0
        while made < ratio and attempts < ratio * 50:
            attempts += 1
            epitope = rng.choice(epitopes)
            hla = hla_by_epitope.get(epitope, row[hla_col])
            candidate = (row[tcr_col], epitope, hla)
            if candidate in positive_pairs:
                continue
            negatives.append({tcr_col: candidate[0], epitope_col: candidate[1], hla_col: candidate[2], "label": 0})
            made += 1
    return pd.DataFrame(negatives)


def make_hla_matched_negatives(
    positives: pd.DataFrame,
    ratio: int,
    tcr_col: str = "tcr_cdr3_beta",
    epitope_col: str = "epitope",
    hla_col: str = "hla",
    seed: int = 13,
) -> pd.DataFrame:
    rng = random.Random(seed)
    base = positives[[tcr_col, epitope_col, hla_col]].drop_duplicates()
    positive_pairs = {(r[tcr_col], r[epitope_col], r[hla_col]) for r in base.to_dict("records")}
    epitopes_by_hla = {
        hla: group[epitope_col].dropna().unique().tolist()
        for hla, group in base.groupby(hla_col)
    }
    negatives = []

    for row in base.to_dict("records"):
        choices = epitopes_by_hla.get(row[hla_col], [])
        if len(choices) < 2:
            continue
        made = 0
        attempts = 0
        while made < ratio and attempts < ratio * 50:
            attempts += 1
            epitope = rng.choice(choices)
            candidate = (row[tcr_col], epitope, row[hla_col])
            if candidate in positive_pairs:
                continue
            negatives.append({tcr_col: candidate[0], epitope_col: candidate[1], hla_col: candidate[2], "label": 0})
            made += 1
    return pd.DataFrame(negatives)
