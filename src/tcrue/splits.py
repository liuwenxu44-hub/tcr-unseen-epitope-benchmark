from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def random_split(
    frame: pd.DataFrame,
    valid_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_valid, test = train_test_split(frame, test_size=test_size, random_state=seed, stratify=frame["label"])
    train, valid = train_test_split(
        train_valid,
        test_size=valid_size / (1 - test_size),
        random_state=seed,
        stratify=train_valid["label"],
    )
    return train, valid, test


def unseen_epitope_split(
    frame: pd.DataFrame,
    epitope_col: str = "epitope",
    valid_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    epitopes = np.array(sorted(frame[epitope_col].dropna().unique()))
    rng.shuffle(epitopes)
    n_test = max(1, int(round(len(epitopes) * test_size)))
    n_valid = max(1, int(round(len(epitopes) * valid_size)))
    test_epitopes = set(epitopes[:n_test])
    valid_epitopes = set(epitopes[n_test:n_test + n_valid])
    test = frame[frame[epitope_col].isin(test_epitopes)]
    valid = frame[frame[epitope_col].isin(valid_epitopes)]
    train = frame[~frame[epitope_col].isin(test_epitopes | valid_epitopes)]
    return train, valid, test
