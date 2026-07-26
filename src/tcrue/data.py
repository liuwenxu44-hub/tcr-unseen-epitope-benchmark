from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset

from .vocab import AminoAcidVocab


@dataclass(frozen=True)
class DatasetColumns:
    tcr: str = "tcr_cdr3_beta"
    epitope: str = "epitope"
    hla: str = "hla"
    label: str = "label"


class TcrEpitopeDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        aa_vocab: AminoAcidVocab,
        hla_vocab: dict[str, int],
        columns: DatasetColumns,
        tcr_max_len: int,
        epitope_max_len: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.aa_vocab = aa_vocab
        self.hla_vocab = hla_vocab
        self.columns = columns
        self.tcr_max_len = tcr_max_len
        self.epitope_max_len = epitope_max_len

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        tcr = self.aa_vocab.encode(str(row[self.columns.tcr]), self.tcr_max_len)
        epitope = self.aa_vocab.encode(str(row[self.columns.epitope]), self.epitope_max_len)
        hla = self.hla_vocab.get(str(row[self.columns.hla]), 0)
        label = float(row[self.columns.label])
        return {
            "tcr": torch.tensor(tcr, dtype=torch.long),
            "epitope": torch.tensor(epitope, dtype=torch.long),
            "hla": torch.tensor(hla, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
