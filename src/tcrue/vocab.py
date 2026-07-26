from __future__ import annotations

AA_TOKENS = list("ACDEFGHIKLMNPQRSTVWY")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class AminoAcidVocab:
    def __init__(self) -> None:
        self.tokens = [PAD_TOKEN, UNK_TOKEN] + AA_TOKENS
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, sequence: str, max_len: int) -> list[int]:
        sequence = (sequence or "").strip().upper()
        ids = [self.token_to_id.get(char, self.unk_id) for char in sequence[:max_len]]
        if len(ids) < max_len:
            ids.extend([self.pad_id] * (max_len - len(ids)))
        return ids


def build_category_vocab(values: list[str]) -> dict[str, int]:
    clean_values = sorted({str(value) for value in values if str(value) and str(value) != "nan"})
    return {value: idx for idx, value in enumerate(clean_values)}
