from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1
STANDARD_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
TOKENS = (PAD_TOKEN, UNK_TOKEN, *STANDARD_AMINO_ACIDS)


@dataclass(frozen=True)
class EncodedSequence:
    token_ids: tuple[int, ...]
    original_length: int
    effective_length: int


class AminoAcidVocab:
    """Fixed vocabulary for standard amino-acid sequences."""

    def __init__(self) -> None:
        self.token_to_id = {token: index for index, token in enumerate(TOKENS)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    def __len__(self) -> int:
        return len(TOKENS)

    def encode(self, sequence: str, max_length: int) -> EncodedSequence:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        normalized = sequence.strip().upper()
        original_length = len(normalized)
        effective_length = min(original_length, max_length)
        token_ids = [
            self.token_to_id.get(amino_acid, UNK_ID)
            for amino_acid in normalized[:max_length]
        ]
        token_ids.extend([PAD_ID] * (max_length - len(token_ids)))
        return EncodedSequence(
            token_ids=tuple(token_ids),
            original_length=original_length,
            effective_length=effective_length,
        )

    def decode(self, token_ids: Iterable[int], *, skip_pad: bool = True) -> str:
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), UNK_TOKEN)
            if skip_pad and token == PAD_TOKEN:
                continue
            tokens.append(token)
        return "".join(tokens)
