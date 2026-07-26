from __future__ import annotations

import torch
from torch import nn


class SequenceCnnEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        channels: int,
        kernel_sizes: list[int],
        dropout: float,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, channels, kernel_size=kernel_size, padding=kernel_size // 2)
            for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = channels * len(kernel_sizes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens).transpose(1, 2)
        features = []
        for conv in self.convs:
            y = torch.relu(conv(x))
            y = torch.max(y, dim=-1).values
            features.append(y)
        return self.dropout(torch.cat(features, dim=-1))


class TwoTowerCnnScorer(nn.Module):
    def __init__(
        self,
        aa_vocab_size: int,
        hla_vocab_size: int,
        aa_embedding_dim: int,
        hla_embedding_dim: int,
        cnn_channels: int,
        kernel_sizes: list[int],
        mlp_hidden_dims: list[int],
        dropout: float,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.tcr_encoder = SequenceCnnEncoder(
            aa_vocab_size, aa_embedding_dim, cnn_channels, kernel_sizes, dropout, padding_idx
        )
        self.epitope_encoder = SequenceCnnEncoder(
            aa_vocab_size, aa_embedding_dim, cnn_channels, kernel_sizes, dropout, padding_idx
        )
        self.hla_embedding = nn.Embedding(max(hla_vocab_size, 1), hla_embedding_dim)

        pair_dim = self.tcr_encoder.output_dim + self.epitope_encoder.output_dim + hla_embedding_dim
        layers: list[nn.Module] = []
        in_dim = pair_dim
        for hidden_dim in mlp_hidden_dims:
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.scorer = nn.Sequential(*layers)

    def forward(self, tcr: torch.Tensor, epitope: torch.Tensor, hla: torch.Tensor) -> torch.Tensor:
        tcr_vec = self.tcr_encoder(tcr)
        epitope_vec = self.epitope_encoder(epitope)
        hla_vec = self.hla_embedding(hla)
        pair_vec = torch.cat([tcr_vec, epitope_vec, hla_vec], dim=-1)
        return self.scorer(pair_vec).squeeze(-1)
