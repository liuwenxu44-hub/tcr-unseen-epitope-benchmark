from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .vocab import PAD_ID, TOKENS


DEFAULT_AA_EMBEDDING_DIM = 64
DEFAULT_CNN_CHANNELS = 128
DEFAULT_HLA_EMBEDDING_DIM = 32
DEFAULT_KERNEL_SIZES = (3, 5)
DEFAULT_MLP_HIDDEN_DIMS = (256, 64)


class MaskedCnnEncoder(nn.Module):
    """Encode an amino-acid sequence without pooling over PAD positions."""

    def __init__(
        self,
        *,
        vocab_size: int = len(TOKENS),
        embedding_dim: int = DEFAULT_AA_EMBEDDING_DIM,
        channels: int = DEFAULT_CNN_CHANNELS,
        kernel_sizes: Sequence[int] = DEFAULT_KERNEL_SIZES,
        padding_idx: int = PAD_ID,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not kernel_sizes or any(size <= 0 or size % 2 == 0 for size in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive odd integers")

        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                embedding_dim,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = channels * len(kernel_sizes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence_length]")
        if token_ids.shape[1] == 0:
            raise ValueError("sequence_length must be positive")

        valid_positions = token_ids.ne(self.padding_idx)
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooling_mask = valid_positions.unsqueeze(1)
        all_padding = ~valid_positions.any(dim=1, keepdim=True)

        pooled_features = []
        for convolution in self.convolutions:
            features = torch.relu(convolution(embedded))
            features = features.masked_fill(~pooling_mask, -torch.inf)
            pooled = features.max(dim=-1).values
            pooled = torch.where(all_padding, torch.zeros_like(pooled), pooled)
            pooled_features.append(pooled)
        return self.dropout(torch.cat(pooled_features, dim=-1))


class TwoTowerCnn(nn.Module):
    """Two-tower CNN scorer that returns one raw interaction logit per sample."""

    def __init__(
        self,
        hla_vocab_size: int,
        *,
        aa_vocab_size: int = len(TOKENS),
        aa_embedding_dim: int = DEFAULT_AA_EMBEDDING_DIM,
        cnn_channels: int = DEFAULT_CNN_CHANNELS,
        hla_embedding_dim: int = DEFAULT_HLA_EMBEDDING_DIM,
        kernel_sizes: Sequence[int] = DEFAULT_KERNEL_SIZES,
        mlp_hidden_dims: Sequence[int] = DEFAULT_MLP_HIDDEN_DIMS,
        padding_idx: int = PAD_ID,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hla_vocab_size <= 0:
            raise ValueError("hla_vocab_size must be positive")
        if not mlp_hidden_dims or any(size <= 0 for size in mlp_hidden_dims):
            raise ValueError("mlp_hidden_dims must contain positive integers")

        encoder_options = {
            "vocab_size": aa_vocab_size,
            "embedding_dim": aa_embedding_dim,
            "channels": cnn_channels,
            "kernel_sizes": kernel_sizes,
            "padding_idx": padding_idx,
            "dropout": dropout,
        }
        self.tcr_encoder = MaskedCnnEncoder(**encoder_options)
        self.epitope_encoder = MaskedCnnEncoder(**encoder_options)
        self.hla_embedding = nn.Embedding(hla_vocab_size, hla_embedding_dim)

        combined_dim = (
            self.tcr_encoder.output_dim
            + self.epitope_encoder.output_dim
            + hla_embedding_dim
        )
        layers: list[nn.Module] = []
        input_dim = combined_dim
        for hidden_dim in mlp_hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        tcr_ids: torch.Tensor,
        epitope_ids: torch.Tensor,
        hla_id: torch.Tensor,
    ) -> torch.Tensor:
        if hla_id.ndim != 1:
            raise ValueError("hla_id must have shape [batch]")
        batch_size = tcr_ids.shape[0]
        if epitope_ids.shape[0] != batch_size or hla_id.shape[0] != batch_size:
            raise ValueError("all model inputs must have the same batch size")

        tcr_vector = self.tcr_encoder(tcr_ids)
        epitope_vector = self.epitope_encoder(epitope_ids)
        hla_vector = self.hla_embedding(hla_id)
        combined = torch.cat((tcr_vector, epitope_vector, hla_vector), dim=-1)
        return self.mlp(combined).squeeze(-1)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
