from __future__ import annotations

import torch
from torch import nn

from pytorch_vit.multihead_self_attention import MultiheadSelfAttention


class TransformerEncoder(nn.Module):
    def __init__(self, num_patches: int, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        normalized_shape: tuple[int, int] = (1 + num_patches, embedding_dim)
        self.layer_norm: nn.Module = nn.LayerNorm(normalized_shape)
        self.msa: MultiheadSelfAttention = MultiheadSelfAttention(num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x)
