from __future__ import annotations

import torch
from torch import nn


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.num_heads: int = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
