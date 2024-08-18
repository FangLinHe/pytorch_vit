from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PatchEmbedder(nn.Module):
    def __init__(
        self,
        patch_length: int,
        embedding_dim: int,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.patch_length: int = patch_length
        self.embedding_dim: int = embedding_dim
        self.class_token: nn.Parameter = nn.Parameter(
            torch.empty(
                (
                    1,
                    self.embedding_dim,
                ),
                **factory_kwargs,
            ),
            requires_grad=True,
        )
        self.linear_layer: nn.Module = nn.Linear(self.patch_length, self.embedding_dim, bias=False, **factory_kwargs)

        nn.init.uniform_(self.class_token, a=-1.0, b=1.0)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        num_batches, num_patches, patch_length = patches.shape
        # BxNxP --> (B*N)xP
        reshaped_patches: torch.Tensor = patches.view(-1, patch_length)
        # (B*N)xD --> BxNxD
        projected_embeddings: torch.Tensor = self.linear_layer(reshaped_patches).view(num_batches, num_patches, -1)

        # BxD --> Bx1xD
        repeated_class_token: nn.Parameter = self.class_token.repeat(num_batches, 1).view(num_batches, 1, -1)
        # concatenate Bx1xD and BxNxD --> Bx(1+N)xD
        embeddings: torch.Tensor = torch.cat((repeated_class_token, projected_embeddings), dim=1)

        return embeddings
