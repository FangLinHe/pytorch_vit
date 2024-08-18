from __future__ import annotations

import torch

from pytorch_vit.patch_embedder import PatchEmbedder


def test_patch_embedder():
    num_batches: int = 2
    num_patches: int = 6
    patch_length: int = 256
    embedding_dim: int = 128
    embedder = PatchEmbedder(num_patches, patch_length, embedding_dim)
    patches: torch.Tensor = torch.rand((num_batches, num_patches, patch_length))
    embeddings: torch.Tensor = embedder(patches)
    assert embeddings.shape == (num_batches, 1 + num_patches, embedding_dim)
    with torch.no_grad():
        inferred_class_tokens: torch.Tensor = (embeddings - embedder.position_embeddings)[:, 0:1, :]
        assert (inferred_class_tokens - embedder.class_token).abs().max() < 1e-5
