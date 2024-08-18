from __future__ import annotations

from pytorch_vit.patch_embedder import PatchEmbedder


def test_construct_patch_embedder():
    embedder = PatchEmbedder()
    assert embedder is not None
