from __future__ import annotations

from pytorch_vit.position_embedder import PositionEmbedder


def test_construct_position_embedder():
    embedder = PositionEmbedder()
    assert embedder is not None
