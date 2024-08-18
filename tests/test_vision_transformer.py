from __future__ import annotations

from pytorch_vit.vision_transformer import VisionTransformer


def test_construct_vit():
    vit = VisionTransformer()
    assert vit is not None
