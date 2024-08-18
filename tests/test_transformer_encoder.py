from __future__ import annotations

from pytorch_vit.transformer_encoder import TransformerEncoder


def test_construct_transformer_encoder():
    encoder = TransformerEncoder(16, 128)
    assert encoder is not None
