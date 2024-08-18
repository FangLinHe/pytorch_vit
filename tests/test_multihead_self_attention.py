from __future__ import annotations

from pytorch_vit.multihead_self_attention import MultiheadSelfAttention


def test_construct_multihead_self_attention():
    msa = MultiheadSelfAttention(16)
    assert msa is not None
