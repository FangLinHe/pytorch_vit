from __future__ import annotations

from pytorch_vit.patch_extractor import PatchExtractor


def test_construct_patch_extractor():
    extractor = PatchExtractor()
    assert extractor is not None
