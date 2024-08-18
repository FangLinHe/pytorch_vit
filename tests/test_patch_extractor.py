from __future__ import annotations

import torch

from pytorch_vit.patch_extractor import PatchExtractor


def test_construct_patch_extractor():
    batches: int = 2
    channels: int = 3
    image_height: int = 28
    image_width: int = 32
    patch_height: int = 4
    patch_width: int = 8

    extractor = PatchExtractor(patch_h=patch_height, patch_w=patch_width)
    images: torch.Tensor = torch.rand((batches, channels, image_height, image_width))
    patches: torch.Tensor = extractor(images)
    expected_num_patches: int = image_height * image_width // patch_height // patch_width
    exptected_patch_len: int = patch_height * patch_width * channels

    assert patches.shape == (batches, expected_num_patches, exptected_patch_len)
