from __future__ import annotations

import torch

from pytorch_vit.patch_extractor import PatchExtractor


def test_extracted_patches_shape():
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

def test_extracted_patches_values():
    batches: int = 2
    image_height: int = 4
    image_width: int = 6
    patch_height: int = 2
    patch_width: int = 3
    image_1_end: int = image_height * image_width
    image_2_end: int = image_1_end * 2

    extractor = PatchExtractor(patch_h=patch_height, patch_w=patch_width)
    image_1: torch.Tensor = torch.arange(0, image_1_end).view((1, image_height, image_width))
    image_2: torch.Tensor = torch.arange(image_1_end, image_2_end).view((1, image_height, image_width))
    images: torch.Tensor = torch.stack((image_1, image_2,), dim=0)
    assert images.shape[0] == batches
    patches: torch.Tensor = extractor(images)

    assert patches.shape == (batches, image_height, image_width)
    expected_patches_1 = torch.Tensor(
        [
            [0, 1, 2, 6, 7, 8,],
            [3, 4, 5, 9, 10, 11,],
            [12, 13, 14, 18, 19, 20,],
            [15, 16, 17, 21, 22, 23,],
        ]
    ).type_as(patches)
    expected_patches_2: torch.Tensor = expected_patches_1 + 24
    expected_patches: torch.Tensor = torch.stack((expected_patches_1, expected_patches_2,), dim=0)
    assert (patches == expected_patches).all()
