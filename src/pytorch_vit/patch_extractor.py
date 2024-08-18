from __future__ import annotations

import torch
from torch import nn


class PatchExtractor(nn.Module):
    def __init__(self, patch_h: int, patch_w: int) -> None:
        super().__init__()
        self.patch_h: int = patch_h
        self.patch_w: int = patch_w
        self.patch_len: int = patch_h * patch_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts patches from input image

        Args:
            x (torch.Tensor): Images have dimension BxCxHxW

        Returns:
            torch.Tensor: sequences of flattened 2D patches with dimension
                BxNx(P_h * P_w * C), where N = H * W / (P_h * P_w)
        """
        axis_moved: torch.Tensor = x.moveaxis(1, 3)  # BCHW to BHWC
        h_patches_unfolded: torch.Tensor = axis_moved.unfold(1, self.patch_h, self.patch_h)  # Bx(H/P_h)xWxCxP_h
        patches_unfolded: torch.Tensor = h_patches_unfolded.unfold(
            2, self.patch_w, self.patch_w
        )  # Bx(H/P_h)x(W/P_w)xCxP_hxP_w
        flattend_patches: torch.Tensor = patches_unfolded.flatten(3)  # Bx(H/P_h)x(W/P_w)x(C * P_h * P_w)
        patches: torch.Tensor = flattend_patches.flatten(1, 2)  # Bx(H/P_h)x(W/P_w)x(C * P_h * P_w)

        return patches
