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
                BxNx(C * P_h * P_w), where N = H * W / (P_h * P_w)
        """
        # Move dim=1 to dim=3: dim(B, C, H, W) --> dim(B, H, W, C)
        axis_moved: torch.Tensor = x.moveaxis(1, 3)

        # Extract patches along dim=1: dim(B, H, W, C) --> dim(B, H / P_h, W, C, P_h)
        h_patches_unfolded: torch.Tensor = axis_moved.unfold(1, self.patch_h, self.patch_h)

        # Extract patches along dim=2: dim(B, H / P_h, W, C, P_h) --> dim(B, H / P_h, W / P_w, C, P_h, P_w)
        patches_unfolded: torch.Tensor = h_patches_unfolded.unfold(2, self.patch_w, self.patch_w)

        # Flatten each patch: dim(B, H / P_h, W / P_w, C, P_h, P_w) --> dim(B, H / P_h, W / P_w, C * P_h * P_w)
        flattend_patches: torch.Tensor = patches_unfolded.flatten(3)

        # Flatten patches: dim(B, H / P_h, W / P_w, C * P_h * P_w) --> dim(B, N, C * P_h * P_w), N = H * W / (P_h * P_w)
        patches: torch.Tensor = flattend_patches.flatten(1, 2)  # BxNx(C * P_h * P_w)

        return patches
