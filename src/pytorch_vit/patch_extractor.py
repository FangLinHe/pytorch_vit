import torch
from torch import nn


class PatchExtractor(nn.Module):
    def __init__(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
