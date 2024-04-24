import torch
import numpy as np
from torch import nn


class NormGPS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Normalize latitude longtitude radians to -1, 1."""  # not used currently
        return x / torch.Tensor([np.pi * 0.5, np.pi]).unsqueeze(0).to(x.device)


class UnormGPS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Unormalize latitude longtitude radians to -1, 1."""
        x = torch.clamp(x, -1, 1)
        return x * torch.Tensor([np.pi * 0.5, np.pi]).unsqueeze(0).to(x.device)
