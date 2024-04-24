from models.networks.utils import UnormGPS
import torch.nn as nn
from torch.nn.functional import tanh
import torch


class RegressionHead(nn.Module):
    def __init__(self, use_tanh=False):
        super().__init__()
        self.unorm = UnormGPS()
        self.use_tanh = use_tanh

    def forward(self, x):
        """Forward pass of the network.
        x : Union[torch.Tensor, dict] with the output of the backbone.
        """
        if self.use_tanh:
            x = tanh(x)
        gps = self.unorm(x)
        return {"gps": gps}


class RegressionHeadAngle(nn.Module):
    def __init__(self):
        super().__init__()
        self.unorm = UnormGPS()

    def forward(self, x):
        """Forward pass of the network.
        x : Union[torch.Tensor, dict] with the output of the backbone.
        """
        x1 = x[:, 0].pow(2)
        x2 = x[:, 1].pow(2)
        x3 = x[:, 2].pow(2)
        x4 = x[:, 3].pow(2)
        cos_lambda = x1 / (x1 + x2)
        sin_lambda = x2 / (x1 + x2)
        cos_phi = x3 / (x3 + x4)
        sin_phi = x4 / (x3 + x4)
        lbd = torch.atan2(sin_lambda, cos_lambda)
        phi = torch.atan2(sin_phi, cos_phi)
        gps = torch.cat((lbd.unsqueeze(1), phi.unsqueeze(1)), dim=1)
        # gps = self.unorm(x)
        return {"gps": gps}
