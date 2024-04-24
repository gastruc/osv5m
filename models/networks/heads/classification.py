import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Classification head for the network."""

    def __init__(self, id_to_gps):
        super().__init__()
        self.id_to_gps = id_to_gps

    def forward(self, x):
        """Forward pass of the network.
        x : Union[torch.Tensor, dict] with the output of the backbone.
        """
        gps = self.id_to_gps(x.argmax(dim=-1))
        return {"label": x, **gps}
