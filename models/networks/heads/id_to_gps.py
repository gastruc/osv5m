import torch
from models.networks.utils import UnormGPS
import torch.nn as nn
import numpy as np


class IdToGPS(nn.Module):
    def __init__(self, id_to_gps: str):
        """Map index to gps coordinates (indices can be country or city ids)"""
        super().__init__()
        if "quadtree" in id_to_gps:
            self.id_to_gps = torch.load(
                "_".join(id_to_gps.split("_")[:-4] + id_to_gps.split("_")[-3:])
            )
        else:
            self.id_to_gps = torch.load(id_to_gps)
        #self.unorm = UnormGPS()

    def forward(self, x):
        """Mapping from country id to gps coordinates
        Args:
            x: torch.Tensor with features
        """

        if isinstance(x, dict):
            # for oracle
            labels, x = x["label"], x["img"]
        else:
            # predicted labels
            labels = x
        self.id_to_gps = self.id_to_gps.to(labels.device)
        #return {"gps": self.unorm(self.id_to_gps[labels])}
        return {"gps": self.id_to_gps[labels]}
