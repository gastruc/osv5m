import pandas as pd
import torch
from torch import nn
from models.networks.utils import UnormGPS


class Random(nn.Module):
    def __init__(self, num_output):
        """Random"""
        super().__init__()
        self.num_output = num_output
        self.unorm = UnormGPS()

    def forward(self, x):
        """Predicts GPS coordinates from an image.
        Args:
            x: torch.Tensor with features
        """
        #x = x["img"]
        gps = torch.rand((x.shape[0], self.num_output), device=x.device) * 2 - 1
        return {"gps": self.unorm(gps)}


class RandomCoords(nn.Module):
    def __init__(self, coords_path: str):
        """Randomly sample from a list of coordinates
        Args:
            coords_path: str with path to csv file with coordinates
        """
        super().__init__()
        coordinates = pd.read_csv(coords_path)
        longitudes = coordinates["longitude"].values / 180
        latitudes = coordinates["latitude"].values / 90
        self.unorm = UnormGPS()
        del coordinates

        self.N = len(longitudes)
        assert len(longitudes) == len(latitudes)
        self.coordinates = torch.stack(
            [torch.tensor(latitudes), torch.tensor(longitudes)],
            dim=-1,
        )
        del longitudes, latitudes

    def forward(self, x):
        """Predicts GPS coordinates from an image.
        Args:
            x: torch.Tensor with features
        """
        x = x["img"]
        # randomly select a coordinate in the list
        n = torch.randint(0, self.N, (x.shape[0],))
        return {"gps": self.unorm(self.coordinates[n].to(x.device))}
