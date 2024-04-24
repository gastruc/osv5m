import torch
import torch.nn as nn
import pandas as pd

from models.networks.utils import UnormGPS


class HybridHead(nn.Module):
    """Classification head followed by regression head for the network."""

    def __init__(self, final_dim, quadtree_path, use_tanh, scale_tanh):
        super().__init__()
        self.final_dim = final_dim
        self.use_tanh = use_tanh
        self.scale_tanh = scale_tanh

        self.unorm = UnormGPS()

        if quadtree_path is not None:
            quadtree = pd.read_csv(quadtree_path)
            self.init_quadtree(quadtree)

    def init_quadtree(self, quadtree):
        quadtree[["min_lat", "max_lat"]] /= 90.0
        quadtree[["min_lon", "max_lon"]] /= 180.0
        self.register_buffer(
            "cell_center",
            0.5 * torch.tensor(quadtree[["max_lat", "max_lon"]].values)
            + 0.5 * torch.tensor(quadtree[["min_lat", "min_lon"]].values),
        )
        self.register_buffer(
            "cell_size",
            torch.tensor(quadtree[["max_lat", "max_lon"]].values)
            - torch.tensor(quadtree[["min_lat", "min_lon"]].values),
        )

    def forward(self, x, gt_label):
        """Forward pass of the network.
        x : Union[torch.Tensor, dict] with the output of the backbone.
        """

        classification_logits = x[..., : self.final_dim]
        classification = classification_logits.argmax(dim=-1)

        regression = x[..., self.final_dim :]

        if self.use_tanh:
            regression = self.scale_tanh * torch.tanh(regression)

        regression = regression.view(regression.shape[0], -1, 2)

        if self.training:
            regression = torch.gather(
                regression,
                1,
                gt_label.unsqueeze(-1).unsqueeze(-1).expand(regression.shape[0], 1, 2),
            )[:, 0, :]
            size = 2.0 / self.cell_size[gt_label]
            center = self.cell_center[gt_label]
            gps = (
                self.cell_center[gt_label] + regression * self.cell_size[gt_label] / 2.0
            )
        else:
            regression = torch.gather(
                regression,
                1,
                classification.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(regression.shape[0], 1, 2),
            )[:, 0, :]
            size = 2.0 / self.cell_size[classification]
            center = self.cell_center[classification]
            gps = (
                self.cell_center[classification]
                + regression * self.cell_size[classification] / 2.0
            )

        gps = self.unorm(gps)

        return {
            "label": classification_logits,
            "gps": gps,
            "size": size,
            "center": center,
            "reg": regression,
        }

class HybridHeadCentroid(nn.Module):
    """Classification head followed by regression head for the network."""

    def __init__(self, final_dim, quadtree_path, use_tanh, scale_tanh):
        super().__init__()
        self.final_dim = final_dim
        self.use_tanh = use_tanh
        self.scale_tanh = scale_tanh

        self.unorm = UnormGPS()
        if quadtree_path is not None:
            quadtree = pd.read_csv(quadtree_path)
            self.init_quadtree(quadtree)

    def init_quadtree(self, quadtree):
        quadtree[["min_lat", "max_lat", "mean_lat"]] /= 90.0
        quadtree[["min_lon", "max_lon", "mean_lon"]] /= 180.0
        self.cell_center = torch.tensor(quadtree[["mean_lat", "mean_lon"]].values)
        self.cell_size_up = torch.tensor(quadtree[["max_lat", "max_lon"]].values) - torch.tensor(quadtree[["mean_lat", "mean_lon"]].values)
        self.cell_size_down = torch.tensor(quadtree[["mean_lat", "mean_lon"]].values) - torch.tensor(quadtree[["min_lat", "min_lon"]].values)

    def forward(self, x, gt_label):
        """Forward pass of the network.
        x : Union[torch.Tensor, dict] with the output of the backbone.
        """
        classification_logits = x[..., : self.final_dim]
        classification = classification_logits.argmax(dim=-1)
        self.cell_size_up = self.cell_size_up.to(classification.device)
        self.cell_center = self.cell_center.to(classification.device)
        self.cell_size_down = self.cell_size_down.to(classification.device)

        regression = x[..., self.final_dim :]

        if self.use_tanh:
            regression = self.scale_tanh * torch.tanh(regression)

        regression = regression.view(regression.shape[0], -1, 2)

        if self.training:
            regression = torch.gather(
                regression,
                1,
                gt_label.unsqueeze(-1).unsqueeze(-1).expand(regression.shape[0], 1, 2),
            )[:, 0, :]
            size = torch.where(
                regression > 0,
                self.cell_size_up[gt_label],
                self.cell_size_down[gt_label],
            )
            center = self.cell_center[gt_label]
            gps = self.cell_center[gt_label] + regression * size
        else:
            regression = torch.gather(
                regression,
                1,
                classification.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(regression.shape[0], 1, 2),
            )[:, 0, :]
            size = torch.where(
                regression > 0,
                self.cell_size_up[classification],
                self.cell_size_down[classification],
            )
            center = self.cell_center[classification]
            gps = self.cell_center[classification] + regression * size

        gps = self.unorm(gps)

        return {
            "label": classification_logits,
            "gps": gps,
            "size": 1.0 / size,
            "center": center,
            "reg": regression,
        }


class SharedHybridHead(HybridHead):
    """Classification head followed by SHARED regression head for the network."""

    def forward(self, x, gt_label):
        """Forward pass of the network.
        x : Union[torch.Tensor, dict] with the output of the backbone.
        """

        classification_logits = x[..., : self.final_dim]
        classification = classification_logits.argmax(dim=-1)

        regression = x[..., self.final_dim :]

        if self.use_tanh:
            regression = self.scale_tanh * torch.tanh(regression)

        if self.training:
            gps = (
                self.cell_center[gt_label] + regression * self.cell_size[gt_label] / 2.0
            )
        else:
            gps = (
                self.cell_center[classification]
                + regression * self.cell_size[classification] / 2.0
            )

        gps = self.unorm(gps)

        return {"label": classification_logits, "gps": gps}
