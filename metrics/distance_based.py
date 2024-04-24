import torch

from metrics.utils import haversine, reverse

from torchmetrics import Metric


class HaversineMetrics(Metric):
    """
    Computes the average haversine distance between the predicted and ground truth points.
    Compute the accuracy given some radiuses.
    Compute the Geoguessr score given some radiuses.

    Args:
        acc_radiuses (list): list of radiuses to compute the accuracy from
        acc_area (list): list of areas to compute the accuracy from.
        acc_data (list): list of auxilliary data to compute the accuracy from.
    """

    def __init__(
        self,
        acc_radiuses=[],
        acc_area=["country", "region", "sub-region", "city"],
        aux_data=[],
    ):
        super().__init__()
        self.add_state("haversine_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("geoguessr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        for acc in acc_radiuses:
            self.add_state(
                f"close_enough_points_{acc}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
        for acc in acc_area:
            self.add_state(
                f"close_enough_points_{acc}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.add_state(
                f"count_{acc}", default=torch.tensor(0), dist_reduce_fx="sum"
            )
        self.acc_radius = acc_radiuses
        self.acc_area = acc_area
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.aux = len(aux_data) > 0
        self.aux_list = aux_data
        if self.aux:
            self.aux_count = {}
            for col in self.aux_list:
                self.add_state(
                    f"aux_{col}",
                    default=torch.tensor(0.0),
                    dist_reduce_fx="sum",
                )

    def update(self, pred, gt):
        haversine_distance = haversine(pred["gps"], gt["gps"])
        for acc in self.acc_radius:
            self.__dict__[f"close_enough_points_{acc}"] += (
                haversine_distance < acc
            ).sum()
        if len(self.acc_area) > 0:
            area_pred, area_gt = reverse(pred["gps"], gt, self.acc_area)
        for acc in self.acc_area:
            self.__dict__[f"close_enough_points_{acc}"] += (
                area_pred[acc] == area_gt["_".join(["unique", acc])]
            ).sum()
            self.__dict__[f"count_{acc}"] += len(area_gt["_".join(["unique", acc])])
        self.haversine_sum += haversine_distance.sum()
        self.geoguessr_sum += 5000 * torch.exp(-haversine_distance / 1492.7).sum()

        if self.aux:
            if "land_cover" in self.aux_list:
                col = "land_cover"
                self.__dict__[f"aux_{col}"] += (
                    pred[col].argmax(dim=1) == gt[col].argmax(dim=1)
                ).sum()
            if "road_index" in self.aux_list:
                col = "road_index"
                self.__dict__[f"aux_{col}"] += (
                    pred[col].argmax(dim=1) == gt[col].argmax(dim=1)
                ).sum()
            if "drive_side" in self.aux_list:
                col = "drive_side"
                self.__dict__[f"aux_{col}"] += (
                    (pred[col] > 0.5).float() == gt[col]
                ).sum()
            if "climate" in self.aux_list:
                col = "climate"
                self.__dict__[f"aux_{col}"] += (
                    pred[col].argmax(dim=1) == gt[col].argmax(dim=1)
                ).sum()
            if "soil" in self.aux_list:
                col = "soil"
                self.__dict__[f"aux_{col}"] += (
                    pred[col].argmax(dim=1) == gt[col].argmax(dim=1)
                ).sum()
            if "dist_sea" in self.aux_list:
                col = "dist_sea"
                self.__dict__[f"aux_{col}"] += (
                    (pred[col] - gt[col]).pow(2).sum(dim=1).sum()
                )

        self.count += pred["gps"].shape[0]

    def compute(self):
        output = {
            "Haversine": self.haversine_sum / self.count,
            "Geoguessr": self.geoguessr_sum / self.count,
        }
        for acc in self.acc_radius:
            output[f"Accuracy_{acc}_km_radius"] = (
                self.__dict__[f"close_enough_points_{acc}"] / self.count
            )
        for acc in self.acc_area:
            output[f"Accuracy_{acc}"] = (
                self.__dict__[f"close_enough_points_{acc}"]
                / self.__dict__[f"count_{acc}"]
            )

        if self.aux:
            for col in self.aux_list:
                output["_".join(["Accuracy", col])] = (
                    self.__dict__[f"aux_{col}"] / self.count
                )

        return output
