import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from os.path import join
from models.networks.utils import NormGPS


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "gps": torch.Tensor Bx2
            y: dict that contains "gps": torch.Tensor Bx2
        Returns:
            torch.Tensor: L1 loss between x and y: torch.Tensor([B])
        """
        return {"L1_loss": torch.abs(x["gps"] - y["gps"]).mean(dim=-1)}


class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "gps": torch.Tensor Bx2
            y: dict that contains "gps": torch.Tensor Bx2
        Returns:
            torch.Tensor: L2 loss between x and y: torch.Tensor([B])
        """
        return {"L2_loss": ((x["gps"] - y["gps"]) ** 2).mean(dim=-1)}


class L2Hybrid(nn.Module):
    def __init__(self):
        super(L2Hybrid, self).__init__()
        self.norm = NormGPS()

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "gps": torch.Tensor Bx2
            y: dict that contains "gps": torch.Tensor Bx2
        Returns:
            torch.Tensor: L2 loss between x and y: torch.Tensor([B])
        """
        return {
            "L2_loss": (
                (x["reg"] - (self.norm(y["gps"]) - x["center"]) * x["size"]) ** 2
            ).mean(dim=-1)
        }


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "label": torch.Tensor BxN
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B])
        """
        return {"cross_entropy_loss": self.loss(x["label"], y["label"])}


class HierarchicalCrossEntropyQuad(nn.Module):
    def __init__(self, data_path=""):
        super(HierarchicalCrossEntropyQuad, self).__init__()
        self.dict_losses = {"classif_loss": nn.CrossEntropyLoss(reduction="none")}
        for i in range(1, 10):
            self.dict_losses[f"quadtree_{i}_loss"] = nn.NLLLoss()
        self.matrixes = torch.load(join(data_path, "quadtree_matrixes.pt"))
        self.dicts = torch.load(join(data_path, "quadtree_dicts.pt"))
        self.id_to_quad = torch.load(join(data_path, "id_to_quad_10_1000.pt"))

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "label": torch.Tensor BxN
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: Hierarchical CrossEntropy for Quadtrees loss between x and y: torch.Tensor([B])
        """
        out = {"classif_loss": self.dict_losses["classif_loss"](x["label"], y["label"])}
        probas = nn.functional.softmax(x["label"], dim=1)
        device = x["label"].device
        gt = self.id_to_quad[y["label"].cpu()]
        for i in range(9):
            logits = torch.log(torch.mm(probas, self.matrixes[i].to(device)) + 1e-10)
            l = [s[: 9 - i] if len(s) >= 10 - i else s for s in gt]
            out[f"quadtree_{i+1}_loss"] = self.dict_losses[f"quadtree_{i+1}_loss"](
                logits, torch.tensor([self.dicts[i][item] for item in l]).to(device)
            )

        return out


class HierarchicalCrossEntropy(nn.Module):
    def __init__(self, path=""):
        super(HierarchicalCrossEntropy, self).__init__()
        self.city_loss = nn.CrossEntropyLoss(reduction="none")
        self.country_loss = nn.NLLLoss()
        self.area_loss = nn.NLLLoss()
        self.region_loss = nn.NLLLoss()
        self.city_to_country = torch.load(path + "city_to_country.pt")
        self.city_to_region = torch.load(path + "city_to_region.pt")
        self.city_to_area = torch.load(path + "city_to_area.pt")
        self.country_to_idx = torch.load(path + "country_to_idx.pt")
        self.region_to_idx = torch.load(path + "region_to_idx.pt")
        self.area_to_idx = torch.load(path + "area_to_idx.pt")

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "label": torch.Tensor BxN
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: Hierarchical CrossEntropy  loss between x and y: torch.Tensor([B])
        """
        country_mask = np.array(y["unique_country"]) != "NaN"
        self.city_to_country = self.city_to_country.to(x["label"].device)
        countries_probas = nn.functional.softmax(x["label"][country_mask], dim=1)
        countries_logits = torch.log(
            torch.mm(countries_probas, self.city_to_country) + 1e-10
        )
        country_gt = torch.tensor(
            [
                self.country_to_idx[item]
                for item in np.array(y["unique_country"])[country_mask]
            ]
        ).to(x["label"].device)

        region_mask = np.array(y["unique_region"]) != "NaN"
        self.city_to_region = self.city_to_region.to(x["label"].device)
        regions_probas = nn.functional.softmax(x["label"][region_mask], dim=1)
        regions_logits = torch.log(
            torch.mm(regions_probas, self.city_to_region) + 1e-10
        )
        region_gt = torch.tensor(
            [
                self.region_to_idx[item]
                for item in np.array(y["unique_region"])[region_mask]
            ]
        ).to(x["label"].device)

        area_mask = np.array(y["unique_sub-region"]) != "NaN"
        self.city_to_area = self.city_to_area.to(x["label"].device)
        areas_probas = nn.functional.softmax(x["label"][area_mask], dim=1)
        areas_logits = torch.log(torch.mm(areas_probas, self.city_to_area) + 1e-10)
        area_gt = torch.tensor(
            [
                self.area_to_idx[item]
                for item in np.array(y["unique_sub-region"])[area_mask]
            ]
        ).to(x["label"].device)

        return {
            "cross_entropy_country_loss": self.country_loss(
                countries_logits, country_gt
            ),
            "cross_entropy_city_loss": self.city_loss(x["label"], y["label"]),
            "cross_entropy_area_loss": self.area_loss(areas_logits, area_gt),
            "cross_entropy_region_loss": self.region_loss(regions_logits, region_gt),
        }


class LandCoverLoss(nn.Module):
    def __init__(self):
        super(LandCoverLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "land_cover": torch.Tensor BxN
            y: dict that contains "land_cover": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B])
        """
        return {
            "land_cover_cross_entropy_loss": self.loss(x["land_cover"], y["land_cover"])
        }


class RoadIndexLoss(nn.Module):
    def __init__(self):
        super(RoadIndexLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "road_index": torch.Tensor BxN
            y: dict that contains "road_index": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B])
        """
        return {"road_index_mse_loss": self.loss(x["road_index"], y["road_index"])}


class DriveSideLoss(nn.Module):
    def __init__(self):
        super(DriveSideLoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "drive_side": torch.Tensor BxN
            y: dict that contains "drive_side": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B])
        """
        return {"drive_side_bce_loss": self.loss(x["drive_side"], y["drive_side"])}


class ClimateLoss(nn.Module):
    def __init__(self):
        super(ClimateLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "climate": torch.Tensor BxN
            y: dict that contains "climate": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B])
        """
        return {"climate_cross_entropy_loss": self.loss(x["climate"], y["climate"])}


class SoilLoss(nn.Module):
    def __init__(self):
        super(SoilLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "soil": torch.Tensor BxN
            y: dict that contains "soil": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B])
        """
        return {"soil_cross_entropy_loss": self.loss(x["soil"], y["soil"])}


class DistSeaLoss(nn.Module):
    def __init__(self):
        super(DistSeaLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "dist_sea": torch.Tensor BxN
            y: dict that contains "dist_sea": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B])
        """
        return {"dist_sea_mse_loss": self.loss(x["dist_sea"], y["dist_sea"])}


class Haversine(nn.Module):
    def __init__(self):
        super(Haversine, self).__init__()

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "gps": torch.Tensor Bx2
            y: dict that contains "gps": torch.Tensor Bx2
        Returns:
            torch.Tensor: Haversine loss between x and y: torch.Tensor([B])
        Note:
            Haversine distance doesn't contain the 2 * 6371 constant.
        """
        x, y = x["gps"], y["gps"]
        lhs = torch.sin((x[:, 0] - y[:, 0]) / 2) ** 2
        rhs = (
            torch.cos(x[:, 0])
            * torch.cos(y[:, 0])
            * torch.sin((x[:, 1] - y[:, 1]) / 2) ** 2
        )
        a = lhs + rhs
        return {
            "haversine_loss": torch.arctan2(torch.sqrt(a), torch.sqrt(1 - a))
        }  # ommitting 2 * 6371 as both are a constant


class GeoguessrLoss(Haversine):
    def __init__(self):
        super(GeoguessrLoss, self).__init__()

    def forward(self, x, y):
        distance = super().forward(x, y)["haversine_loss"]
        loss = torch.exp(-distance / 1852)
        return {"geoguessr_loss": loss}


class InfoNCE(nn.Module):
    def __init__(self, tau=0.1):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def cosine_similarity(self, a, b, normalize=True):
        if normalize:
            w1 = a.norm(p=2, dim=1, keepdim=True)
            w2 = b.norm(p=2, dim=1, keepdim=True)
            sim_matrix = torch.mm(a, b.t()) / (w1 * w2.t()).clamp(min=1e-8)
        else:
            sim_matrix = torch.mm(a, b.t())
        return sim_matrix

    def forward(self, x, y=None):
        """
        neg_sim: BxB
        pos_sim: Bx1
        """
        features = x["features"]
        positive_features = x["pos_features"]
        pos_sim = F.cosine_similarity(
            features, positive_features, dim=1, eps=1e-8
        ).unsqueeze(1)
        neg_sim = self.cosine_similarity(features, features, normalize=True)

        b = neg_sim.shape[0]
        logits = (1 - torch.eye(b)).type_as(neg_sim) * neg_sim + torch.eye(b).type_as(
            pos_sim
        ) * pos_sim
        logits = logits / self.tau
        labels = torch.arange(b, dtype=torch.long).cuda()
        loss = F.cross_entropy(logits, labels)
        return {
            "contrastive_loss": loss,
        }


class TextNCE(nn.Module):
    def __init__(self, tau=0.1, num_devices=1):
        super(TextNCE, self).__init__()
        self.distributed = num_devices > 1
        self.tau = tau

    def cosine_similarity(self, a, b, normalize=True):
        if normalize:
            w1 = a.norm(p=2, dim=1, keepdim=True)
            w2 = b.norm(p=2, dim=1, keepdim=True)
            sim_matrix = torch.mm(a, b.t()) / (w1 * w2.t()).clamp(min=1e-8)
        else:
            sim_matrix = torch.mm(a, b.t())
        return sim_matrix

    def forward(self, x, y=None):
        """
        neg_sim: BxB
        pos_sim: Bx1
        """
        if self.distributed:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(x["features"]), dim=0
            )
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(x["text_features"]), dim=0
            )
            all_labels = torch.cat(torch.distributed.nn.all_gather(y["label"]), dim=0)
        else:
            all_image_features = x["features"]
            all_text_features = x["text_features"]
            all_labels = y["label"]
        labels_u = torch.unique(all_labels)
        logits = self.cosine_similarity(
            all_image_features, all_text_features, normalize=True
        )
        rows, cols = logits.size()
        indices = torch.arange(0, rows, device=all_image_features.device)
        loss = torch.sum(
            torch.logsumexp(
                logits[indices != indices.view(-1, 1)].view(rows, cols - 1) / self.tau,
                dim=1,
            )
        )
        for label in labels_u:
            if not (label == "NaN"):
                # Get the positive and negative examples
                idx = all_labels == label
                pos_logits = logits[idx][:, idx]
                # Compute the MIL-NCE loss
                loss += torch.sum(-torch.logsumexp(pos_logits / self.tau, dim=1))
        return {
            "contrastive_loss": loss,
        }


class MILNCE(nn.Module):
    def __init__(self, tau=0.1, num_devices=1):
        super(MILNCE, self).__init__()
        self.distributed = num_devices > 1
        self.tau = tau

    def cosine_similarity(self, a, b, normalize=True):
        if normalize:
            w1 = a.norm(p=2, dim=1, keepdim=True)
            w2 = b.norm(p=2, dim=1, keepdim=True)
            sim_matrix = torch.mm(a, b.t()) / (w1 * w2.t()).clamp(min=1e-8)
        else:
            sim_matrix = torch.mm(a, b.t())
        return sim_matrix

    def forward(self, x, y=None):
        """
        COmpute MIL-NCE loss
        """
        if self.distributed:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(x["features"]), dim=0
            )
            all_pos_features = torch.cat(
                torch.distributed.nn.all_gather(x["pos_features"]), dim=0
            )
            all_labels = torch.cat(torch.distributed.nn.all_gather(y["label"]), dim=0)
        else:
            all_image_features = x["features"]
            all_pos_features = x["pos_features"]
            all_labels = y["label"]
        labels_u = torch.unique(all_labels)
        features = torch.cat([all_image_features, all_pos_features])
        labels = torch.cat([all_labels, all_labels])
        logits = self.cosine_similarity(features, features, normalize=True)
        rows, cols = logits.size()
        indices = torch.arange(0, rows, device=features.device)
        loss = torch.sum(
            torch.logsumexp(
                logits[indices != indices.view(-1, 1)].view(rows, cols - 1) / self.tau,
                dim=1,
            )
        )
        for label in labels_u:
            if not (label == "NaN"):
                # Get the positive and negative examples
                idx = labels == label
                pos_logits = logits[idx][:, idx]

                rows, cols = pos_logits.size()
                indices = torch.arange(0, rows, device=features.device)
                pos_logits = pos_logits[indices != indices.view(-1, 1)].view(
                    rows, cols - 1
                )

                # Compute the MIL-NCE loss
                loss += torch.sum(-torch.logsumexp(pos_logits / self.tau, dim=1))
        return {
            "contrastive_loss": loss,
        }


class RegionMILNCE(nn.Module):
    def __init__(self, tau=0.1, num_devices=1):
        super(RegionMILNCE, self).__init__()
        self.distributed = num_devices > 1
        self.tau = tau

    def cosine_similarity(self, a, b, normalize=True):
        if normalize:
            w1 = a.norm(p=2, dim=1, keepdim=True)
            w2 = b.norm(p=2, dim=1, keepdim=True)
            sim_matrix = torch.mm(a, b.t()) / (w1 * w2.t()).clamp(min=1e-8)
        else:
            sim_matrix = torch.mm(a, b.t())
        return sim_matrix

    def forward(self, x, y=None):
        """
        neg_sim: BxB
        pos_sim: Bx1
        """
        if self.distributed:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(x["features"]), dim=0
            )
            all_pos_features = torch.cat(
                torch.distributed.nn.all_gather(x["pos_features"]), dim=0
            )
            all_labels = torch.cat(torch.distributed.nn.all_gather(y["label"]), dim=0)
        else:
            all_image_features = x["features"]
            all_pos_features = x["pos_features"]
            all_labels = y["label"]
        labels_u = torch.unique(all_labels)
        features = torch.cat([all_image_features, all_pos_features])
        labels = torch.cat([all_labels, all_labels])
        logits = self.cosine_similarity(features, features, normalize=True)
        rows, cols = logits.size()
        indices = torch.arange(0, rows, device=features.device)
        loss = torch.sum(
            torch.logsumexp(
                logits[indices != indices.view(-1, 1)].view(rows, cols - 1) / self.tau,
                dim=1,
            )
        )
        for label in labels_u:
            if not (label == "NaN"):
                # Get the positive and negative examples
                idx = labels == label
                pos_logits = logits[idx][:, idx]

                rows, cols = pos_logits.size()
                indices = torch.arange(0, rows, device=features.device)
                pos_logits = pos_logits[indices != indices.view(-1, 1)].view(
                    rows, cols - 1
                )

                # Compute the MIL-NCE loss
                loss += torch.sum(-torch.logsumexp(pos_logits / self.tau, dim=1))
        return {
            "contrastive_loss": loss / len(all_labels),
        }


LOSSES = {
    "l1": L1,
    "l2": L2,
    "l2_hybrid": L2Hybrid,
    "haversine": Haversine,
    "geoguessr": GeoguessrLoss,
    "crossentropy": CrossEntropy,
    "infonce": InfoNCE,
    "mil-nce": MILNCE,
    "text-nce": TextNCE,
    "land_cover": LandCoverLoss,
    "road_index": RoadIndexLoss,
    "drive_side": DriveSideLoss,
    "climate": ClimateLoss,
    "soil": SoilLoss,
    "dist_sea": DistSeaLoss,
    "hierarchical": HierarchicalCrossEntropy,
    "hier_quad": HierarchicalCrossEntropyQuad,
    "region_mil": RegionMILNCE,
}
AVERAGE = {False: lambda x: x, True: lambda x: x.mean(dim=-1)}


class Losses(nn.Module):
    """The Losses meta-object that can take a mix of losses."""

    def __init__(self, mix={}, aux_data=[], path="", num_devices=1):
        """Initializes the Losses object.
        Args:
            mix (dict): dictionary with keys "loss_name" and values weight
        """
        super(Losses, self).__init__()
        assert len(mix)
        self.aux = len(aux_data) > 0
        if self.aux:
            self.aux_list = aux_data
            total = ["land_cover", "drive_side", "climate", "soil", "dist_sea"]
            for col in self.aux_list:
                total.remove(col)
            for col in total:
                del mix[col]
        self.init_losses(mix, path, num_devices)

    def init_losses(self, mix, path="", num_devices=1):
        """Initializes the losses.
        Args:
            mix (dict): dictionary with keys "loss_name" and values weight
        """
        self.loss = {}
        for m, v in mix.items():
            m = m.lower()
            if m in ["hierarchical", "hier_quad"]:
                try:
                    self.loss[m] = (LOSSES[m](path), v)
                except KeyError:
                    raise KeyError(f"Loss {m} not found in {LOSSES.keys()}")
            elif m in ["region_mil", "mil-nce", "text-nce"]:
                try:
                    self.loss[m] = (LOSSES[m](num_devices=num_devices), v)
                except KeyError:
                    raise KeyError(f"Loss {m} not found in {LOSSES.keys()}")
            else:
                try:
                    self.loss[m] = (LOSSES[m](), v)
                except KeyError:
                    raise KeyError(f"Loss {m} not found in {LOSSES.keys()}")

    def forward(self, x, y, average=True):
        """Computes the losses.
        Args:
            x: dict that contains "gps": torch.Tensor Bx2 or "label": torch.Tensor BxN
            y: dict that contains "gps": torch.Tensor Bx2 or "label": torch.Tensor BxN
            average (bool): whether to average the losses or not
        Returns:
            dict: dictionary with losses
        """
        output = {"loss": 0}
        for loss_name, (loss, weight) in self.loss.items():
            loss_output = loss(x, y)
            for k, v in loss_output.items():
                v = AVERAGE[average](v)
                if k.endswith("_loss"):
                    output["loss"] += weight * v
                output[k] = v
        return output
