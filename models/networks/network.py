import torch
import numpy as np
from abc import ABC, abstractmethod
from torch import nn
from hydra.utils import instantiate
import copy
from peft import LoraConfig, get_peft_model
from utils.model_utils import print_trainable_parameters


def freeze(model):
    """Freezes the parameters of a model."""
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def unfreeze(model):
    """Unfreezes the parameters of a model.
    for p in model.parameters():
        p.requires_grad = True"""
    model_parameters = model.named_parameters()
    for name, param in model_parameters:
        if name in [
            "clip.vision_model.post_layernorm.weight",
            "clip.vision_model.post_layernorm.bias",
        ]:
            param.requires_grad = False
        else:
            param.requires_grad = True
    model.train()


def unfreeze_last(model):
    """Unfreezes the parameters of a model.
    for p in model.parameters():
        p.requires_grad = True"""
    model_parameters = model.named_parameters()
    for name, param in model_parameters:
        if len(name.split(".")) > 5:
            if name.split(".")[4] == "11":
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False
    model.train()


class FrozenBackbone(nn.Module):
    """Freezes the backbone of a network."""

    def __init__(self, backbone, mid, head):
        super().__init__()
        self.backbone = backbone.instance
        self.mid = mid.instance
        self.head = head.instance
        self.target_key = head.target_key
        freeze(self.backbone)

    def forward(self, x):
        """Forward pass of the network.
        x : Union[torch.Tensor, dict] with the output of the backbone.
        """
        with torch.no_grad():
            x = self.backbone(x)
        x = self.mid(x)
        x = self.head(x)
        return x


class UnfrozenBackbone(nn.Module):
    """Unfreezes the backbone of a network."""

    def __init__(self, backbone, mid, head):
        super().__init__()
        self.backbone = backbone.instance
        self.mid = mid.instance
        self.head = head.instance
        self.target_key = head.target_key
        unfreeze(self.backbone)

    def forward(self, x):
        """Forward pass of the network.
        x : Union[torch.Tensor, dict] with the output of the backbone.
        """
        x = self.backbone(x)
        x = self.mid(x)
        x = self.head(x)
        return x


class UnfrozenPartBackbone(nn.Module):
    """Unfreezes the backbone of a network."""

    def __init__(self, backbone, mid, head):
        super().__init__()
        self.backbone = backbone.instance
        self.mid = mid.instance
        self.head = head.instance
        self.target_key = head.target_key
        unfreeze_last(self.backbone)

    def forward(self, x):
        """Forward pass of the network.
        x : Union[torch.Tensor, dict] with the output of the backbone.
        """
        x = self.backbone(x)
        x = self.mid(x)
        x = self.head(x)
        return x


class NoFeatureBackbone(nn.Module):
    """Randomizes the backbone of a network."""

    def __init__(self, head):
        super().__init__()
        self.head = head.instance
        self.target_key = head.target_key

    def forward(self, x):
        """Forward pass of the network.
        x : Union[torch.Tensor, dict] with the output of the backbone.
        """
        return self.head(x)


class ContrastiveFrozenBackbone(FrozenBackbone):
    """Freezes the backbone of a network."""

    def __init__(self, backbone, mid, head, mode):
        super().__init__(backbone, mid, head)
        self.mode = mode

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
            if self.mode != "eval":
                x_pos = {
                    k.strip("pos_"): v.clone()
                    if isinstance(v, torch.Tensor)
                    else copy.deepcopy(v)
                    for k, v in x.items()
                    if k.startswith("pos_")
                }
                pos_features = self.backbone(x_pos)
        x = self.mid(features)
        x = self.head(x)
        if self.mode != "eval":
            return {
                "features": features[:, 0, :],
                "pos_features": pos_features[:, 0, :],
                **x,
            }
        return {
            "features": features[:, 0, :],
            **x,
        }


class ContrastiveUnFrozenPartBackbone(UnfrozenPartBackbone):
    """Freezes the backbone of a network."""

    def __init__(self, backbone, mid, head, mode):
        super().__init__(backbone, mid, head)
        self.mode = mode

    def forward(self, x):
        features = self.backbone(x)
        if self.mode != "eval":
            x_pos = {
                k.strip("pos_"): v.clone()
                if isinstance(v, torch.Tensor)
                else copy.deepcopy(v)
                for k, v in x.items()
                if k.startswith("pos_")
            }
            pos_features = self.backbone(x_pos)
        x = self.mid(features)
        x = self.head(x)
        if self.mode != "eval":
            return {
                "features": features[:, 0, :],
                "pos_features": pos_features[:, 0, :],
                **x,
            }
        return {
            "features": features[:, 0, :],
            **x,
        }


class ContrastiveUnFrozenBackbone(UnfrozenBackbone):
    """Freezes the backbone of a network."""

    def __init__(self, backbone, mid, head, mode):
        super().__init__(backbone, mid, head)
        self.mode = mode

    def forward(self, x):
        features = self.backbone(x)
        if self.mode != "eval":
            x_pos = {
                k.strip("pos_"): v.clone()
                if isinstance(v, torch.Tensor)
                else copy.deepcopy(v)
                for k, v in x.items()
                if k.startswith("pos_")
            }
            pos_features = self.backbone(x_pos)
        x = self.mid(features)
        x = self.head(x)
        if self.mode != "eval":
            return {
                "features": features[:, 0, :],
                "pos_features": pos_features[:, 0, :],
                **x,
            }
        return {
            "features": features[:, 0, :],
            **x,
        }


class TextContrastiveUnFrozenBackbone(UnfrozenBackbone):
    """Freezes the backbone of a network."""

    def __init__(self, backbone, mid, head):
        super().__init__(backbone, mid, head)

    def forward(self, x):
        con, features = self.backbone(x)
        x = self.mid(features)
        x = self.head(x)
        return {
            "features": con,
            **x,
        }


class LoraBackbone(nn.Module):
    """Wraps the backbone in a PEFT model for LoRA tuning."""

    def __init__(self, backbone, mid, head, r, alpha, dropout, bias):
        super().__init__()
        self.backbone = backbone.instance
        self.mid = mid.instance
        self.head = head.instance
        self.target_key = head.target_key
        freeze(self.backbone)

        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=bias,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        self.backbone = get_peft_model(self.backbone, config)
        print_trainable_parameters(self)

    def forward(self, x):
        """Forward pass of the network.
        x : Union[torch.Tensor, dict] with the output of the backbone.
        """
        x = self.backbone(x)
        x = self.mid(x)
        return self.head(x)


class HybridFrozenBackbone(FrozenBackbone):
    """Freezes the backbone of a network."""

    def forward(self, x):
        """Forward pass of the network.
        x : Union[torch.Tensor, dict] with the output of the backbone.
        """

        gt_label = x["label"] if self.training else None

        with torch.no_grad():
            x = self.backbone(x)
        x = self.mid(x)
        x = self.head(x, gt_label)
        return x


class HybridUnfrozenBackbone(UnfrozenBackbone):
    """Unfreezes the backbone of a network."""

    def forward(self, x):
        """Forward pass of the network.
        x : Union[torch.Tensor, dict] with the output of the backbone.
        """

        gt_label = x["label"] if self.training else None

        x = self.backbone(x)
        x = self.mid(x)
        x = self.head(x, gt_label)
        return x


class ContrastiveHybridUnFrozenBackbone(UnfrozenBackbone):
    """Freezes the backbone of a network."""

    def __init__(self, backbone, mid, head, mode):
        super().__init__(backbone, mid, head)
        self.mode = mode

    def forward(self, x):
        gt_label = x["label"] if self.training else None
        features = self.backbone(x)
        if self.mode != "eval":
            x_pos = {
                k.strip("pos_"): v.clone()
                if isinstance(v, torch.Tensor)
                else copy.deepcopy(v)
                for k, v in x.items()
                if k.startswith("pos_")
            }
            pos_features = self.backbone(x_pos)
        x = self.mid(features)
        x = self.head(x, gt_label)
        if self.mode != "eval":
            return {
                "features": features[:, 0, :],
                "pos_features": pos_features[:, 0, :],
                **x,
            }
        return {
            "features": features[:, 0, :],
            **x,
        }
