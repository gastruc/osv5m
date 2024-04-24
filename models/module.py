import os
from typing import Any
import pytorch_lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate
import copy
import pandas as pd
import numpy as np


class Geolocalizer(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(cfg.network.instance)
        if cfg.text_tuning:
            self.text_model = instantiate(cfg.text_network.instance)
        self.loss = instantiate(cfg.loss)
        self.val_metrics = instantiate(cfg.val_metrics)
        self.test_metrics = instantiate(cfg.test_metrics)
        self.text_tuning = cfg.text_tuning

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        if self.text_tuning:
            pred["text_features"] = self.text_model(batch)
        loss = self.loss(pred, batch, average=True)
        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pred = self.model(batch)
        if self.text_tuning:
            pred["text_features"] = self.text_model(batch)
        loss = self.loss(pred, batch, average=True)["loss"]
        self.val_metrics.update(pred, batch)
        self.log("val/loss", loss, sync_dist=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pred = self.model(batch)
        self.test_metrics.update(pred, batch)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"test/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):
        lora_params = []
        backbone_params = []
        other_params = []
        last_block_params = []
        for name, param in self.model.named_parameters():
            if "lora" in name:
                lora_params.append(param)
            elif "backbone" in name:
                if self.cfg.optimizer.diff_backbone_last and ".11." in name:
                    last_block_params.append(param)
                else:
                    backbone_params.append(param)
            else:
                other_params.append(param)

        params_to_optimize = [{"params": other_params}]
        if self.cfg.optimizer.unfreeze_lr:
            params_to_optimize += [
                {"params": backbone_params, "lr": self.cfg.optimizer.backbone_lr}
            ]
            if self.cfg.optimizer.diff_backbone_last:
                params_to_optimize += [
                    {
                        "params": last_block_params,
                        "lr": self.cfg.optimizer.last_block_lr,
                    }
                ]
        if len(lora_params) > 0:
            # LoRA params sometimes train better with a different lr (~1e-4 for CLIP)
            params_to_optimize += [
                {"params": lora_params, "lr": self.cfg.optimizer.lora_lr}
            ]
        if self.cfg.optimizer.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(self.model, [nn.LayerNorm])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.cfg.optimizer.optim.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = instantiate(
                self.cfg.optimizer.optim, optimizer_grouped_parameters
            )
        else:
            optimizer = instantiate(self.cfg.optimizer.optim, params_to_optimize)
        scheduler = instantiate(self.cfg.lr_scheduler)(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    Taken from HuggingFace transformers.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
