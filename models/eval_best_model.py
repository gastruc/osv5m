import os
from typing import Any
import pytorch_lightning as L
import torch
from hydra.utils import instantiate
from models.huggingface import Geolocalizer

class EvalModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        os.chdir(cfg.network.root_dir)
        self.model = Geolocalizer.from_pretrained('osv5m/baseline')
        self.test_metrics = instantiate(cfg.test_metrics)

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        pass

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pred = self.model(batch)
        pass

    def on_validation_epoch_end(self):
        pass

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pred = self.model.forward_tensor(batch)
        self.test_metrics.update({"gps": pred}, batch)

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
