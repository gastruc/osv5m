import torch
from torch import nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
from huggingface_hub import PyTorchModelHubMixin

class Geolocalizer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        self.config = OmegaConf.create(config)
        self.transform = instantiate(self.config.transform)
        self.model = instantiate(self.config.model)
        self.head = self.model.head
        self.mid = self.model.mid
        self.backbone = self.model.backbone

    def forward(self, img: torch.Tensor):
        output = self.head(self.mid(self.backbone({"img": img})), None)
        return output["gps"]

    def forward_tensor(self, img: torch.Tensor):
        output = self.head(self.mid(self.backbone(img)), None)
        return output["gps"]
        