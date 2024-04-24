import torch.hub

from transformers import (
    CLIPVisionModel,
    CLIPVisionConfig,
    CLIPModel,
    CLIPProcessor,
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPTextConfig,
    CLIPVisionModelWithProjection,
    ResNetModel,
    ResNetConfig
)
from torch import nn

from PIL import Image
import requests


class CLIP(nn.Module):
    def __init__(self, path):
        """Initializes the CLIP model."""
        super().__init__()
        if path == "":
            config_vision = CLIPVisionConfig()
            self.clip = CLIPVisionModel(config_vision)
        else:
            self.clip = CLIPVisionModel.from_pretrained(path)

    def forward(self, x):
        """Predicts CLIP features from an image.
        Args:
            x (dict that contains "img": torch.Tensor): Input batch
        """
        features = self.clip(pixel_values=x["img"])["last_hidden_state"]
        return features


class CLIPJZ(nn.Module):
    def __init__(self, path):
        """Initializes the CLIP model."""
        super().__init__()
        if path == "":
            config_vision = CLIPVisionConfig()
            self.clip = CLIPVisionModel(config_vision)
        else:
            self.clip = CLIPVisionModel.from_pretrained(path)

    def forward(self, x):
        """Predicts CLIP features from an image.
        Args:
            x (dict that contains "img": torch.Tensor): Input batch
        """
        features = self.clip(pixel_values=x["img"])["last_hidden_state"]
        return features


class StreetCLIP(nn.Module):
    def __init__(self, path):
        """Initializes the CLIP model."""
        super().__init__()
        self.clip = CLIPModel.from_pretrained(path)
        self.transform = CLIPProcessor.from_pretrained(path)

    def forward(self, x):
        """Predicts CLIP features from an image.
        Args:
            x (dict that contains "img": torch.Tensor): Input batch
        """
        features = self.clip.get_image_features(
            **self.transform(images=x["img"], return_tensors="pt").to(x["gps"].device)
        ).unsqueeze(1)
        return features


class CLIPText(nn.Module):
    def __init__(self, path):
        """Initializes the CLIP model."""
        super().__init__()
        if path == "":
            config_vision = CLIPVisionConfig()
            self.clip = CLIPVisionModel(config_vision)
        else:
            self.clip = CLIPVisionModelWithProjection.from_pretrained(path)

    def forward(self, x):
        """Predicts CLIP features from an image.
        Args:
            x (dict that contains "img": torch.Tensor): Input batch
        """
        features = self.clip(pixel_values=x["img"])
        return features.image_embeds, features.last_hidden_state


class TextEncoder(nn.Module):
    def __init__(self, path):
        """Initializes the CLIP text model."""
        super().__init__()
        if path == "":
            config_vision = CLIPTextConfig()
            self.clip = CLIPTextModelWithProjection(config_vision)
            self.transform = AutoTokenizer()
        else:
            self.clip = CLIPTextModelWithProjection.from_pretrained(path)
            self.transform = AutoTokenizer.from_pretrained(path)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

    def forward(self, x):
        """Predicts CLIP features from text.
        Args:
            x (dict that contains "text": list): Input batch
        """
        features = self.clip(
            **self.transform(x["text"], padding=True, return_tensors="pt").to(
                x["gps"].device
            )
        ).text_embeds
        return features


class DINOv2(nn.Module):
    def __init__(self, tag) -> None:
        """Initializes the DINO model."""
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", tag)
        self.stride = 14  # ugly but dinov2 stride = 14

    def forward(self, x):
        """Predicts DINO features from an image."""
        x = x["img"]

        # crop for stride
        _, _, H, W = x.shape
        H_new = H - H % self.stride
        W_new = W - W % self.stride
        x = x[:, :, :H_new, :W_new]

        # forward features
        x = self.dino.forward_features(x)
        x = x["x_prenorm"]
        return x
    
class ResNet(nn.Module):
    def __init__(self, path):
        """Initializes the ResNet model."""
        super().__init__()
        if path == "":
            config_vision = ResNetConfig()
            self.resnet = ResNetModel(config_vision)
        else:
            self.resnet = ResNetModel.from_pretrained(path)

    def forward(self, x):
        """Predicts ResNet50 features from an image.
        Args:
            x (dict that contains "img": torch.Tensor): Input batch
        """
        features = self.resnet(x["img"])["pooler_output"]
        return features.squeeze()
