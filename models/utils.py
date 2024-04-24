import os
from os.path import abspath as abp
import torch
import hydra
from hydra import initialize, compose
from models.module import Geolocalizer
from omegaconf import OmegaConf, open_dict
from os.path import join
from hydra.utils import instantiate

def load_model_config(path):
    # given the directory of os.cwd()
    # compute the relative path to path
    path = abp(path)
    rel_path = os.path.relpath(path, start=os.path.split(__file__)[0])

    with initialize(version_base=None, config_path=rel_path):
        cfg = compose(config_name="config", overrides=[])

    checkpoint = torch.load(join(path, "last.ckpt"))
    del checkpoint["state_dict"][
        "model.backbone.clip.vision_model.embeddings.position_ids"
    ]
    torch.save(checkpoint, join(path, "last2.ckpt"))

    with open_dict(cfg):
        cfg.checkpoint = join(path, "last2.ckpt")

    cfg.num_classes = 11399
    cfg.model.network.mid.instance.final_dim = cfg.num_classes * 3
    cfg.model.network.head.final_dim = cfg.num_classes * 3
    cfg.model.network.head.instance.quadtree_path = join(path, "quadtree_10_1000.csv")

    cfg.dataset.train_dataset.path = ""
    cfg.dataset.val_dataset.path = ""
    cfg.dataset.test_dataset.path = ""
    cfg.logger.save_dir = ""
    cfg.data_dir = ""
    cfg.root_dir = ""
    cfg.mode = "test"
    cfg.model.network.backbone.instance.path = (
        "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    )
    return cfg.dataset.test_transform, cfg.model, join(path, "last2.ckpt"), True

def load_model(path):
    transform_config, model_config, checkpoint_path, delete = load_model_config(path)

    transform = instantiate(transform_config)
    model = Geolocalizer.load_from_checkpoint(checkpoint_path, cfg=model_config)
    if delete:
        os.remove(checkpoint_path)

    return model, transform