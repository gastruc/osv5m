import os
from models.module import Geolocalizer
import hydra
import wandb
from os.path import isfile, join
from shutil import copyfile

import torch

from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import LearningRateMonitor
from lightning_fabric.utilities.rank_zero import _get_rank

from models.module import Geolocalizer

torch.set_float32_matmul_precision("high")  # TODO do we need that?

# Registering the "eval" resolver allows for advanced config
# interpolation with arithmetic operations in hydra:
# https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html
OmegaConf.register_new_resolver("eval", eval)


def wandb_init(cfg):
    directory = cfg.checkpoints.dirpath
    if isfile(join(directory, "wandb_id.txt")):
        with open(join(directory, "wandb_id.txt"), "r") as f:
            wandb_id = f.readline()
    else:
        rank = _get_rank()
        wandb_id = wandb.util.generate_id()
        print(f"Generated wandb id: {wandb_id}")
        if rank == 0 or rank is None:
            with open(join(directory, "wandb_id.txt"), "w") as f:
                f.write(str(wandb_id))

    return wandb_id


def load_model(cfg, dict_config, wandb_id, callbacks):
    directory = cfg.checkpoints.dirpath
    if isfile(join(directory, "last.ckpt")):
        checkpoint_path = join(directory, "last.ckpt")
        logger = instantiate(cfg.logger, id=wandb_id, resume="allow")
        model = Geolocalizer.load_from_checkpoint(checkpoint_path, cfg=cfg.model)
        ckpt_path = join(directory, "last.ckpt")
        print(f"Loading form checkpoint ... {ckpt_path}")
    else:
        ckpt_path = None
        logger = instantiate(cfg.logger, id=wandb_id, resume="allow")
        log_dict = {"model": dict_config["model"], "dataset": dict_config["dataset"]}
        logger._wandb_init.update({"config": log_dict})
        model = Geolocalizer(cfg.model)

    trainer, strategy = cfg.trainer, cfg.trainer.strategy
    trainer = instantiate(
        trainer, strategy=strategy, logger=logger, callbacks=callbacks
    )
    return trainer, model, ckpt_path


def project_init(cfg):
    print("Working directory set to {}".format(os.getcwd()))
    directory = cfg.checkpoints.dirpath
    os.makedirs(directory, exist_ok=True)
    copyfile(".hydra/config.yaml", join(directory, "config.yaml"))


def callback_init(cfg):
    checkpoint_callback = instantiate(cfg.checkpoints)
    progress_bar = instantiate(cfg.progress_bar)
    lr_monitor = LearningRateMonitor()
    callbacks = [checkpoint_callback, progress_bar, lr_monitor]
    return callbacks


def init_datamodule(cfg):
    class_name = cfg.datamodule.train_dataset.class_name
    datamodule = instantiate(cfg.datamodule)
    if class_name is not None:
        cfg.num_classes = datamodule.num_classes
    return datamodule


def hydra_boilerplate(cfg):
    dict_config = OmegaConf.to_container(cfg, resolve=True)
    callbacks = callback_init(cfg)
    datamodule = init_datamodule(cfg)
    project_init(cfg)
    wandb_id = wandb_init(cfg)
    trainer, model, ckpt_path = load_model(cfg, dict_config, wandb_id, callbacks)
    return trainer, model, datamodule, ckpt_path


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    trainer, model, datamodule, ckpt_path = hydra_boilerplate(cfg)
    model.datamodule = datamodule
    # model = torch.compile(model)
    if cfg.mode == "train":
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    elif cfg.mode == "eval":
        trainer.test(model, datamodule=datamodule)
    elif cfg.mode == "traineval":
        cfg.mode = "train"
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        cfg.mode = "test"
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
