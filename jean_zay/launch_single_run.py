from launch import JeanZayExperiment
import argparse
from pathlib import Path

import os


def parse_mode():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    return args


cmd_modifiers = []
exps = []

exp_name = f"clip_cls"
job_name = f"self-cond"
model_name = f"first_exp"
jz_exp = JeanZayExperiment(model_name, job_name)
jz_exp.nodes = 1
jz_exp.num_gpus_per_node = 4
jz_exp.qos = "dev"
jz_exp.account = "ufh"
jz_exp.gpu_type = "v100"
jz_exp.time = "1:00:00"

exps.append(jz_exp)

trainer_modifiers = {
    "exp": exp_name,
    "model.name": model_name,
    "computer": "cluster-node-v100.yaml",
    "computer.devices": jz_exp.num_gpus_per_node,
    "computer.progress_bar_refresh_rate": 10,
    "computer.num_nodes": jz_exp.nodes,
    "data_dir": Path("/gpfsscratch/rech/ufh/ult23zz/datasets"),
    "model.network.backbone.instance.path": Path(
        "$DSDIR/HuggingFace_Models/laion/CLIP-ViT-B-32-laion2B-s34B-b79"
    ),
    "dataset.global_batch_size": 2048,
    "logger.offline": True,
}

exp_modifier = {"class_name": "country"}

cmd_modifiers.append(dict(trainer_modifiers, **exp_modifier))


if __name__ == "__main__":
    for exp, cmd_modifier in zip(exps, cmd_modifiers):
        args = parse_mode()
        exp.build_cmd(cmd_modifier)
        if args.launch == True:
            exp.launch()
