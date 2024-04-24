import os
import argparse
from os.path import join

import PIL
import torch

from models.utils import load_model


def operate(transform, fp):
    return transform(PIL.Image.open(fp))


@torch.no_grad()
def inference(model, x):
    features = model.model.backbone({"img": x.to(model.device)})
    x = model.model.mid(features)
    x = model.model.head(x, None)
    return torch.rad2deg(x["gps"]).squeeze(0).cpu().tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to the model")
    parser.add_argument("--input_dir", help="Path to the input directory")
    parser.add_argument("--output_file", help="Path to the output file")
    args = parser.parse_args()

    model, transform = load_model(args.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.eval()
    file = open(args.output_file, "w")
    for f in os.listdir(args.input_dir):
        if not f.endswith(".jpg"):
            continue
        gps = inference(model, operate(transform, join(args.input_dir, f)).unsqueeze(0))
        print(f"{f},{gps[0]},{gps[1]}", file=file)
