import os
import hydra
import PIL
from os.path import join
from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import ModelCard, EvalResult, ModelCardData
from huggingface_hub import login, whoami, create_repo
from models.module import Geolocalizer as GeolocalizerPT
from models.utils import load_model_config


def save_as_huggingface(args):
    transform_config, model_config, checkpoint_path = load_model_config(args.config_path)[:3]
    model_pt = GeolocalizerPT.load_from_checkpoint(checkpoint_path, cfg=model_config)

    ocd = {
        'model': OmegaConf.to_container(model_config.network.instance, resolve=True),
        'transform': OmegaConf.to_container(transform_config)
    }

    hf_model = Geolocalizer(ocd)
    hf_model.model.load_state_dict(model_pt.model.state_dict())
    hf_model.save_pretrained('test/', config=ocd)
    Geolocalizer.from_pretrained('test/')

    if args.tag is not None:
        login()
        user = whoami()['name']
        repo_id = f'{user}/{args.tag}'

        url = create_repo(repo_id, exist_ok=True)
        if args.update_model_card:
            card_data = ModelCardData(
                language='en', license='mit', library_name='pytorch',
                model_name=f'{args.tag}',
                eval_results = [
                    EvalResult(
                        task_type='Geoscore',
                        dataset_type='geolocation',
                        dataset_name='OSV-5M',
                        metric_type='geoscore',
                        metric_value=3361
                    ),
                    EvalResult(
                        task_type='Haversine Distance',
                        dataset_type='geolocation',
                        dataset_name='OSV-5M',
                        metric_type='haversine distance',
                        metric_value=1814
                    ),
                    EvalResult(
                        task_type='Country classification',
                        dataset_type='geolocation',
                        dataset_name='OSV-5M',
                        metric_type='country accuracy',
                        metric_value=68
                    ),
                    EvalResult(
                        task_type='Region classification',
                        dataset_type='geolocation',
                        dataset_name='OSV-5M',
                        metric_type='region accuracy',
                        metric_value=39.4
                    ),
                    EvalResult(
                        task_type='Area classification',
                        dataset_type='geolocation',
                        dataset_name='OSV-5M',
                        metric_type='area accuracy',
                        metric_value=10.3
                    ),
                    EvalResult(
                        task_type='City classification',
                        dataset_type='geolocation',
                        dataset_name='OSV-5M',
                        metric_type='city accuracy',
                        metric_value=5.9
                    ),
                ]
            )

            card = ModelCard.from_template(
                card_data,
                model_id=f'{args.tag}',
                model_description="Geolocation benchmark on OpenStreetView-5M dataset",
                developers="<tobereleased>", #OpenStreetView-5M Team (Imagine - ENPC/CNRS/LIGM/UGE/IGN)
                repo="<tobereleased>", #https://github.com/gastruc/osv5M
                
            )
            card.push_to_hub(repo_id)
        hf_model.push_to_hub(repo_id, config=ocd)
        Geolocalizer.from_pretrained(repo_id)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--config_path", help="Path to the model")
    parser.add_argument('-t', "--tag", help="Tag for the model")
    parser.add_argument("--images_dir", help="Path to the input directory")
    parser.add_argument("--update_model_card", action='store_true')
    args = parser.parse_args()

    save_as_huggingface(args)
    geoloc = Geolocalizer.from_pretrained('test/').eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for f in os.listdir(args.images_dir):
        if not f.endswith(('jpg', 'png', 'jpeg')):
            continue
        gps = geoloc(geoloc.transform(PIL.Image.open(join(args.images_dir, f))).unsqueeze(0))
