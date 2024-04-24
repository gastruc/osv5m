# Plonk

## Getting Started
Install an environment:

```bash
conda create -n plonk pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda activate plonk
conda install -c conda-forge cartopy -y
pip install -r requirements.txt
pre-commit install
```

## Dataloader
```bash
# Using more workers in the dataloader
computer.num_workers=20

# Using all data augmentation
dataset/train_transform=augmentation

# train unfreeze
python train.py exp=clip_reg_unfrozen computer.num_workers=20 class_name=country use_contrast_loss=True experiment_name=regression_unfrozen_contrast_country
python train.py exp=clip_reg_unfrozen computer.num_workers=20 class_name=city use_contrast_loss=True experiment_name=regression_unfrozen_contrast_city
python train.py exp=clip_reg_unfrozen computer.num_workers=20 experiment_name=reg_unfrozen
```

### Dataset structure location
You can specify the location of your dataset folder through
The default location is at `datasets` in the current directory.
In the folder, the dataset should have the following structure:

dataset_name:
- train
    - [train.csv](https://drive.google.com/file/d/1UyBPIH3pNjIiYkfb08-Mn3BUYlso_UFC/view?usp=drive_link)
    - images/
- test
    - [test.csv](https://drive.google.com/file/d/11z0Go4JEW9_yEzXbyFdFmfnQcRnbAh1P/view?usp=sharing)
    - images/

For `OpenWorld`, `dataset_name = OpenWorld`

Additional files to add in the dataset folder for oracles (used in `exp=clip_cls`):
- [index_to_gps_country.pt](https://drive.google.com/file/d/1BUx2YO6iU2cumyJ4Mw4QiDIxWsnt6w0o/view)
- [index_to_gps_city.pt](https://drive.google.com/file/d/13yC3Kf1MEMsy88GVzGkK-U0VHBB76rUL/view)

### Testing
First download the baselines from here [baselines](https://www.dropbox.com/scl/fi/f40uo0t2n83qp7dte2ytv/baselines.zip?rlkey=8kjuzqaptkeeg62hy82adfcy8&dl=0).

```
python test.py +test_dir=configs/dataset/baselines --config-dir <MODEL_PATH> --config-name config.yaml
```


### Contributing Rules
Run `pre-commit` before committing your code:
```bash
pre-commit run --all-files
```

It will ensure that everyone has the same formatting avoiding bugs and long PR because of different autoformatting.

Create a branch for independent features. We don't want to end-up with huge PRs hard to understand.
> **Warning**
> Try to think: One feature = one branch = one PR.
