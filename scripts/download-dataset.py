import os, zipfile
from huggingface_hub import snapshot_download

if not(os.path.exists("datasets/OpenWorld")):
    os.mkdir("datasets/OpenWorld")

snapshot_download(repo_id="osv5m/osv5m", local_dir="datasets/OpenWorld", repo_type='dataset')
for root, dirs, files in os.walk("datasets/OpenWorld"):
    for file in files:
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(root, file), 'r') as zip_ref:
                zip_ref.extractall(root)
                os.remove(os.path.join(root, file))
