# src/ibbi/utils/hub.py

import json
from typing import Any

from huggingface_hub import hf_hub_download

from .cache import get_cache_dir


def download_from_hf_hub(repo_id: str, filename: str) -> str:
    """
    Downloads a model file from a Hugging Face Hub repository.
    """
    cache_path = get_cache_dir()
    print(f"Downloading {filename} from Hugging Face hub repository '{repo_id}'...")

    # Pass the cache_dir to the download function
    local_model_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache_path))
    print("Download complete. Model cached at:", local_model_path)
    return local_model_path


def get_model_config_from_hub(repo_id: str) -> dict[str, Any]:
    """
    Downloads and loads the 'config.json' file from a Hugging Face Hub repository.
    """
    cache_path = get_cache_dir()
    print(f"Downloading config.json from Hugging Face hub repository '{repo_id}'...")

    # Pass the cache_dir to the download function
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=str(cache_path))
    print("Download complete. Config cached at:", config_path)
    with open(config_path) as f:
        config = json.load(f)
    return config
