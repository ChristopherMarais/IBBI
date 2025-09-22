# src/ibbi/utils/hub.py

"""
This module provides utility functions for interacting with the Hugging Face Hub.
It includes helpers for downloading model files and configuration files, while
ensuring they are stored in the appropriate local cache directory managed by the
`ibbi.utils.cache` module.
"""

import json
from typing import Any

from huggingface_hub import hf_hub_download

from .cache import get_cache_dir


def download_from_hf_hub(repo_id: str, filename: str) -> str:
    """Downloads a model file from a Hugging Face Hub repository.

    This function handles the download of a specific file from a repository on the
    Hugging Face Hub. It uses the package's caching mechanism to store the file
    locally, avoiding repeated downloads.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub (e.g., "IBBI-bio/ibbi_yolov10_od").
        filename (str): The name of the file to download from the repository (e.g., "model.pt").

    Returns:
        str: The local file path to the downloaded model file.
    """
    cache_path = get_cache_dir()
    print(f"Downloading {filename} from Hugging Face hub repository '{repo_id}'...")

    # Pass the cache_dir to the download function
    local_model_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache_path))
    print("Download complete. Model cached at:", local_model_path)
    return local_model_path


def get_model_config_from_hub(repo_id: str) -> dict[str, Any]:
    """Downloads and loads the 'config.json' file from a Hugging Face Hub repository.

    This function specifically targets the `config.json` file within a given repository.
    It downloads the file, caches it, and then loads its JSON content into a Python dictionary.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.

    Returns:
        dict[str, Any]: A dictionary containing the parsed JSON configuration.
    """
    cache_path = get_cache_dir()
    print(f"Downloading config.json from Hugging Face hub repository '{repo_id}'...")

    # Pass the cache_dir to the download function
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=str(cache_path))
    print("Download complete. Config cached at:", config_path)
    with open(config_path) as f:
        config = json.load(f)
    return config
