# src/ibbi/utils/hub.py

"""
Downloads model files from the Hugging Face Hub into the ibbi cache.

Set the environment variable `IBBI_MODELS_DIR` to a folder that contains one sub-folder per repository name (for
example `$IBBI_MODELS_DIR/ibbi_yolo11x_arthropod_detector/model.pt`) to load models from disk instead, e.g. on a
machine without internet access.
"""

import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from .cache import get_cache_dir

HF_ORG = "IBBI-bio"


def download_from_hf_hub(repo_id: str, filename: str, revision: str | None = None) -> str:
    """Returns a local path to `filename` from `repo_id`, downloading it into the ibbi cache if needed.

    Args:
        repo_id (str): Repository on the Hugging Face Hub, e.g. "IBBI-bio/ibbi_yolo11x_arthropod_detector".
        filename (str): File inside the repository, e.g. "model.pt".
        revision (str | None): Branch, tag or commit. Defaults to the main branch.

    Returns:
        str: Local file path.
    """
    local_root = os.getenv("IBBI_MODELS_DIR")
    if local_root:
        p = Path(local_root) / repo_id.split("/")[-1] / filename
        if p.exists():
            return str(p)
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, cache_dir=str(get_cache_dir()))


def get_model_config_from_hub(repo_id: str, revision: str | None = None) -> dict[str, Any]:
    """Downloads and parses `config.json` of a model repository."""
    with open(download_from_hf_hub(repo_id, "config.json", revision=revision)) as f:
        return json.load(f)
