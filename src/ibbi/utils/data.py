# src/ibbi/utils/data.py

"""
Utilities for dataset handling.
"""

import zipfile
from pathlib import Path
from typing import Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

# Import the cache utility to manage download locations
from .cache import get_cache_dir


def get_dataset(
    repo_id: str = "IBBI-bio/ibbi_test_data",
    local_dir: str = "ibbi_test_data",
    split: str = "train",
    **kwargs,
) -> Dataset:
    """
    Downloads and loads a dataset from the Hugging Face Hub into a specified local directory.

    This function ensures the dataset is stored in a folder with a clean name.
    If the dataset already exists locally, the download is skipped.

    Args:
        repo_id (str): The Hugging Face Hub repository ID of the dataset.
                         Defaults to "IBBI-bio/ibbi_test_data".
        local_dir (str): The desired local directory name to store the dataset.
                         Defaults to "ibbi_test_data".
        split (str): The dataset split to use (e.g., "train", "test").
                         Defaults to "train".
        **kwargs: Additional keyword arguments passed directly to
                  `datasets.load_dataset`.

    Returns:
        Dataset: The loaded dataset object from the Hugging Face Hub.

    Raises:
        TypeError: If the loaded object is not of type `Dataset`.
    """
    dataset_path = Path(local_dir)

    if not dataset_path.exists():
        print(f"Dataset not found locally. Downloading from '{repo_id}' to '{dataset_path}'...")
        snapshot_download(
            repo_id=repo_id, repo_type="dataset", local_dir=str(dataset_path), local_dir_use_symlinks=False
        )
        print("Download complete.")
    else:
        print(f"Found cached dataset at '{dataset_path}'. Loading from disk.")

    try:
        dataset: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict] = load_dataset(
            str(dataset_path), split=split, trust_remote_code=True, **kwargs
        )

        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"Expected a 'Dataset' object for split '{split}', but received type '{type(dataset).__name__}'."
            )

        print("Dataset loaded successfully.")
        return dataset
    except Exception as e:
        print(f"Failed to load dataset from '{dataset_path}'. Please check the path and your connection.")
        raise e


def get_shap_background_dataset(image_size: tuple[int, int] = (224, 224)) -> list[dict]:
    """
    Downloads, unzips, and loads the default IBBI SHAP background dataset.

    This function fetches a specific .zip file of images, not a standard
    `datasets` object. The data is downloaded and stored in the package's
    central cache directory for subsequent runs. If the data is already
    unzipped in the cache, it will be loaded directly without re-downloading.

    Args:
        image_size (tuple[int, int], optional): The target size (width, height) to resize
                                                the images to. Defaults to (224, 224).

    Returns:
        A list of dictionaries, where each dict has an "image" key with a resized PIL Image object.
    """
    repo_id = "IBBI-bio/ibbi_shap_dataset"
    filename = "ibbi_shap_dataset.zip"
    cache_dir = get_cache_dir()
    unzip_dir = cache_dir / "unzipped_shap_data"
    image_dir = unzip_dir / "shap_dataset" / "images" / "train"

    if not image_dir.exists() or not any(image_dir.iterdir()):
        print(f"SHAP background data not found in cache. Downloading from '{repo_id}'...")
        downloaded_zip_path = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset", cache_dir=str(cache_dir)
        )

        print("Decompressing SHAP background dataset...")
        unzip_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(downloaded_zip_path, "r") as zip_ref:
            zip_ref.extractall(unzip_dir)
    else:
        print("Found cached SHAP background data. Loading from disk.")

    background_images = []
    print(f"Loading and resizing SHAP background images to {image_size}...")
    image_paths = list(image_dir.glob("*"))

    for img_path in image_paths:
        with Image.open(img_path) as img:
            resized_img = img.resize(image_size)
            background_images.append({"image": resized_img.copy()})

    print("SHAP background dataset loaded and resized successfully.")
    return background_images
