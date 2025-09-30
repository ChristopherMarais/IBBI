# src/ibbi/utils/data.py

"""
This module provides utility functions for dataset handling within the `ibbi` package.
It includes functions for downloading, caching, and loading datasets from the Hugging Face Hub,
ensuring that users have easy and efficient access to the data required for model
evaluation and explainability tasks.
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
    """Downloads and loads a dataset from the Hugging Face Hub.

    This function facilitates the use of datasets hosted on the Hugging Face Hub by
    handling the download and caching process. It downloads the dataset to a local
    directory, and on subsequent calls, it will load the data directly from the local
    cache to save time and bandwidth.

    Args:
        repo_id (str, optional): The repository ID of the dataset on the Hugging Face Hub.
                                 Defaults to "IBBI-bio/ibbi_test_data".
        local_dir (str, optional): The name of the local directory where the dataset will be stored.
                                   Defaults to "ibbi_test_data".
        split (str, optional): The name of the dataset split to load (e.g., "train", "test", "validation").
                               Defaults to "train".
        **kwargs: Additional keyword arguments that will be passed directly to the
                  `datasets.load_dataset` function. This allows for advanced customization
                  of the data loading process.

    Returns:
        Dataset: The loaded dataset as a `datasets.Dataset` object.

    Raises:
        TypeError: If the object loaded for the specified split is not of type `datasets.Dataset`.
    """
    dataset_path = Path(local_dir)

    if not dataset_path.exists():
        print(f"Dataset not found locally. Downloading from '{repo_id}' to '{dataset_path}'...")
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(dataset_path))
        print("Download complete.")
    else:
        print(f"Found cached dataset at '{dataset_path}'. Loading from disk.")

    try:
        dataset: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict] = load_dataset(
            str(dataset_path), split=split, trust_remote_code=True, **kwargs
        )

        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected a 'Dataset' object for split '{split}', but received type '{type(dataset).__name__}'.")

        print("Dataset loaded successfully.")
        return dataset
    except Exception as e:
        print(f"Failed to load dataset from '{dataset_path}'. Please check the path and your connection.")
        raise e


def get_shap_background_dataset(image_size: tuple[int, int] = (224, 224)) -> list[dict]:
    """Downloads, unzips, and loads the default IBBI SHAP background dataset.

    This function is specifically designed to fetch the background dataset required for the
    SHAP (SHapley Additive exPlanations) explainability method. It handles the download of a
    zip archive from the Hugging Face Hub, extracts its contents, and loads the images into
    memory. The data is stored in the package's central cache directory to avoid re-downloads.

    Args:
        image_size (tuple[int, int], optional): The target size (width, height) to which the
                                                background images will be resized. This should
                                                match the input size expected by the model being
                                                explained. Defaults to (224, 224).

    Returns:
        list[dict]: A list of dictionaries, where each dictionary has an "image" key with a
                    resized PIL Image object. This format is ready to be used with the
                    `ibbi.Explainer.with_shap` method.
    """
    repo_id = "IBBI-bio/ibbi_shap_dataset"
    filename = "ibbi_shap_dataset.zip"
    cache_dir = get_cache_dir()
    unzip_dir = cache_dir / "unzipped_shap_data"
    image_dir = unzip_dir / "shap_dataset" / "images" / "train"

    if not image_dir.exists() or not any(image_dir.iterdir()):
        print(f"SHAP background data not found in cache. Downloading from '{repo_id}'...")
        downloaded_zip_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", cache_dir=str(cache_dir))

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


def get_ood_dataset(
    repo_id: str = "IBBI-bio/ibbi_ood_data",
    local_dir: str = "ibbi_ood_data",
    split: str = "train",
    **kwargs,
) -> Dataset:
    """Downloads and loads the out-of-distribution (OOD) dataset from the Hugging Face Hub.

    This function handles the download and caching of the OOD dataset. On subsequent
    calls, it will load the data directly from the local cache.

    Args:
        repo_id (str, optional): The repository ID of the OOD dataset on the Hugging Face Hub.
                                 Defaults to "IBBI-bio/ibbi_ood_data".
        local_dir (str, optional): The name of the local directory where the dataset will be stored.
                                   Defaults to "ibbi_ood_data".
        split (str, optional): The name of the dataset split to load. Defaults to "train".
        **kwargs: Additional keyword arguments for the `datasets.load_dataset` function.

    Returns:
        Dataset: The loaded OOD dataset as a `datasets.Dataset` object.
    """
    return get_dataset(repo_id=repo_id, local_dir=local_dir, split=split, **kwargs)
