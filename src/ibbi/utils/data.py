# src/ibbi/utils/data.py

"""
Utilities for dataset handling.
"""

import zipfile
from pathlib import Path
from typing import Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

# Import the cache utility to manage download locations
from .cache import get_cache_dir


def get_dataset(
    repo_id: str = "IBBI-bio/ibbi_test_data",
    split: str = "train",
    **kwargs,
) -> Dataset:
    """
    Loads a dataset from the Hugging Face Hub.

    This function is a wrapper around `datasets.load_dataset` and returns
    the raw Dataset object, allowing for direct manipulation. By default,
    it downloads data to a local '.cache' directory where the script is run.

    Args:
        repo_id (str): The Hugging Face Hub repository ID of the dataset.
                         Defaults to "IBBI-bio/ibbi_test_data".
        split (str): The dataset split to use (e.g., "train", "test").
                         Defaults to "train".
        **kwargs: Additional keyword arguments passed directly to
                  `datasets.load_dataset`.

    Returns:
        Dataset: The loaded dataset object from the Hugging Face Hub.

    Raises:
        TypeError: If the loaded object is not of type `Dataset`.

    Example:
        ```python
        import ibbi

        # Load the default test dataset
        test_data = ibbi.get_dataset()

        # Iterate through the first 5 examples
        for i, example in enumerate(test_data):
            if i >= 5:
                break
            print(example['image'])
        ```
    """
    print(f"Loading dataset '{repo_id}' into local directory...")
    try:
        # '.cache' folder relative to the script's execution path.
        dataset: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict] = load_dataset(
            repo_id, split=split, trust_remote_code=True, cache_dir=".", **kwargs
        )

        # Ensure that the returned object is a Dataset
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"Expected a 'Dataset' object for split '{split}', but received type '{type(dataset).__name__}'."
            )

        print("Dataset loaded successfully.")
        return dataset
    except Exception as e:
        print(f"Failed to load dataset '{repo_id}'. Please check the repository ID and your connection.")
        raise e


def get_shap_background_dataset() -> list[dict]:
    """
    Downloads, unzips, and loads the default IBBI SHAP background dataset.

    This function fetches a specific .zip file of images, not a standard
    `datasets` object. The data is downloaded and stored in the package's
    central cache directory for subsequent runs.

    Returns:
        A list of dictionaries, where each dict has an "image" key with a PIL Image object.
    """
    repo_id = "IBBI-bio/ibbi_shap_dataset"
    filename = "ibbi_shap_dataset.zip"

    # Use the centralized cache directory for SHAP data
    cache_dir = get_cache_dir()

    print(f"Loading SHAP background dataset from '{repo_id}' into cache: {cache_dir}")
    downloaded_zip_path = hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset", cache_dir=str(cache_dir)
    )

    print("Decompressing SHAP background dataset...")
    background_images = []

    # This prevents re-unzipping on every run.
    unzip_dir = Path(downloaded_zip_path).parent / "unzipped_shap_data"
    unzip_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(downloaded_zip_path, "r") as zip_ref:
        # Only extract if the directory is empty to avoid re-extraction
        if not any(unzip_dir.iterdir()):
            zip_ref.extractall(unzip_dir)

    image_dir = unzip_dir / "shap_dataset" / "images" / "train"
    image_paths = list(image_dir.glob("*"))

    for img_path in image_paths:
        with Image.open(img_path) as img:
            background_images.append({"image": img.copy()})

    print("SHAP background dataset loaded successfully.")
    return background_images
