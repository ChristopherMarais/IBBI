# src/ibbi/utils/data.py

"""
Utilities for data loading and management.
"""

from importlib import resources
from typing import Union

import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def get_species_table() -> pd.DataFrame:
    """
    Returns a pandas DataFrame of the IBBI species table.

    This table contains information about the beetle species included in the
    classification dataset. It reads the data from the CSV file bundled
    with the package.

    Returns:
        pd.DataFrame: A DataFrame with species information.
    """
    try:
        # Use importlib.resources to safely access package data files.
        # The path is relative to the root of the 'ibbi' package.
        with resources.files("ibbi").joinpath("data/ibbi_species_table.csv").open("r") as f:
            return pd.read_csv(f)
    except (FileNotFoundError, ModuleNotFoundError):
        # Provide a helpful error message if the file can't be found.
        print("Error: Could not find the species table data file. " "Please ensure the package is installed correctly.")
        # Return an empty DataFrame as a fallback.
        return pd.DataFrame()


def download_dataset(
    repo_id: str,
) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
    """
    Downloads a dataset from the Hugging Face Hub.

    This function can handle repositories containing either a single data split
    (which returns a `Dataset`) or multiple splits (which returns a `DatasetDict`).
    It also correctly types the return for streaming (iterable) datasets.

    Args:
        repo_id (str): The repository ID of the dataset on the Hub.

    Returns:
        Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]: The
        downloaded dataset object.
    """
    print(f"Downloading dataset from Hugging Face Hub: {repo_id}")
    dataset = load_dataset(repo_id)
    return dataset
