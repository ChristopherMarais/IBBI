# src/ibbi/utils/data.py

"""
Utilities for dataset handling.
"""

from typing import Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset


def get_dataset(
    repo_id: str = "IBBI-bio/ibbi_test_data",
    split: str = "train",
    **kwargs,
) -> Dataset:
    """
    Loads a dataset from the Hugging Face Hub.

    This function is a wrapper around `datasets.load_dataset` and returns
    the raw Dataset object, allowing for direct manipulation.

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
    print(f"Loading dataset '{repo_id}' (split: '{split}') from Hugging Face Hub...")
    try:
        # Load the dataset from the hub
        dataset: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict] = load_dataset(
            repo_id, split=split, trust_remote_code=True, **kwargs
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
