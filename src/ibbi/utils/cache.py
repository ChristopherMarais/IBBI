# src/ibbi/utils/cache.py

import os
import shutil
from pathlib import Path


def get_cache_dir() -> Path:
    """
    Gets the cache directory for the ibbi package.

    Checks for the IBBI_CACHE_DIR environment variable first. If not set,
    it defaults to ~/.cache/ibbi. Ensures the directory exists.

    Returns:
        Path: The path to the cache directory.
    """
    # Check for the custom environment variable
    cache_env_var = os.getenv("IBBI_CACHE_DIR")
    if cache_env_var:
        cache_dir = Path(cache_env_var)
    else:
        # Default to a user's home cache directory
        cache_dir = Path.home() / ".cache" / "ibbi"

    # Create the directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def clean_cache():
    """
    Removes the entire ibbi cache directory.

    This function will delete all downloaded models and datasets associated
    with the ibbi package's cache.
    """
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        print(f"Removing cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
        print("Cache cleaned successfully.")
    else:
        print("Cache directory not found. Nothing to clean.")
