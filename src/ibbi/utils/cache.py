# src/ibbi/utils/cache.py

"""This module provides utility functions for managing the local cache directory.

The `ibbi` package downloads and stores model weights and datasets locally to avoid
repeated downloads. This module contains functions to determine the appropriate
cache location and to clear the cache if needed.

The cache directory is determined in the following order of priority:
1. The path specified by the `IBBI_CACHE_DIR` environment variable.
2. The default user cache directory at `~/.cache/ibbi`.
"""

import os
import shutil
from pathlib import Path


def get_cache_dir() -> Path:
    """Gets the cache directory for the ibbi package.

    This function determines the appropriate directory for storing cached files,
    such as downloaded model weights and datasets. It first checks for a custom path
    set by the `IBBI_CACHE_DIR` environment variable. If the variable is not set,
    it defaults to a standard user cache location (`~/.cache/ibbi`).

    The function also ensures that the cache directory exists by creating it if it
    does not already.

    Returns:
        Path: A `pathlib.Path` object representing the path to the cache directory.
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
    """Removes the entire ibbi cache directory.

    This function will permanently delete all downloaded models and datasets
    associated with the `ibbi` package's cache. This can be useful for forcing
    a fresh download of all assets or for freeing up disk space.
    """
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        print(f"Removing cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
        print("Cache cleaned successfully.")
    else:
        print("Cache directory not found. Nothing to clean.")
