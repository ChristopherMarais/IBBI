# src/ibbi/datasets/__init__.py

"""
Utilities for downloading and loading datasets.
"""

from ..utils.data import get_dataset, get_shap_background_dataset

__all__ = [
    "get_dataset",
    "get_shap_background_dataset",
]
