# src/ibbi/__init__.py

"""
Main initialization file for the ibbi package.

This file serves as the primary entry point for the `ibbi` library. It exposes the most
important high-level functions and classes, making them directly accessible to the user
under the `ibbi` namespace. This includes the core model creation factory (`create_model`),
the main workflow classes (`Evaluator`, `Explainer`), and key utility functions for
accessing datasets and managing the cache.

The goal of this top-level `__init__.py` is to provide a clean and intuitive API,
simplifying the user experience by abstracting away the underlying module structure.
"""

import importlib.metadata
from typing import Any

# --- Get the package version dynamically ---
try:
    __version__ = importlib.metadata.version("ibbi")
except importlib.metadata.PackageNotFoundError:
    # Fallback version for when the package is not installed
    __version__ = "Package not installed"

# --- Core Functionality ---
# --- High-level classes for streamlined workflow ---
from .evaluate import Evaluator
from .explain import Explainer, plot_lime_explanation, plot_shap_explanation
from .models import ModelType
from .models._registry import model_registry
from .utils.cache import clean_cache, get_cache_dir
from .utils.data import get_dataset, get_ood_dataset, get_shap_background_dataset
from .utils.info import list_models

# --- Model Aliases for User Convenience ---
MODEL_ALIASES = {
    "beetle_detector": "yolov10x_bb_detect_model",
    "species_classifier": "yolov12x_bb_multi_class_detect_model",
    "feature_extractor": "dinov3_vitl16_lvd1689m_features_model",
    "zero_shot_detector": "grounding_dino_detect_model",
}


def create_model(model_name: str, pretrained: bool = False, **kwargs: Any) -> ModelType:
    """Creates a model from a name or a task-based alias.

    This function is the main entry point for instantiating models within the `ibbi`
    package. It uses a model registry to look up and create a model instance based on
    the provided `model_name`. Users can either specify the exact name of a model
    or use a convenient, task-based alias (e.g., "species_classifier").

    When `pretrained=True`, the function will download the model's weights from the
    Hugging Face Hub and cache them locally for future use.

    Args:
        model_name (str): The name or alias of the model to create. A list of available
                          model names and aliases can be obtained using `ibbi.list_models()`.
        pretrained (bool, optional): If True, loads pretrained weights for the model.
                                     Defaults to False.
        **kwargs (Any): Additional keyword arguments that will be passed to the underlying
                        model's factory function. This allows for advanced customization.

    Returns:
        ModelType: An instantiated model object ready for prediction or feature extraction.

    Raises:
        KeyError: If the provided `model_name` or its resolved alias is not found in the
                  model registry.
    """
    # Resolve alias if used
    if model_name in MODEL_ALIASES:
        model_name = MODEL_ALIASES[model_name]

    if model_name not in model_registry:
        available = ", ".join(model_registry.keys())
        aliases = ", ".join(MODEL_ALIASES.keys())
        raise KeyError(f"Model '{model_name}' not found. Available models: [{available}]. Available aliases: [{aliases}].")

    model_factory = model_registry[model_name]
    model = model_factory(pretrained=pretrained, **kwargs)
    return model


__all__ = [
    "Evaluator",
    "Explainer",
    "ModelType",
    "__version__",
    "clean_cache",
    "create_model",
    "get_cache_dir",
    "get_dataset",
    "get_ood_dataset",
    "get_shap_background_dataset",
    "list_models",
    "plot_lime_explanation",
    "plot_shap_explanation",
]
