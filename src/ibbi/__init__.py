# src/ibbi/__init__.py

"""
Main initialization file for the ibbi package.

Provides a high-level API for model creation, evaluation, and explanation.
"""

from typing import Any

# --- Core Functionality ---
from .datasets import get_dataset, get_shap_background_dataset

# --- High-level classes for streamlined workflow ---
from .evaluate import Evaluator
from .explain import Explainer, plot_lime_explanation, plot_shap_explanation
from .models import ModelType
from .models._registry import model_registry
from .utils.cache import clean_cache, get_cache_dir
from .utils.info import list_models

# --- Model Aliases for User Convenience ---
MODEL_ALIASES = {
    "beetle_detector": "yolov10x_bb_detect_model",
    "species_classifier": "yolov12x_bb_multi_class_detect_model",
    "feature_extractor": "dinov3_vitl16_lvd1689m_features_model",
    "zero_shot_detector": "grounding_dino_detect_model",
}


def create_model(model_name: str, pretrained: bool = False, **kwargs: Any) -> ModelType:
    """
    Creates a model from a name or a task-based alias.

    This is the main entry point for creating models. It can use a specific
    model name or a simplified alias like "species_classifier".

    Args:
        model_name (str): Name or alias of the model to create.
        pretrained (bool): Whether to load pretrained weights. Defaults to False.
        **kwargs (Any): Extra arguments for the model function.

    Returns:
        An instance of the requested model.
    """
    # Resolve alias if used
    if model_name in MODEL_ALIASES:
        model_name = MODEL_ALIASES[model_name]

    if model_name not in model_registry:
        available = ", ".join(model_registry.keys())
        aliases = ", ".join(MODEL_ALIASES.keys())
        raise KeyError(
            f"Model '{model_name}' not found. Available models: [{available}]. Available aliases: [{aliases}]."
        )

    model_factory = model_registry[model_name]
    model = model_factory(pretrained=pretrained, **kwargs)
    return model


__all__ = [
    "Evaluator",
    "Explainer",
    "ModelType",
    "clean_cache",
    "create_model",
    "get_cache_dir",
    "get_dataset",
    "get_shap_background_dataset",
    "list_models",
    "plot_lime_explanation",
    "plot_shap_explanation",
]
