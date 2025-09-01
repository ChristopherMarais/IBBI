# src/ibbi/__init__.py

"""
Main initialization file for the ibbi package.
"""

from typing import Any

from .evaluate.features import EmbeddingEvaluator
from .evaluate.performance import classification_performance, object_detection_performance

# Import model modules to ensure registry is populated
# --- Import ModelType from its new location ---
from .models import (
    ModelType,
    multi_class_detection,  # noqa: F401
    single_class_detection,  # noqa: F401
    untrained,  # noqa: F401
    zero_shot_detection,  # noqa: F401
)

# Import the populated registry
from .models._registry import model_registry
from .utils.cache import clean_cache, get_cache_dir
from .utils.data import get_dataset, get_shap_background_dataset

# --- Top-level function imports ---
from .utils.info import list_models
from .xai.lime import explain_with_lime, plot_lime_explanation
from .xai.shap import explain_with_shap, plot_shap_explanation


def create_model(model_name: str, pretrained: bool = False, **kwargs: Any) -> ModelType:
    """
    Creates a model from a name.

    This factory function is the main entry point for users of the package.
    It looks up the requested model in the registry, downloads pretrained
    weights from the Hugging Face Hub if requested, and returns an
    instantiated model object.

    Args:
        model_name (str): Name of the model to create.
        pretrained (bool): Whether to load pretrained weights from the Hugging Face Hub.
                            Defaults to False.
        **kwargs (Any): Extra arguments to pass to the model-creating function.

    Returns:
        ModelType: An instance of the requested model (e.g., YOLOSingleClassBeetleDetector or
                   YOLOBeetleMultiClassDetector).

    Raises:
        KeyError: If the requested `model_name` is not found in the model registry.

    Example:
        ```python
        import ibbi

        # Create a pretrained single-class detection model
        detector = ibbi.create_model("yolov10x_bb_detect_model", pretrained=True)

        # Create a pretrained multi-class detection model
        multi_class_detector = ibbi.create_model("yolov10x_bb_multi_class_detect_model", pretrained=True)
        ```
    """
    if model_name not in model_registry:
        available = ", ".join(model_registry.keys())
        raise KeyError(f"Model '{model_name}' not found. Available models: [{available}]")

    # Look up the factory function in the registry and call it
    model_factory = model_registry[model_name]
    model = model_factory(pretrained=pretrained, **kwargs)

    return model


__all__ = [
    "EmbeddingEvaluator",
    "ModelType",
    "classification_performance",
    "clean_cache",
    "create_model",
    "explain_with_lime",
    "explain_with_shap",
    "get_cache_dir",
    "get_dataset",
    "get_shap_background_dataset",
    "list_models",
    "object_detection_performance",
    "plot_lime_explanation",
    "plot_shap_explanation",
]
