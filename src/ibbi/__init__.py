# src/ibbi/__init__.py

"""
Main initialization file for the ibbi package.

This file exposes the primary user-facing function, `create_model`, which acts as a
factory for instantiating various beetle detection and classification models.
"""

from typing import Any, Union

# --- IMPORTANT ---
# Import the model definition files first.
# This ensures that the @register_model decorator runs and populates
# the model_registry before we try to use it.
from .models import classification, detection  # noqa: F401

# Now, import the registry that has been populated.
from .models._registry import model_registry
from .models.classification import YOLOv10BeetleClassifier
from .models.detection import YOLOv10BeetleDetector

# Define a type hint for the models that can be returned
ModelType = Union[YOLOv10BeetleDetector, YOLOv10BeetleClassifier]


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
        ModelType: An instance of the requested model (e.g., YOLOv10BeetleDetector or
                   YOLOv10BeetleClassifier).

    Raises:
        KeyError: If the requested `model_name` is not found in the model registry.

    Example:
        ```python
        import ibbi

        # Create a pretrained detection model
        detector = ibbi.create_model("yolov10x_bb_detect_model", pretrained=True)

        # Create a pretrained classification model
        classifier = ibbi.create_model("yolov10x_bb_classify_model", pretrained=True)
        ```
    """
    if model_name not in model_registry:
        available = ", ".join(model_registry.keys())
        raise KeyError(f"Model '{model_name}' not found. Available models: [{available}]")

    # Look up the factory function in the registry and call it
    model_factory = model_registry[model_name]
    model = model_factory(pretrained=pretrained, **kwargs)

    return model
