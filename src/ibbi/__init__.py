# src/ibbi/__init__.py

"""
ibbi: Intelligent Bark Beetle Identifier.

Detect, identify and evaluate bark and ambrosia beetles (Curculionidae: Scolytinae and Platypodinae) with models
trained and benchmarked on the Bark and Ambrosia Beetle Detection Benchmark.

    import ibbi
    pipe = ibbi.create_pipeline()                      # arthropod detector + hierarchical classifier
    result = pipe.predict("beetles.jpg")
    detector = ibbi.create_model("species_detector")   # one-step species detector
    data = ibbi.get_dataset("iid_test")                # benchmark split
    ibbi.Evaluator(detector).benchmark()               # crowd-aware benchmark scores
"""

import importlib.metadata
from typing import Any

try:
    __version__ = importlib.metadata.version("ibbi")
except importlib.metadata.PackageNotFoundError:
    __version__ = "Package not installed"

from .evaluate import Evaluator
from .explain import Explainer, plot_lime_explanation, plot_shap_explanation
from .models import ModelType
from .models._registry import model_registry
from .pipeline import IdentificationPipeline, create_pipeline
from .utils.cache import clean_cache, get_cache_dir
from .utils.data import download_benchmark, get_dataset, get_ood_dataset, get_shap_background_dataset, get_taxonomy
from .utils.info import list_models

# --- Task-based aliases -----------------------------------------------------------------------------------------------
MODEL_ALIASES = {
    "arthropod_detector": "yolo11x_arthropod_detector",
    "beetle_detector": "yolo11x_arthropod_detector",
    "species_detector": "yolo12x_species_detector",
    "hierarchical_classifier": "dinov3_hierarchical_classifier",
    "species_classifier": "dinov3_hierarchical_classifier",
    "feature_extractor": "dinov3_hierarchical_classifier",
    "zero_shot_detector": "grounding_dino_zero_shot_detector",
}


def create_model(model_name: str, pretrained: bool = True, **kwargs: Any) -> ModelType:
    """Creates a model from its name or a task alias.

    Args:
        model_name (str): A model name (see `ibbi.list_models()`) or an alias: "arthropod_detector" (also
            "beetle_detector"), "species_detector", "hierarchical_classifier" (also "species_classifier" and
            "feature_extractor") or "zero_shot_detector".
        pretrained (bool): Load the trained IBBI weights (default True). For detectors, False loads the generic
            Ultralytics COCO checkpoint of the same architecture.
        **kwargs: Passed to the model factory, e.g. `device="cpu"`, `revision=...`, `operating_point="0.95"` for
            classifiers, `prompts=[...]` / `tile=0` for zero-shot detectors.

    Returns:
        ModelType: The model, ready for `predict`, `predict_proba` and `extract_features`.

    Raises:
        KeyError: If the name is neither a model nor an alias.
    """
    model_name = MODEL_ALIASES.get(model_name, model_name)
    if model_name not in model_registry:
        available = ", ".join(model_registry.keys())
        aliases = ", ".join(MODEL_ALIASES.keys())
        raise KeyError(f"Model '{model_name}' not found. Available models: [{available}]. Available aliases: [{aliases}].")
    return model_registry[model_name](pretrained=pretrained, **kwargs)


__all__ = [
    "MODEL_ALIASES",
    "Evaluator",
    "Explainer",
    "IdentificationPipeline",
    "ModelType",
    "__version__",
    "clean_cache",
    "create_model",
    "create_pipeline",
    "download_benchmark",
    "get_cache_dir",
    "get_dataset",
    "get_ood_dataset",
    "get_shap_background_dataset",
    "get_taxonomy",
    "list_models",
    "plot_lime_explanation",
    "plot_shap_explanation",
]
