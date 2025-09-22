# src/ibbi/models/__init__.py

"""Initializes the models subpackage for the ibbi library.

This file serves as the public API for the `ibbi.models` module. It imports all
model-related classes and factory functions from their respective modules
(e.g., `single_class`, `multi_class`, `zero_shot`, `feature_extractors`),
making them directly accessible under the `ibbi.models` namespace.

It also defines a generic `ModelType` for type hinting purposes and explicitly
lists all public symbols in `__all__` to ensure a clean and well-defined API.
"""

from typing import TypeVar

# --- Import all model classes and factory functions to populate the registry ---
from .feature_extractors import (
    HuggingFaceFeatureExtractor,
    UntrainedFeatureExtractor,
    convformer_b36_features_model,
    dinov2_vitl14_lvd142m_features_model,
    dinov3_vitl16_lvd1689m_features_model,
    eva02_base_patch14_224_mim_in22k_features_model,
)
from .multi_class import (
    RTDETRBeetleMultiClassDetector,
    YOLOBeetleMultiClassDetector,
    rtdetrx_bb_multi_class_detect_model,
    yolov8x_bb_multi_class_detect_model,
    yolov9e_bb_multi_class_detect_model,
    yolov10x_bb_multi_class_detect_model,
    yolov11x_bb_multi_class_detect_model,
    yolov12x_bb_multi_class_detect_model,
)
from .single_class import (
    RTDETRSingleClassBeetleDetector,
    YOLOSingleClassBeetleDetector,
    rtdetrx_bb_detect_model,
    yolov8x_bb_detect_model,
    yolov9e_bb_detect_model,
    yolov10x_bb_detect_model,
    yolov11x_bb_detect_model,
    yolov12x_bb_detect_model,
)
from .zero_shot import (
    GroundingDINOModel,
    YOLOWorldModel,
    grounding_dino_detect_model,
    yoloworldv2_bb_detect_model,
)

# --- Define a Generic ModelType for type hinting ---
ModelType = TypeVar(
    "ModelType",
    YOLOSingleClassBeetleDetector,
    RTDETRSingleClassBeetleDetector,
    YOLOBeetleMultiClassDetector,
    RTDETRBeetleMultiClassDetector,
    GroundingDINOModel,
    YOLOWorldModel,
    UntrainedFeatureExtractor,
    HuggingFaceFeatureExtractor,
)
"""A generic TypeVar for representing any of the model wrapper classes in the ibbi package.

This is used for type hinting in functions and methods that can accept or return any
of the available model types, providing flexibility while maintaining static type safety.
"""

# --- Explicitly define the public API of this module ---
__all__ = [
    "GroundingDINOModel",
    "HuggingFaceFeatureExtractor",
    "ModelType",
    "RTDETRBeetleMultiClassDetector",
    "RTDETRSingleClassBeetleDetector",
    "UntrainedFeatureExtractor",
    "YOLOBeetleMultiClassDetector",
    "YOLOSingleClassBeetleDetector",
    "YOLOWorldModel",
    "convformer_b36_features_model",
    "dinov2_vitl14_lvd142m_features_model",
    "dinov3_vitl16_lvd1689m_features_model",
    "eva02_base_patch14_224_mim_in22k_features_model",
    "grounding_dino_detect_model",
    "rtdetrx_bb_detect_model",
    "rtdetrx_bb_multi_class_detect_model",
    "yolov8x_bb_detect_model",
    "yolov8x_bb_multi_class_detect_model",
    "yolov9e_bb_detect_model",
    "yolov9e_bb_multi_class_detect_model",
    "yolov10x_bb_detect_model",
    "yolov10x_bb_multi_class_detect_model",
    "yolov11x_bb_detect_model",
    "yolov11x_bb_multi_class_detect_model",
    "yolov12x_bb_detect_model",
    "yolov12x_bb_multi_class_detect_model",
    "yoloworldv2_bb_detect_model",
]
