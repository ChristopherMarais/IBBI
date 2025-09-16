# src/ibbi/models/__init__.py

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
