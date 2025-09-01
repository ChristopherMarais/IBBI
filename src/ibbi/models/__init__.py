# src/ibbi/models/__init__.py

from typing import TypeVar

# --- Import the actual model classes ---
from .multi_class_detection import (
    RTDETRBeetleMultiClassDetector,
    YOLOBeetleMultiClassDetector,
    rtdetrx_bb_multi_class_detect_model,
    yolov8x_bb_multi_class_detect_model,
    yolov9e_bb_multi_class_detect_model,
    yolov10x_bb_multi_class_detect_model,
    yolov11x_bb_multi_class_detect_model,
    yolov12x_bb_multi_class_detect_model,
)
from .single_class_detection import (
    RTDETRSingleClassBeetleDetector,
    YOLOSingleClassBeetleDetector,
    rtdetrx_bb_detect_model,
    yolov8x_bb_detect_model,
    yolov9e_bb_detect_model,
    yolov10x_bb_detect_model,
    yolov11x_bb_detect_model,
    yolov12x_bb_detect_model,
)
from .untrained import (
    UntrainedFeatureExtractor,
    convformer_b36_features_model,
    dinov2_vitl14_lvd142m_features_model,
    dinov3_vitl16_lvd1689m_features_model,
    eva02_base_patch14_224_mim_in22k_features_model,
)
from .zero_shot_detection import (
    GroundingDINOModel,
    YOLOWorldModel,
    grounding_dino_detect_model,
    yoloworldv2_bb_detect_model,
)

ModelType = TypeVar(
    "ModelType",
    YOLOSingleClassBeetleDetector,
    RTDETRSingleClassBeetleDetector,
    YOLOBeetleMultiClassDetector,
    RTDETRBeetleMultiClassDetector,
    GroundingDINOModel,
    YOLOWorldModel,
    UntrainedFeatureExtractor,
)


__all__ = [
    "ModelType",
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
