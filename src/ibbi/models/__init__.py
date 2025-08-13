# src/ibbi/models/__init__.py

from typing import TypeVar

# --- Import the actual model classes ---
from .multi_class_detection import (
    RTDETRBeetleMultiClassDetector,
    YOLOBeetleMultiClassDetector,
    YOLOEBeetleMultiClassDetector,
    rtdetrx_bb_multi_class_detect_model,
    yoloe11l_seg_bb_multi_class_detect_model,
    yolov8x_bb_multi_class_detect_model,
    yolov9e_bb_multi_class_detect_model,
    yolov10x_bb_multi_class_detect_model,
    yolov11x_bb_multi_class_detect_model,
    yolov12x_bb_multi_class_detect_model,
)
from .single_class_detection import (
    RTDETRSingleClassBeetleDetector,
    YOLOESingleClassBeetleDetector,
    YOLOSingleClassBeetleDetector,
    YOLOWorldSingleClassBeetleDetector,
    rtdetrx_bb_detect_model,
    yoloe11l_seg_bb_detect_model,
    yolov8x_bb_detect_model,
    yolov9e_bb_detect_model,
    yolov10x_bb_detect_model,
    yolov11x_bb_detect_model,
    yolov12x_bb_detect_model,
    yoloworldv2_bb_detect_model,
)
from .untrained import (
    UntrainedFeatureExtractor,
    convformer_b36_features_model,
    dinov2_vitl14_lvd142m_features_model,
    eva02_base_patch14_224_mim_in22k_features_model,
)
from .zero_shot_detection import GroundingDINOModel, grounding_dino_detect_model

# --- CORRECTED: Define the ModelType TypeVar ---
# Pass the allowed types directly to the TypeVar constructor,
# instead of using a tuple with the 'bound' argument.
ModelType = TypeVar(
    "ModelType",
    YOLOSingleClassBeetleDetector,
    RTDETRSingleClassBeetleDetector,
    YOLOWorldSingleClassBeetleDetector,
    YOLOESingleClassBeetleDetector,
    YOLOBeetleMultiClassDetector,
    RTDETRBeetleMultiClassDetector,
    YOLOEBeetleMultiClassDetector,
    GroundingDINOModel,
    UntrainedFeatureExtractor,
)


__all__ = [
    "ModelType",
    # Single-Class Detection Models
    "yolov10x_bb_detect_model",
    "yolov8x_bb_detect_model",
    "yolov9e_bb_detect_model",
    "yolov11x_bb_detect_model",
    "yolov12x_bb_detect_model",
    "rtdetrx_bb_detect_model",
    "yoloworldv2_bb_detect_model",
    "yoloe11l_seg_bb_detect_model",
    # Multi-Class Detection Models
    "yolov10x_bb_multi_class_detect_model",
    "yolov8x_bb_multi_class_detect_model",
    "yolov9e_bb_multi_class_detect_model",
    "yolov11x_bb_multi_class_detect_model",
    "yolov12x_bb_multi_class_detect_model",
    "rtdetrx_bb_multi_class_detect_model",
    "yoloe11l_seg_bb_multi_class_detect_model",
    # Zero-Shot Detection Models
    "grounding_dino_detect_model",
    # untrained feature extractors
    "dinov2_vitl14_lvd142m_features_model",
    "eva02_base_patch14_224_mim_in22k_features_model",
    "convformer_b36_features_model",
]
