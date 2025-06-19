# src/ibbi/models/__init__.py

from .multi_class_detection import (
    rtdetrx_bb_multi_class_detect_model,
    yoloe11l_seg_bb_multi_class_detect_model,
    yolov8x_bb_multi_class_detect_model,
    yolov9e_bb_multi_class_detect_model,
    yolov10x_bb_multi_class_detect_model,
    yolov11x_bb_multi_class_detect_model,
    yolov12x_bb_multi_class_detect_model,
)
from .single_class_detection import (
    rtdetrx_bb_detect_model,
    yoloe11l_seg_bb_detect_model,
    yolov8x_bb_detect_model,
    yolov9e_bb_detect_model,
    yolov10x_bb_detect_model,
    yolov11x_bb_detect_model,
    yolov12x_bb_detect_model,
    yoloworldv2_bb_detect_model,
)
from .zero_shot_detection import grounding_dino_detect_model

__all__ = [
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
]
