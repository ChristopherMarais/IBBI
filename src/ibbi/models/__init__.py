# src/ibbi/models/__init__.py

"""Model wrappers of the ibbi package. Importing this module registers every model factory in `model_registry`.

Families:
    species detectors       one-step detection + species naming (65 species), six architectures
    arthropod detectors     find any arthropod (stage 1 of the identification pipeline): YOLO11x, and the larger
                            and more accurate Co-DINO (EVA-02-L)
    zero-shot detectors     text-prompted detection with released foundation models
    hierarchical classifiers subfamily / tribe / genus / species with per-level abstention (stage 2)
"""

from typing import Union

from .classifiers import HierarchicalClassifier, bioclip2_hierarchical_classifier, dinov3_hierarchical_classifier
from .codino import CoDINODetector, codino_arthropod_detector
from .detectors import (
    ArthropodDetector,
    SpeciesDetector,
    UltralyticsDetector,
    rtdetrx_species_detector,
    yolo11x_arthropod_detector,
    yolo11x_species_detector,
    yolo12x_species_detector,
    yolov8x_species_detector,
    yolov9e_species_detector,
    yolov10x_species_detector,
)
from .zero_shot import (
    GroundingDINOModel,
    OWLv2Model,
    SAM3Model,
    YOLOWorldModel,
    ZeroShotDetector,
    grounding_dino_zero_shot_detector,
    owlv2_zero_shot_detector,
    sam3_zero_shot_detector,
    yoloworld_zero_shot_detector,
)

ModelType = Union[UltralyticsDetector, SpeciesDetector, ArthropodDetector, CoDINODetector, ZeroShotDetector, HierarchicalClassifier]
"""Any model wrapper of the ibbi package (for type hints)."""

__all__ = [
    "ArthropodDetector",
    "CoDINODetector",
    "GroundingDINOModel",
    "HierarchicalClassifier",
    "ModelType",
    "OWLv2Model",
    "SAM3Model",
    "SpeciesDetector",
    "UltralyticsDetector",
    "YOLOWorldModel",
    "ZeroShotDetector",
    "bioclip2_hierarchical_classifier",
    "codino_arthropod_detector",
    "dinov3_hierarchical_classifier",
    "grounding_dino_zero_shot_detector",
    "owlv2_zero_shot_detector",
    "rtdetrx_species_detector",
    "sam3_zero_shot_detector",
    "yolo11x_arthropod_detector",
    "yolo11x_species_detector",
    "yolo12x_species_detector",
    "yolov8x_species_detector",
    "yolov9e_species_detector",
    "yolov10x_species_detector",
    "yoloworld_zero_shot_detector",
]
