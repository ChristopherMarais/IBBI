# src/ibbi/models/single_class_detection.py

"""
Single-class beetle object detection models.
"""

import torch
import torch.nn as nn
from ultralytics import RTDETR, YOLO, YOLOE, YOLOWorld

# FIX: Import DetectionModel to assist the type checker
from ultralytics.nn.tasks import DetectionModel

from ..utils.hub import download_from_hf_hub
from ._registry import register_model

# --- Base Classes for different architectures ---


class YOLOSingleClassBeetleDetector:
    """A wrapper class for single-class YOLO beetle detection models."""

    model: YOLO
    device: str

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"YOLO Model loaded on device: {self.device}")

    def to(self, device):
        """Moves the internal model to the specified device."""
        self.device = device
        self.model.to(device)
        return self

    def predict(self, image, **kwargs):
        return self.model.predict(image, **kwargs)

    def extract_features(self, image, **kwargs):
        """Extracts feature embeddings. Returns a list of tensors for a batch of images."""
        kwargs["stream"] = False
        return self.model.embed(image, **kwargs)

    def predict_from_features(self, features, **kwargs):
        """Gets classification logits from feature embeddings. Required for TCAV."""
        # FIX: Assert the type of the internal model to guide pyright
        internal_model_module = self.model.model
        assert isinstance(internal_model_module, DetectionModel)
        if hasattr(internal_model_module, "model") and isinstance(internal_model_module.model, nn.Sequential):
            detection_head = internal_model_module.model[-1]
            return detection_head(features)
        raise NotImplementedError("This YOLO model architecture does not have a standard indexable final layer.")

    def get_classes(self) -> list[str]:
        return self.classes


class RTDETRSingleClassBeetleDetector:
    """A wrapper class for single-class RT-DETR beetle detection models."""

    model: RTDETR
    device: str

    def __init__(self, model_path: str):
        self.model = RTDETR(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"RT-DETR Model loaded on device: {self.device}")

    def to(self, device):
        """Moves the internal model to the specified device."""
        self.device = device
        self.model.to(device)
        return self

    def predict(self, image, **kwargs):
        return self.model.predict(image, **kwargs)

    def extract_features(self, image, **kwargs):
        """Extracts feature embeddings. Returns a list of tensors for a batch of images."""
        kwargs["stream"] = False
        return self.model.embed(image, **kwargs)

    def predict_from_features(self, features, **kwargs):
        """Gets classification logits from feature embeddings. Required for TCAV."""
        internal_model_module = self.model.model
        assert isinstance(internal_model_module, DetectionModel)
        if hasattr(internal_model_module, "head"):
            # Add this assertion to confirm the head is a function
            assert callable(internal_model_module.head)
            return internal_model_module.head(features)
        raise NotImplementedError("This RT-DETR model architecture does not have a standard '.head' attribute.")

    def get_classes(self) -> list[str]:
        return self.classes


class YOLOWorldSingleClassBeetleDetector:
    """A wrapper class for single-class YOLO-World beetle detection models."""

    model: YOLOWorld
    device: str

    def __init__(self, model_path: str):
        self.model = YOLOWorld(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"YOLO-World Model loaded on device: {self.device}")

    def to(self, device):
        """Moves the internal model to the specified device."""
        self.device = device
        self.model.to(device)
        return self

    def predict(self, image, **kwargs):
        return self.model.predict(image, **kwargs)

    def extract_features(self, image, **kwargs):
        """Extracts feature embeddings. Returns a list of tensors for a batch of images."""
        kwargs["stream"] = False
        return self.model.embed(image, **kwargs)

    def predict_from_features(self, features, **kwargs):
        """Gets classification logits from feature embeddings. Required for TCAV."""
        # FIX: Assert the type of the internal model to guide pyright
        internal_model_module = self.model.model
        assert isinstance(internal_model_module, DetectionModel)
        if hasattr(internal_model_module, "model") and isinstance(internal_model_module.model, nn.Sequential):
            detection_head = internal_model_module.model[-1]
            return detection_head(features)
        raise NotImplementedError("This YOLO-World model architecture does not have a standard indexable final layer.")

    def get_classes(self) -> list[str]:
        return self.classes


class YOLOESingleClassBeetleDetector:
    """A wrapper class for single-class YOLOE beetle detection models."""

    model: YOLOE
    device: str

    def __init__(self, model_path: str):
        self.model = YOLOE(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"YOLOE Model loaded on device: {self.device}")

    def to(self, device):
        """Moves the internal model to the specified device."""
        self.device = device
        self.model.to(device)
        return self

    def predict(self, image, **kwargs):
        return self.model.predict(image, **kwargs)

    def extract_features(self, image, **kwargs):
        """Extracts feature embeddings. Returns a list of tensors for a batch of images."""
        kwargs["stream"] = False
        return self.model.embed(image, **kwargs)

    def predict_from_features(self, features, **kwargs):
        """Gets classification logits from feature embeddings. Required for TCAV."""
        # FIX: Assert the type of the internal model to guide pyright
        internal_model_module = self.model.model
        assert isinstance(internal_model_module, DetectionModel)
        if hasattr(internal_model_module, "model") and isinstance(internal_model_module.model, nn.Sequential):
            detection_head = internal_model_module.model[-1]
            return detection_head(features)
        raise NotImplementedError("This YOLOE model architecture does not have a standard indexable final layer.")

    def get_classes(self) -> list[str]:
        return self.classes


@register_model
def yolov10x_bb_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov10_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov10x.pt"
    return YOLOSingleClassBeetleDetector(model_path=local_weights_path)


@register_model
def yolov8x_bb_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov8_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov8x.pt"
    return YOLOSingleClassBeetleDetector(model_path=local_weights_path)


@register_model
def yolov9e_bb_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov9_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov9e.pt"
    return YOLOSingleClassBeetleDetector(model_path=local_weights_path)


@register_model
def yolov11x_bb_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov11_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolo11x.pt"
    return YOLOSingleClassBeetleDetector(model_path=local_weights_path)


@register_model
def yolov12x_bb_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov12_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolo12x.pt"
    return YOLOSingleClassBeetleDetector(model_path=local_weights_path)


@register_model
def rtdetrx_bb_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_rtdetr_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "rtdetr-x.pt"
    return RTDETRSingleClassBeetleDetector(model_path=local_weights_path)


@register_model
def yoloworldv2_bb_detect_model(pretrained: bool = False, **kwargs):
    local_weights_path = "yolov8x-worldv2.pt"
    return YOLOWorldSingleClassBeetleDetector(model_path=local_weights_path)


@register_model
def yoloe11l_seg_bb_detect_model(pretrained: bool = False, **kwargs):
    local_weights_path = "yoloe-11l-seg.pt"
    return YOLOESingleClassBeetleDetector(model_path=local_weights_path)
