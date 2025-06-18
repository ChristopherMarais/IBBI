# src/ibbi/models/detection.py

"""
Beetle detection models.
"""

import torch
from ultralytics import RTDETR, YOLO, YOLOE, YOLOWorld

from ..utils.hub import download_from_hf_hub
from ._registry import register_model

# --- Base Classes for different architectures ---


class YOLOBeetleDetector:
    """A wrapper class for YOLO beetle detection models."""

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"YOLO Model loaded on device: {self.device}")

    def predict(self, image, **kwargs):
        return self.model.predict(image, **kwargs)

    def extract_features(self, image, **kwargs):
        features = self.model.embed(image, **kwargs)
        return features[0] if features else None

    def get_classes(self) -> list[str]:
        return self.classes


class RTDETRBeetleDetector:
    """A wrapper class for RT-DETR beetle detection models."""

    def __init__(self, model_path: str):
        self.model = RTDETR(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"RT-DETR Model loaded on device: {self.device}")

    def predict(self, image, **kwargs):
        return self.model.predict(image, **kwargs)

    def extract_features(self, image, **kwargs):
        features = self.model.embed(image, **kwargs)
        return features[0] if features else None

    def get_classes(self) -> list[str]:
        return self.classes


class YOLOWorldBeetleDetector:
    """A wrapper class for YOLO-World beetle detection models."""

    def __init__(self, model_path: str):
        self.model = YOLOWorld(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"YOLO-World Model loaded on device: {self.device}")

    def predict(self, image, **kwargs):
        return self.model.predict(image, **kwargs)

    def extract_features(self, image, **kwargs):
        features = self.model.embed(image, **kwargs)
        return features[0] if features else None

    def get_classes(self) -> list[str]:
        return self.classes


class YOLOEBeetleDetector:
    """A wrapper class for YOLOE beetle detection models."""

    def __init__(self, model_path: str):
        self.model = YOLOE(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"YOLOE Model loaded on device: {self.device}")

    def predict(self, image, **kwargs):
        return self.model.predict(image, **kwargs)

    def extract_features(self, image, **kwargs):
        features = self.model.embed(image, **kwargs)
        return features[0] if features else None

    def get_classes(self) -> list[str]:
        return self.classes


# --- Factory Functions for Detection Models (No changes below) ---


@register_model
def yolov10x_bb_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov10_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov10x.pt"
    return YOLOBeetleDetector(model_path=local_weights_path)


@register_model
def yolov8x_bb_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov8_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov8x.pt"
    return YOLOBeetleDetector(model_path=local_weights_path)


@register_model
def yolov9e_bb_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov9_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov9e.pt"
    return YOLOBeetleDetector(model_path=local_weights_path)


@register_model
def yolov11x_bb_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov11_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolo11x.pt"
    return YOLOBeetleDetector(model_path=local_weights_path)


@register_model
def yolov12x_bb_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov12_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolo12x.pt"
    return YOLOBeetleDetector(model_path=local_weights_path)


@register_model
def rtdetrx_bb_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_rtdetr_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "rtdetr-x.pt"
    return RTDETRBeetleDetector(model_path=local_weights_path)


@register_model
def yoloworldv2_bb_detect_model(pretrained: bool = False, **kwargs):
    local_weights_path = "yolov8x-worldv2.pt"
    return YOLOWorldBeetleDetector(model_path=local_weights_path)


@register_model
def yoloe11l_seg_bb_detect_model(pretrained: bool = False, **kwargs):
    local_weights_path = "yoloe-11l-seg.pt"
    return YOLOEBeetleDetector(model_path=local_weights_path)
