# src/ibbi/models/multi_class_detection.py

"""
Multi-class beetle object detection models.
"""

import torch
from ultralytics import RTDETR, YOLO

from ..utils.hub import download_from_hf_hub
from ._registry import register_model


class YOLOBeetleMultiClassDetector:
    """A wrapper class for YOLO multi-class beetle detector models."""

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"YOLO Multi-Class Detector Model loaded on device: {self.device}")

    def predict(self, image, **kwargs):
        return self.model.predict(image, **kwargs)

    def extract_features(self, image, **kwargs):
        features = self.model.embed(image, **kwargs)
        return features[0] if features else None

    def get_classes(self) -> list[str]:
        return self.classes


class RTDETRBeetleMultiClassDetector:
    """A wrapper class for RT-DETR multi-class beetle detector models."""

    def __init__(self, model_path: str):
        self.model = RTDETR(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"RT-DETR Multi-Class Detector Model loaded on device: {self.device}")

    def predict(self, image, **kwargs):
        return self.model.predict(image, **kwargs)

    def extract_features(self, image, **kwargs):
        features = self.model.embed(image, **kwargs)
        return features[0] if features else None

    def get_classes(self) -> list[str]:
        return self.classes


@register_model
def yolov10x_bb_multi_class_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov10_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov10x.pt"
    return YOLOBeetleMultiClassDetector(model_path=local_weights_path)


@register_model
def yolov8x_bb_multi_class_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov8_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov8x.pt"
    return YOLOBeetleMultiClassDetector(model_path=local_weights_path)


@register_model
def yolov9e_bb_multi_class_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov9_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov9e.pt"
    return YOLOBeetleMultiClassDetector(model_path=local_weights_path)


@register_model
def yolov11x_bb_multi_class_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov11_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolo11x.pt"
    return YOLOBeetleMultiClassDetector(model_path=local_weights_path)


@register_model
def yolov12x_bb_multi_class_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov12_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolo12x.pt"
    return YOLOBeetleMultiClassDetector(model_path=local_weights_path)


@register_model
def rtdetrx_bb_multi_class_detect_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_rtdetr_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "rtdetr-x.pt"
    return RTDETRBeetleMultiClassDetector(model_path=local_weights_path)
