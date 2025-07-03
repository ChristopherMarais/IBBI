# src/ibbi/models/multi_class_detection.py

"""
Multi-class beetle object detection models.
"""

from typing import Any, List

import torch
import torch.nn as nn
from ultralytics import RTDETR, YOLO, YOLOE
from ultralytics.nn.tasks import DetectionModel

from ..utils.hub import download_from_hf_hub
from ._registry import register_model


class YOLOBeetleMultiClassDetector:
    """A wrapper class for YOLO multi-class beetle detector models."""

    model: YOLO
    device: str
    classes: List[str]
    feature_maps: List[torch.Tensor]
    ckpt_path: str

    def __init__(self, model_path: str, device: str | None = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = YOLO(model_path).to(self.device)

        self.classes = list(self.model.names.values())
        self.feature_maps = []

        self.ckpt_path = model_path
        if hasattr(self.model, "model"):
            internal_model = self.model.model
            assert isinstance(internal_model, DetectionModel)
            internal_model.ckpt_path = model_path  # type: ignore

        print(f"YOLO Multi-Class Detector Model loaded on device: {self.device}")

    def to(self, device: str) -> "YOLOBeetleMultiClassDetector":
        self.device = device
        self.model.to(device)
        return self

    def predict(self, image: Any, **kwargs: Any) -> Any:
        """High-level prediction for detection/cropping."""
        return self.model.predict(image, **kwargs)

    # ------------------------------------------------------------
    #                       Feature-extraction helpers
    # ------------------------------------------------------------

    def _get_head_module(self) -> torch.nn.Module:
        """
        Ultralytics YOLO-v8/v9/v10 models are nested as
        `YOLO -> DetectionModel -> nn.Sequential -> ...` with the detection
        head at the very end of the innermost Sequential. Keeping this
        in one place makes the rest of the class easier to maintain.
        """
        internal_model = self.model.model
        assert isinstance(internal_model, DetectionModel)
        assert isinstance(internal_model.model, nn.Sequential)
        return internal_model.model[-1]

    def _hook_factory(self) -> Any:
        """Creates a hook to capture the feature maps from the neck."""

        def hook(module: Any, input_data: Any) -> None:
            if input_data and isinstance(input_data, tuple) and len(input_data) > 0:
                self.feature_maps = [fm.clone().detach() for fm in input_data[0]]

        return hook

    def extract_raw_features(self, image_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Runs a single forward pass and uses a hook to capture the raw, spatial
        feature maps from the neck of the network. This is the most robust method.
        """
        head_module = self._get_head_module()
        hook_handle = head_module.register_forward_pre_hook(self._hook_factory())
        with torch.no_grad():
            self.model(image_tensor)
        hook_handle.remove()

        if not self.feature_maps:
            raise RuntimeError(
                "The neck-feature hook did not fire - verify that the head "
                "module path is still correct for this Ultralytics release."
            )

        return self.feature_maps

    def predict_from_features(self, features: List[torch.Tensor]) -> Any:
        """Gets predictions directly from the detection head."""
        detection_head = self._get_head_module()
        return detection_head(features)

    def get_classes(self) -> List[str]:
        return self.classes


class RTDETRBeetleMultiClassDetector:
    """A wrapper class for RT-DETR multi-class beetle detector models."""

    model: RTDETR
    device: str
    classes: List[str]
    feature_maps: List[torch.Tensor]

    def __init__(self, model_path: str, device: str | None = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = RTDETR(model_path).to(self.device)
        self.classes = list(self.model.names.values())
        self.feature_maps = []
        print(f"RT-DETR Multi-Class Detector Model loaded on device: {self.device}")

    def to(self, device: str) -> "RTDETRBeetleMultiClassDetector":
        self.device = device
        self.model.to(device)
        return self

    def predict(self, image: Any, **kwargs: Any) -> Any:
        return self.model.predict(image, **kwargs)

    def extract_features(self, image: Any, **kwargs: Any) -> Any:
        kwargs["stream"] = False
        return self.model.embed(image, **kwargs)

    def get_classes(self) -> List[str]:
        return self.classes


class YOLOEBeetleMultiClassDetector:
    """A wrapper class for YOLOE multi-class beetle detector models."""

    model: YOLOE
    device: str
    classes: List[str]
    feature_maps: List[torch.Tensor]

    def __init__(self, model_path: str, device: str | None = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = YOLOE(model_path).to(self.device)
        self.classes = list(self.model.names.values())
        self.feature_maps = []
        print(f"YOLOE Multi-Class Detector Model loaded on device: {self.device}")

    def to(self, device: str) -> "YOLOEBeetleMultiClassDetector":
        self.device = device
        self.model.to(device)
        return self

    def predict(self, image: Any, **kwargs: Any) -> Any:
        return self.model.predict(image, **kwargs)

    def extract_features(self, image: Any, **kwargs: Any) -> Any:
        kwargs["stream"] = False
        return self.model.embed(image, **kwargs)

    def get_classes(self) -> List[str]:
        return self.classes


# --- Factory Functions for Multi-Class Detection Models ---


@register_model
def yolov10x_bb_multi_class_detect_model(pretrained: bool = False, **kwargs: Any) -> YOLOBeetleMultiClassDetector:
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov10_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov10x.pt"
    return YOLOBeetleMultiClassDetector(model_path=local_weights_path, device=kwargs.get("device"))


@register_model
def yolov8x_bb_multi_class_detect_model(pretrained: bool = False, **kwargs: Any) -> YOLOBeetleMultiClassDetector:
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov8_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov8x.pt"
    return YOLOBeetleMultiClassDetector(model_path=local_weights_path, device=kwargs.get("device"))


@register_model
def yolov9e_bb_multi_class_detect_model(pretrained: bool = False, **kwargs: Any) -> YOLOBeetleMultiClassDetector:
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov9_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov9e.pt"
    return YOLOBeetleMultiClassDetector(model_path=local_weights_path, device=kwargs.get("device"))


@register_model
def yolov11x_bb_multi_class_detect_model(pretrained: bool = False, **kwargs: Any) -> YOLOBeetleMultiClassDetector:
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov11_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolo11x.pt"
    return YOLOBeetleMultiClassDetector(model_path=local_weights_path, device=kwargs.get("device"))


@register_model
def yolov12x_bb_multi_class_detect_model(pretrained: bool = False, **kwargs: Any) -> YOLOBeetleMultiClassDetector:
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov12_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolo12x.pt"
    return YOLOBeetleMultiClassDetector(model_path=local_weights_path, device=kwargs.get("device"))


@register_model
def rtdetrx_bb_multi_class_detect_model(pretrained: bool = False, **kwargs: Any) -> RTDETRBeetleMultiClassDetector:
    if pretrained:
        repo_id = "IBBI-bio/ibbi_rtdetr_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "rtdetr-x.pt"
    return RTDETRBeetleMultiClassDetector(model_path=local_weights_path, device=kwargs.get("device"))


@register_model
def yoloe11l_seg_bb_multi_class_detect_model(pretrained: bool = False, **kwargs: Any) -> YOLOEBeetleMultiClassDetector:
    local_weights_path = "yoloe-11l-seg.pt"
    return YOLOEBeetleMultiClassDetector(model_path=local_weights_path, device=kwargs.get("device"))
