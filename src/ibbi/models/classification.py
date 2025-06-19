# src/ibbi/models/classification.py

"""
Beetle classification models.
"""

import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.ops import batched_nms
from ultralytics import RTDETR, YOLO, YOLOE
from ultralytics.engine.results import Results
from ultralytics.utils import ops

from ..utils.hub import download_from_hf_hub
from ._registry import register_model

# --- Base Classes for different architectures ---


class YOLOBeetleClassifier:
    """A wrapper class for YOLO beetle classifier models."""

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"YOLO Classifier Model loaded on device: {self.device}")

    def predict(self, image, **kwargs):
        return self.model.predict(image, **kwargs)

    def extract_features(self, image, **kwargs):
        features = self.model.embed(image, **kwargs)
        return features[0] if features else None

    def get_classes(self) -> list[str]:
        return self.classes


class RTDETRBeetleClassifier:
    """A wrapper class for RT-DETR beetle classifier models."""

    def __init__(self, model_path: str):
        self.model = RTDETR(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"RT-DETR Classifier Model loaded on device: {self.device}")

    def predict(self, image, conf=0.25, iou=0.45, **kwargs):
        """
        Applies a full manual prediction pipeline with type checks to satisfy linters.
        """
        # 1. Manually pre-process the image
        img_tensor = F.to_tensor(image).unsqueeze(0).to(self.device).float()

        # 2. Get raw predictions
        predictor = self.model.model
        if not callable(predictor):
            raise TypeError("The underlying model predictor is not callable.")

        raw_preds_output = predictor(img_tensor, **kwargs)
        # Add checks to satisfy pyright about the output type
        if not isinstance(raw_preds_output, (list, tuple)) or len(raw_preds_output) == 0:
            return [Results(orig_img=np.array(image), path=None, names=self.model.names, boxes=None)]
        preds = raw_preds_output[0]

        # 3. Post-process raw predictions into clean detections
        nd = preds.shape[-1]
        bboxes_xywh, scores_logits = preds.split((4, nd - 4), dim=-1)
        max_scores, class_indices = scores_logits.sigmoid().max(-1, keepdim=True)
        confident_detections_mask = max_scores.squeeze(-1) > conf

        clean_detections = torch.cat([bboxes_xywh, max_scores, class_indices.float()], dim=-1)[
            confident_detections_mask
        ]

        if clean_detections.numel() == 0:
            return [Results(orig_img=np.array(image), path=None, names=self.model.names, boxes=None)]

        # 4. Apply Per-Class NMS
        boxes_for_nms = ops.xywh2xyxy(clean_detections[:, :4])
        scores_for_nms = clean_detections[:, 4]
        classes_for_nms = clean_detections[:, 5]

        # Explicitly ensure all inputs to batched_nms are tensors to satisfy pyright
        indices_to_keep = batched_nms(
            torch.as_tensor(boxes_for_nms), torch.as_tensor(scores_for_nms), torch.as_tensor(classes_for_nms), iou
        )

        final_detections = clean_detections[indices_to_keep]

        # 5. Create the final tensor in the correct format (xyxy)
        final_boxes_xyxy = ops.xywh2xyxy(final_detections[:, :4])
        final_scores = final_detections[:, 4].unsqueeze(1)
        final_classes = final_detections[:, 5].unsqueeze(1)

        # Ensure all tensors in the list are torch.Tensor for torch.cat
        final_tensor = torch.cat([torch.as_tensor(t) for t in [final_boxes_xyxy, final_scores, final_classes]], dim=1)

        # 6. Convert to a Results object
        orig_img_np = np.array(image)
        final_results = Results(orig_img=orig_img_np, path=None, names=self.model.names, boxes=final_tensor)

        return [final_results]

    def extract_features(self, image, **kwargs):
        features = self.model.embed(image, **kwargs)
        return features[0] if features else None

    def get_classes(self) -> list[str]:
        return self.classes


class YOLOEBeetleClassifier:
    """A wrapper class for YOLOE beetle classifier models."""

    def __init__(self, model_path: str):
        self.model = YOLOE(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"YOLOE Classifier Model loaded on device: {self.device}")

    def predict(self, image, **kwargs):
        return self.model.predict(image, **kwargs)

    def extract_features(self, image, **kwargs):
        features = self.model.embed(image, **kwargs)
        return features[0] if features else None

    def get_classes(self) -> list[str]:
        return self.classes


# --- Factory Functions for Classification Models (No changes below) ---


@register_model
def yolov10x_bb_classify_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov10_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov10x.pt"
    return YOLOBeetleClassifier(model_path=local_weights_path)


@register_model
def yolov8x_bb_classify_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov8_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov8x.pt"
    return YOLOBeetleClassifier(model_path=local_weights_path)


@register_model
def yolov9e_bb_classify_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov9_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov9e.pt"
    return YOLOBeetleClassifier(model_path=local_weights_path)


@register_model
def yolov11x_bb_classify_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov11_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolo11x.pt"
    return YOLOBeetleClassifier(model_path=local_weights_path)


@register_model
def yolov12x_bb_classify_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov12_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolo12x.pt"
    return YOLOBeetleClassifier(model_path=local_weights_path)


@register_model
def rtdetrx_bb_classify_model(pretrained: bool = False, **kwargs):
    if pretrained:
        repo_id = "IBBI-bio/ibbi_rtdetr_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "rtdetr-x.pt"
    return RTDETRBeetleClassifier(model_path=local_weights_path)


@register_model
def yoloe11l_seg_bb_classify_model(pretrained: bool = False, **kwargs):
    local_weights_path = "yoloe-11l-seg.pt"
    return YOLOEBeetleClassifier(model_path=local_weights_path)
