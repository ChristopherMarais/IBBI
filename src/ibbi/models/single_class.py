# src/ibbi/models/single_class.py

"""
This module provides models for single-class object detection of beetles. These models
are optimized to identify the presence and location of any beetle in an image, without
classifying the species. This is particularly useful for initial screening and counting tasks.

The module includes two primary wrapper classes for different model architectures:
- `YOLOSingleClassBeetleDetector`: For models based on the YOLO (You Only Look Once) architecture.
- `RTDETRSingleClassBeetleDetector`: For models based on the RT-DETR (Real-Time Detection Transformer) architecture.

Additionally, it provides several factory functions, decorated with `@register_model`,
to easily instantiate specific, pretrained single-class detection models.
"""

import numpy as np
import torch
from ultralytics import RTDETR, YOLO

from ..utils.hub import download_from_hf_hub
from ._registry import register_model


class YOLOSingleClassBeetleDetector:
    """A wrapper class for single-class YOLO beetle detection models.

    This class provides a standardized interface for using YOLO-based models that have been
    trained for the specific task of detecting the presence of any beetle. It handles
    model loading, device placement, prediction, and feature extraction.

    Args:
        model_path (str): The local file path to the YOLO model's weights file (e.g., a '.pt' file).
    """

    def __init__(self, model_path: str):
        """Initializes the YOLOSingleClassBeetleDetector.

        Args:
            model_path (str): Path to the YOLO model weights file.
        """
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"YOLO Model loaded on device: {self.device}")

    def predict(self, image, include_full_probabilities: bool = False, **kwargs):
        """Performs single-class object detection on an image.

        Args:
            image (Union[str, np.ndarray, Image.Image]): The input image. Can be a file path,
                                                         a numpy array, or a PIL Image object.
            include_full_probabilities (bool, optional): If True, includes a 'full_results' key in the
                                                         output with detailed probabilities for each class.
                                                         Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the underlying
                      `ultralytics.YOLO.predict` method.

        Returns:
            dict: A dictionary containing the detection results.
        """
        results = self.model.predict(image, **kwargs)

        result_dict = {"scores": [], "labels": [], "boxes": []}
        if include_full_probabilities:
            result_dict["full_results"] = []
            result_dict["class_names"] = self.classes

        if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
            for box in results[0].boxes:
                confidence = box.conf.item()
                class_id = int(box.cls)
                label = self.model.names[class_id]
                bbox = box.xyxy[0].tolist()

                result_dict["scores"].append(confidence)
                result_dict["labels"].append(label)
                result_dict["boxes"].append(bbox)

                if include_full_probabilities:
                    # For single-class, we create a proxy probability distribution
                    probabilities = np.zeros(len(self.classes))
                    if self.classes:
                        probabilities[class_id] = confidence

                    result_dict["full_results"].append(
                        {
                            "predicted_class": label,
                            "predicted_class_id": class_id,
                            "confidence": confidence,
                            "class_probabilities": probabilities.tolist(),
                            "bbox": bbox,
                        }
                    )

        return result_dict

    def extract_features(self, image, **kwargs):
        """Extracts deep feature embeddings from an image.

        Args:
            image (Union[str, np.ndarray, Image.Image]): The input image.
            **kwargs: Additional keyword arguments to be passed to the underlying
                      `ultralytics.YOLO.embed` method.

        Returns:
            Optional[torch.Tensor]: A tensor containing the extracted feature embeddings,
                                    or None if no features could be extracted.
        """
        features = self.model.embed(image, **kwargs)
        return features[0] if features else None

    def get_classes(self) -> list[str]:
        """Returns the list of class names the model was trained on.

        For single-class models, this will typically be a list with one item (e.g., ['beetle']).

        Returns:
            list[str]: A list of strings, where each string is a class name.
        """
        return self.classes


class RTDETRSingleClassBeetleDetector:
    """A wrapper class for single-class RT-DETR beetle detection models.

    This class provides a standardized interface for using RT-DETR models trained for
    single-class beetle detection. It handles model loading, device placement, prediction,
    and feature extraction.

    Args:
        model_path (str): The local file path to the RT-DETR model's weights file.
    """

    def __init__(self, model_path: str):
        """Initializes the RTDETRSingleClassBeetleDetector.

        Args:
            model_path (str): Path to the RT-DETR model weights file.
        """
        self.model = RTDETR(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"RT-DETR Model loaded on device: {self.device}")

    def predict(self, image, include_full_probabilities: bool = False, **kwargs):
        """Performs single-class object detection on an image.

        Args:
            image (Union[str, np.ndarray, Image.Image]): The input image.
            **kwargs: Additional keyword arguments for the `ultralytics.RTDETR.predict` method.

        Returns:
            dict: A dictionary containing detection results with keys for 'scores',
                  'labels', and 'boxes'.
        """
        results = self.model.predict(image, **kwargs)

        result_dict = {"scores": [], "labels": [], "boxes": []}
        if include_full_probabilities:
            result_dict["full_results"] = []
            result_dict["class_names"] = self.classes

        if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
            for box in results[0].boxes:
                confidence = box.conf.item()
                class_id = int(box.cls)
                label = self.model.names[class_id]
                bbox = box.xyxy[0].tolist()

                result_dict["scores"].append(confidence)
                result_dict["labels"].append(label)
                result_dict["boxes"].append(bbox)

                if include_full_probabilities:
                    probabilities = np.zeros(len(self.classes))
                    if self.classes:
                        probabilities[class_id] = confidence

                    result_dict["full_results"].append(
                        {
                            "predicted_class": label,
                            "predicted_class_id": class_id,
                            "confidence": confidence,
                            "class_probabilities": probabilities.tolist(),
                            "bbox": bbox,
                        }
                    )

        return result_dict

    def extract_features(self, image, **kwargs):
        """Extracts deep feature embeddings from an image.

        Args:
            image (Union[str, np.ndarray, Image.Image]): The input image.
            **kwargs: Additional keyword arguments for the `ultralytics.RTDETR.embed` method.

        Returns:
            Optional[torch.Tensor]: A tensor of feature embeddings, or None.
        """
        features = self.model.embed(image, **kwargs)
        return features[0] if features else None

    def get_classes(self) -> list[str]:
        """Returns the list of class names the model was trained on.

        Returns:
            list[str]: A list of class names.
        """
        return self.classes


@register_model
def yolov10x_bb_detect_model(pretrained: bool = False, **kwargs):
    """Factory function for the YOLOv10x single-class beetle detector.

    Args:
        pretrained (bool, optional): If True, downloads pretrained weights from the Hugging Face Hub.
                                     If False, loads a local model file named 'yolov10x.pt'. Defaults to False.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        YOLOSingleClassBeetleDetector: An instance of the YOLOv10x single-class detector.
    """
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov10_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov10x.pt"
    return YOLOSingleClassBeetleDetector(model_path=local_weights_path)


@register_model
def yolov8x_bb_detect_model(pretrained: bool = False, **kwargs):
    """Factory function for the YOLOv8x single-class beetle detector.

    Args:
        pretrained (bool, optional): If True, downloads pretrained weights. Defaults to False.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        YOLOSingleClassBeetleDetector: An instance of the YOLOv8x single-class detector.
    """
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov8_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov8x.pt"
    return YOLOSingleClassBeetleDetector(model_path=local_weights_path)


@register_model
def yolov9e_bb_detect_model(pretrained: bool = False, **kwargs):
    """Factory function for the YOLOv9e single-class beetle detector.

    Args:
        pretrained (bool, optional): If True, downloads pretrained weights. Defaults to False.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        YOLOSingleClassBeetleDetector: An instance of the YOLOv9e single-class detector.
    """
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov9_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov9e.pt"
    return YOLOSingleClassBeetleDetector(model_path=local_weights_path)


@register_model
def yolov11x_bb_detect_model(pretrained: bool = False, **kwargs):
    """Factory function for the YOLOv11x single-class beetle detector.

    Args:
        pretrained (bool, optional): If True, downloads pretrained weights. Defaults to False.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        YOLOSingleClassBeetleDetector: An instance of the YOLOv11x single-class detector.
    """
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov11_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolo11x.pt"
    return YOLOSingleClassBeetleDetector(model_path=local_weights_path)


@register_model
def yolov12x_bb_detect_model(pretrained: bool = False, **kwargs):
    """Factory function for the YOLOv12x single-class beetle detector.

    Args:
        pretrained (bool, optional): If True, downloads pretrained weights. Defaults to False.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        YOLOSingleClassBeetleDetector: An instance of the YOLOv12x single-class detector.
    """
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov12_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolo12x.pt"
    return YOLOSingleClassBeetleDetector(model_path=local_weights_path)


@register_model
def rtdetrx_bb_detect_model(pretrained: bool = False, **kwargs):
    """Factory function for the RT-DETR-x single-class beetle detector.

    Args:
        pretrained (bool, optional): If True, downloads pretrained weights. Defaults to False.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        RTDETRSingleClassBeetleDetector: An instance of the RT-DETR-x single-class detector.
    """
    if pretrained:
        repo_id = "IBBI-bio/ibbi_rtdetr_od"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "rtdetr-x.pt"
    return RTDETRSingleClassBeetleDetector(model_path=local_weights_path)
