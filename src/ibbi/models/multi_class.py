# src/ibbi/models/multi_class.py

"""
This module provides models for multi-class object detection of beetles. These models
are designed to not only identify the location of a beetle in an image but also to
classify its species from a predefined set of classes.

The module includes two primary wrapper classes for different model architectures:
- `YOLOBeetleMultiClassDetector`: For models based on the YOLO (You Only Look Once) architecture.
- `RTDETRBeetleMultiClassDetector`: For models based on the RT-DETR (Real-Time Detection Transformer) architecture.

Additionally, it provides several factory functions, decorated with `@register_model`,
to easily instantiate specific, pretrained multi-class detection models.
"""

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import RTDETR, YOLO

from ..utils.hub import download_from_hf_hub
from ._registry import register_model


class YOLOBeetleMultiClassDetector:
    """A wrapper class for YOLO multi-class beetle detector models.

    This class provides a standardized interface for using YOLO-based models that have been
    trained to detect and classify multiple species of beetles. It handles model loading,
    device placement, prediction, and feature extraction.

    Args:
        model_path (str): The local file path to the YOLO model's weights file (e.g., a '.pt' file).
    """

    def __init__(self, model_path: str):
        """Initializes the YOLOBeetleMultiClassDetector.

        Args:
            model_path (str): Path to the YOLO model weights file.
        """
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"YOLO Multi-Class Detector Model loaded on device: {self.device}")

        # Pre-calculate genus map
        self.genus_map = {}
        for i, class_name in enumerate(self.classes):
            genus = class_name.split("_")[0]
            if genus not in self.genus_map:
                self.genus_map[genus] = []
            self.genus_map[genus].append(i)

    def predict(self, image, include_full_probabilities: bool = True, classify_genus: bool = False, **kwargs):
        """Performs multi-class object detection on an image.

        Args:
            image (Union[str, np.ndarray, Image.Image]): The input image. Can be a file path,
                                                         a numpy array, or a PIL Image object.
            include_full_probabilities (bool, optional): If True, includes a 'full_results' key in the
                                                         output with detailed probabilities for each class.
                                                         Defaults to True.
            classify_genus (bool, optional): If True, classifies detections at the genus level by
                                             taking the max species probabilities per genus. The primary 'labels' and
                                             'scores' will reflect the top genus. Defaults to False.
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
            for i, box in enumerate(results[0].boxes):  # type: ignore
                confidence = box.conf.item()
                class_id = int(box.cls)
                species_label = self.model.names[class_id]
                bbox = box.xyxy[0].tolist()

                probabilities = None
                if hasattr(results[0], "probs") and results[0].probs is not None and results[0].probs.shape[0] > i:
                    probabilities = F.softmax(results[0].probs[i], dim=-1).cpu().numpy()  # type: ignore
                elif hasattr(box, "probs") and box.probs is not None:
                    probabilities = box.probs.cpu().numpy()
                else:
                    # Fallback: create a proxy distribution
                    probabilities = np.zeros(len(self.classes))
                    probabilities[class_id] = confidence
                    remaining_prob = 1.0 - confidence
                    other_classes_count = len(self.classes) - 1
                    if other_classes_count > 0:
                        distributed_prob = remaining_prob / other_classes_count
                        probabilities = np.full(len(self.classes), distributed_prob)
                        probabilities[class_id] = confidence

                if classify_genus:
                    if probabilities is not None:
                        genus_probabilities = {genus: np.max([probabilities[i] for i in indices]) for genus, indices in self.genus_map.items()}
                        top_genus = max(genus_probabilities, key=lambda k: genus_probabilities[k])
                        top_genus_score = genus_probabilities[top_genus]

                        result_dict["scores"].append(top_genus_score)
                        result_dict["labels"].append(top_genus)
                        result_dict["boxes"].append(bbox)
                else:
                    result_dict["scores"].append(confidence)
                    result_dict["labels"].append(species_label)
                    result_dict["boxes"].append(bbox)

                if include_full_probabilities:
                    result_dict["full_results"].append(
                        {
                            "predicted_class": species_label,
                            "predicted_class_id": class_id,
                            "confidence": confidence,
                            "class_probabilities": probabilities.tolist() if probabilities is not None else [],
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

        Returns:
            list[str]: A list of strings, where each string is a class name.
        """
        return self.classes


class RTDETRBeetleMultiClassDetector:
    """A wrapper class for RT-DETR multi-class beetle detector models.

    This class provides a standardized interface for using RT-DETR models that have been
    trained for multi-class beetle detection. It handles model loading, device placement,
    prediction, and feature extraction.

    Args:
        model_path (str): The local file path to the RT-DETR model's weights file.
    """

    def __init__(self, model_path: str):
        """Initializes the RTDETRBeetleMultiClassDetector.

        Args:
            model_path (str): Path to the RT-DETR model weights file.
        """
        self.model = RTDETR(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes = list(self.model.names.values())
        print(f"RT-DETR Multi-Class Detector Model loaded on device: {self.device}")

        # Pre-calculate genus map
        self.genus_map = {}
        for i, class_name in enumerate(self.classes):
            genus = class_name.split("_")[0]
            if genus not in self.genus_map:
                self.genus_map[genus] = []
            self.genus_map[genus].append(i)

    def predict(self, image, include_full_probabilities: bool = True, classify_genus: bool = False, **kwargs):
        """Performs multi-class object detection on an image.

        Args:
            image (Union[str, np.ndarray, Image.Image]): The input image.
            include_full_probabilities (bool, optional): If True, includes a 'full_results' key with detailed probabilities.
                                                         Defaults to True.
            classify_genus (bool, optional): If True, classifies detections at the genus level by
                                             taking the max species probabilities per genus. The primary 'labels' and
                                             'scores' will reflect the top genus. Defaults to False.
            **kwargs: Additional keyword arguments for the `ultralytics.RTDETR.predict` method.

        Returns:
            dict: A dictionary containing detection results.
        """
        results = self.model.predict(image, **kwargs)

        result_dict = {"scores": [], "labels": [], "boxes": []}
        if include_full_probabilities:
            result_dict["full_results"] = []
            result_dict["class_names"] = self.classes

        if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):  # type: ignore
                confidence = box.conf.item()
                class_id = int(box.cls)
                species_label = self.model.names[class_id]
                bbox = box.xyxy[0].tolist()

                probabilities = None
                if hasattr(results[0], "probs") and results[0].probs is not None and results[0].probs.shape[0] > i:
                    probabilities = F.softmax(results[0].probs[i], dim=-1).cpu().numpy()  # type: ignore
                elif hasattr(box, "probs") and box.probs is not None:
                    probabilities = box.probs.cpu().numpy()
                else:
                    # Fallback: create a proxy distribution
                    probabilities = np.zeros(len(self.classes))
                    probabilities[class_id] = confidence
                    remaining_prob = 1.0 - confidence
                    other_classes_count = len(self.classes) - 1
                    if other_classes_count > 0:
                        distributed_prob = remaining_prob / other_classes_count
                        probabilities = np.full(len(self.classes), distributed_prob)
                        probabilities[class_id] = confidence

                if classify_genus:
                    if probabilities is not None:
                        genus_probabilities = {genus: np.max([probabilities[i] for i in indices]) for genus, indices in self.genus_map.items()}
                        top_genus = max(genus_probabilities, key=lambda k: genus_probabilities[k])
                        top_genus_score = genus_probabilities[top_genus]

                        result_dict["scores"].append(top_genus_score)
                        result_dict["labels"].append(top_genus)
                        result_dict["boxes"].append(bbox)
                else:
                    result_dict["scores"].append(confidence)
                    result_dict["labels"].append(species_label)
                    result_dict["boxes"].append(bbox)

                if include_full_probabilities:
                    result_dict["full_results"].append(
                        {
                            "predicted_class": species_label,
                            "predicted_class_id": class_id,
                            "confidence": confidence,
                            "class_probabilities": probabilities.tolist() if probabilities is not None else [],
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
def yolov10x_bb_multi_class_detect_model(pretrained: bool = False, **kwargs):
    """Factory function for the YOLOv10x multi-class beetle detector.

    Args:
        pretrained (bool, optional): If True, downloads pretrained weights from the Hugging Face Hub.
                                     If False, loads a local model file named 'yolov10x.pt'. Defaults to False.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        YOLOBeetleMultiClassDetector: An instance of the YOLOv10x multi-class detector.
    """
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov10_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov10x.pt"
    return YOLOBeetleMultiClassDetector(model_path=local_weights_path)


@register_model
def yolov8x_bb_multi_class_detect_model(pretrained: bool = False, **kwargs):
    """Factory function for the YOLOv8x multi-class beetle detector.

    Args:
        pretrained (bool, optional): If True, downloads pretrained weights. Defaults to False.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        YOLOBeetleMultiClassDetector: An instance of the YOLOv8x multi-class detector.
    """
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov8_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov8x.pt"
    return YOLOBeetleMultiClassDetector(model_path=local_weights_path)


@register_model
def yolov9e_bb_multi_class_detect_model(pretrained: bool = False, **kwargs):
    """Factory function for the YOLOv9e multi-class beetle detector.

    Args:
        pretrained (bool, optional): If True, downloads pretrained weights. Defaults to False.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        YOLOBeetleMultiClassDetector: An instance of the YOLOv9e multi-class detector.
    """
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov9_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolov9e.pt"
    return YOLOBeetleMultiClassDetector(model_path=local_weights_path)


@register_model
def yolov11x_bb_multi_class_detect_model(pretrained: bool = False, **kwargs):
    """Factory function for the YOLOv11x multi-class beetle detector.

    Args:
        pretrained (bool, optional): If True, downloads pretrained weights. Defaults to False.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        YOLOBeetleMultiClassDetector: An instance of the YOLOv11x multi-class detector.
    """
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov11_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolo11x.pt"
    return YOLOBeetleMultiClassDetector(model_path=local_weights_path)


@register_model
def yolov12x_bb_multi_class_detect_model(pretrained: bool = False, **kwargs):
    """Factory function for the YOLOv12x multi-class beetle detector.

    Args:
        pretrained (bool, optional): If True, downloads pretrained weights. Defaults to False.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        YOLOBeetleMultiClassDetector: An instance of the YOLOv12x multi-class detector.
    """
    if pretrained:
        repo_id = "IBBI-bio/ibbi_yolov12_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "yolo12x.pt"
    return YOLOBeetleMultiClassDetector(model_path=local_weights_path)


@register_model
def rtdetrx_bb_multi_class_detect_model(pretrained: bool = False, **kwargs):
    """Factory function for the RT-DETR-x multi-class beetle detector.

    Args:
        pretrained (bool, optional): If True, downloads pretrained weights. Defaults to False.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        RTDETRBeetleMultiClassDetector: An instance of the RT-DETR-x multi-class detector.
    """
    if pretrained:
        repo_id = "IBBI-bio/ibbi_rtdetr_oc"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        local_weights_path = "rtdetr-x.pt"
    return RTDETRBeetleMultiClassDetector(model_path=local_weights_path)
