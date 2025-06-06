# src/ibbi/models/detection.py

import torch
from ultralytics import YOLO

from ..utils.hub import download_from_hf_hub
from ._registry import register_model


class YOLOv10BeetleDetector:
    """
    A wrapper class for the YOLOv10 beetle detector model.

    This class provides a clean interface for both object detection (inference)
    and feature extraction.
    """

    def __init__(self, model_path: str):
        """
        Initializes the wrapper with a loaded YOLO model.

        Args:
            model_path (str): The local path to the .pt model file.
        """
        self.model = YOLO(model_path)
        # Set the device based on availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")

    def predict(self, image, **kwargs):
        """
        Performs object detection inference on an image.
        """
        print("Running object detection (predict)...")
        return self.model.predict(image, **kwargs)

    def extract_features(self, image, **kwargs):
        """
        Extracts deep features from the backbone of the model for an image.
        """
        print("Extracting features (embed)...")
        # The `embed` method is the correct one to use from the ultralytics library
        features = self.model.embed(image, **kwargs)
        if features:
            return features[0]
        return None


@register_model
def yolov10x_beetle_detector(pretrained: bool = False, **kwargs):
    """
    Factory function for the YOLOv10 beetle detector.

    This function now returns a wrapper object that can be used for both
    inference and feature extraction.
    """
    if pretrained:
        repo_id = "ChristopherMarais/ibbi_yolov10_od_20250601"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)

    else:
        local_weights_path = "yolov10x.pt"

    # CRITICAL CHANGE: Return an instance of our new wrapper class
    return YOLOv10BeetleDetector(model_path=local_weights_path)
