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

        Args:
            image: The input image (e.g., path, numpy array).
            **kwargs: Additional arguments to pass to the ultralytics `predict` method.

        Returns:
            The detection results from the ultralytics model.
        """
        print("Running object detection (predict)...")
        return self.model.predict(image, **kwargs)

    def extract_features(self, image, **kwargs):
        """
        Extracts deep features from the backbone of the model for an image.

        This uses the `embed` method from the ultralytics library, which typically
        returns the output of the final feature map before the detection heads.

        Args:
            image: The input image (e.g., path, numpy array).
            **kwargs: Additional arguments to pass to the ultralytics `embed` method.

        Returns:
            A tensor representing the extracted features.
        """
        print("Extracting features (embed)...")
        # The `embed` method is designed for this purpose.
        # It returns a list of feature tensors, one per image.
        # We'll return the first result.
        features = self.model.embed(image, **kwargs)
        return features[0]


@register_model
def yolov10x_beetle_detector(pretrained: bool = False, **kwargs):
    """
    Factory function for the YOLOv10 beetle detector.

    This function now returns a wrapper object that can be used for both
    inference and feature extraction.

    Args:
        pretrained (bool): If True, downloads the model from Hugging Face Hub.
                           If False, loads a base non-fine-tuned model.
        **kwargs: Not used in this factory, but kept for API consistency.

    Returns:
        An instance of the YOLOv10BeetleDetector wrapper class.
    """
    if pretrained:
        # Link to your Hugging Face model
        repo_id = "ChristopherMarais/ibbi_yolov10_od_20250601"
        filename = "model.pt"

        # Download the weights file and get its local path
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)

    else:
        # Load a base, non-fine-tuned model from Ultralytics
        # This will be downloaded by ultralytics if not cached
        local_weights_path = "yolov10x.pt"

    # Create and return an instance of our new wrapper class
    return YOLOv10BeetleDetector(model_path=local_weights_path)
