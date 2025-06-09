# src/ibbi/models/classification.py

"""
Beetle classification models.
"""

import torch
from ultralytics import YOLO

from ..utils.hub import download_from_hf_hub
from ._registry import register_model


class YOLOv10BeetleClassifier:
    """
    A wrapper class for the YOLOv10 beetle classifier model.

    Provides a clean interface for image classification and feature extraction.

    Attributes:
        model (YOLO): The underlying `ultralytics.YOLO` model instance.
        device (str): The compute device ('cuda' or 'cpu') the model is loaded on.
    """

    def __init__(self, model_path: str):
        """
        Initializes the YOLOv10BeetleClassifier.

        Args:
            model_path (str): The local path to the .pt model file.
        """
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")

    def predict(self, image, **kwargs):
        """
        Performs image classification on an image.

        This method takes an image and returns the predicted class probabilities.
        Accepts any arguments that the `ultralytics.YOLO.predict` method accepts.

        Args:
            image: The input image(s). Can be a path, URL, numpy array, PIL image, etc.
            **kwargs: Additional keyword arguments to pass to the underlying
                      `predict` method of the YOLO model.

        Returns:
            A list of `ultralytics.engine.results.Results` objects containing the
            predicted class probabilities.
        """
        print("Running image classification (predict)...")
        return self.model.predict(image, **kwargs)

    def extract_features(self, image, **kwargs):
        """
        Extracts deep features from the backbone of the model for an image.

        This is useful for downstream tasks like clustering or similarity search.

        Args:
            image: The input image(s).
            **kwargs: Additional keyword arguments to pass to the underlying
                      `embed` method of the YOLO model.

        Returns:
            A tensor of features if successful, otherwise None.
        """
        print("Extracting features (embed)...")
        features = self.model.embed(image, **kwargs)
        if features:
            return features[0]
        return None


@register_model
def yolov10x_bb_classify_model(pretrained: bool = False, **kwargs):
    """
    Factory function for the YOLOv10 beetle classifier.

    Instantiates a `YOLOv10BeetleClassifier` model. If `pretrained` is True,
    it downloads the official weights from the Hugging Face Hub.

    Args:
        pretrained (bool): If True, downloads pretrained weights.
                           Defaults to False.
        **kwargs: Additional arguments (not currently used).

    Returns:
        YOLOv10BeetleClassifier: An instance of the classifier class.
    """
    if pretrained:
        repo_id = "ChristopherMarais/ibbi_yolov10_c_20250608"
        filename = "model.pt"
        local_weights_path = download_from_hf_hub(repo_id=repo_id, filename=filename)
    else:
        # Note: A non-pretrained YOLOv10x model is not meaningful for this task,
        # but this path is kept for API consistency.
        local_weights_path = "yolov10x.pt"

    return YOLOv10BeetleClassifier(model_path=local_weights_path)
