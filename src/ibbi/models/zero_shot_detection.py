"""
Zero-shot object detection models.
"""

from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import YOLOWorld

from ._registry import register_model


class GroundingDINOModel:
    """
    A wrapper class for the GroundingDINO zero-shot object detection model.
    """

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base"):
        """
        Initializes the GroundingDINOModel.
        """
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"GroundingDINO model loaded on device: {self.device}")

    def get_classes(self):
        """
        This method is not applicable to zero-shot models.
        """
        raise NotImplementedError(
            "The GroundingDINOModel is a zero-shot detection model and does not have a fixed set of classes. "
            "Classes are defined dynamically at inference time via the 'text_prompt' argument in the 'predict' method."
        )

    def predict(self, image, text_prompt: str, box_threshold: float = 0.35, text_threshold: float = 0.25):
        """
        Performs zero-shot object detection on an image given a text prompt.
        """
        print(f"Running GroundingDINO detection for prompt: '{text_prompt}'...")

        if isinstance(image, str):
            if image.startswith("http"):
                response = requests.get(image)
                image_pil = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image_pil = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image_pil = image.convert("RGB")
        else:
            raise ValueError("Unsupported image type. Use a file path, URL, numpy array, or PIL image.")

        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,  # Corrected argument name
            text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )
        return results[0]

    def extract_features(self, image, text_prompt: str = "object"):
        """
        Extracts deep features (embeddings) from the model for an image.
        """
        print(f"Extracting features from GroundingDINO using prompt: '{text_prompt}'...")

        if isinstance(image, str):
            if image.startswith("http"):
                response = requests.get(image)
                image_pil = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image_pil = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image_pil = image.convert("RGB")
        else:
            raise ValueError("Unsupported image type. Use a file path, URL, numpy array, or PIL image.")

        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        if (
            hasattr(outputs, "encoder_last_hidden_state_vision")
            and outputs.encoder_last_hidden_state_vision is not None
        ):
            vision_features = outputs.encoder_last_hidden_state_vision
            pooled_features = torch.mean(vision_features, dim=1)
            return pooled_features
        else:
            print("Could not extract 'encoder_last_hidden_state_vision' from GroundingDINO output.")
            print(f"Available attributes in 'outputs': {dir(outputs)}")
            return None


class YOLOWorldModel:
    """
    A wrapper class for the YOLOWorld zero-shot object detection model.
    """

    def __init__(self, model_path: str):
        self.model = YOLOWorld(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"YOLO-World model loaded on device: {self.device}")

    def get_classes(self):
        """
        Returns the classes the model is currently set to detect.
        """
        return list(self.model.names.values())

    def set_classes(self, classes: list[str]):
        """
        Sets the classes for the model to detect.
        """
        self.model.set_classes(classes)
        print(f"YOLOWorld classes set to: {classes}")

    def predict(self, image, **kwargs):
        """
        Performs zero-shot object detection on an image.
        """
        return self.model.predict(image, **kwargs)

    def extract_features(self, image, **kwargs):
        """
        Extracts deep features (embeddings) from the model for an image.
        """
        features = self.model.embed(image, **kwargs)
        return features[0] if features else None


@register_model
def grounding_dino_detect_model(pretrained: bool = True, **kwargs):
    """
    Factory function for the GroundingDINO beetle detector.
    """
    if not pretrained:
        print("Warning: `pretrained=False` has no effect. GroundingDINO is always loaded from pretrained weights.")
    model_id = kwargs.get("model_id", "IDEA-Research/grounding-dino-base")
    return GroundingDINOModel(model_id=model_id)


@register_model
def yoloworldv2_bb_detect_model(pretrained: bool = True, **kwargs):
    """
    Factory function for the YOLOWorld beetle detector.
    Note: `pretrained` flag is for consistency; this model always loads local weights.
    """
    local_weights_path = "yolov8x-worldv2.pt"
    return YOLOWorldModel(model_path=local_weights_path)
