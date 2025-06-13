# src/ibbi/models/zero_shot_detection.py

"""
Zero-shot object detection models.
"""

from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from ._registry import register_model


class GroundingDINOBeetleDetector:
    """
    A wrapper class for the GroundingDINO zero-shot object detection model.

    This class provides an interface for performing object detection using
    free-text prompts.

    Attributes:
        model (AutoModelForZeroShotObjectDetection): The underlying Transformers model.
        processor (AutoProcessor): The processor for handling image and text inputs.
        device (str): The compute device ('cuda' or 'cpu') the model is loaded on.
    """

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base"):
        """
        Initializes the GroundingDINOBeetleDetector.

        Args:
            model_id (str): The Hugging Face model ID for GroundingDINO.
        """
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"GroundingDINO model loaded on device: {self.device}")

    def predict(self, image, text_prompt: str, box_threshold: float = 0.35, text_threshold: float = 0.25):
        """
        Performs zero-shot object detection on an image given a text prompt.

        Args:
            image: The input image. Can be a path, URL, numpy array, or PIL image.
            text_prompt (str): The text prompt describing the object to detect.
            box_threshold (float): Confidence threshold for bounding box predictions.
            text_threshold (float): Confidence threshold for text-based filtering.

        Returns:
            A dictionary containing the predicted boxes, scores, and labels.
        """
        print(f"Running GroundingDINO detection for prompt: '{text_prompt}'...")

        # Image loading logic
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
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )

        # The result is a list of dictionaries, one for each image.
        # Since we process one image at a time, we take the first element.
        return results[0]


@register_model
def grounding_dino_detect_model(pretrained: bool = True, **kwargs):
    """
    Factory function for the GroundingDINO beetle detector.

    Instantiates a `GroundingDINOBeetleDetector`. The `pretrained` flag is
    included for API consistency but this model always uses pretrained weights
    from Hugging Face.

    Args:
        pretrained (bool): If True, uses the default pretrained model.
                           Defaults to True.
        **kwargs: Additional arguments, including `model_id` to specify a
                  different GroundingDINO model from Hugging Face.

    Returns:
        GroundingDINOBeetleDetector: An instance of the detector class.
    """
    if not pretrained:
        # A non-pretrained model is not useful, but we handle the flag.
        print("Warning: `pretrained=False` has no effect. GroundingDINO is always loaded from pretrained weights.")

    model_id = kwargs.get("model_id", "IDEA-Research/grounding-dino-base")
    return GroundingDINOBeetleDetector(model_id=model_id)
