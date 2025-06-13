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


class GroundingDINOModel:
    """
    A wrapper class for the GroundingDINO zero-shot object detection model.
    """

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base"):
        """
        Initializes the GroundingDINOBeetleDetector.
        """
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"GroundingDINO model loaded on device: {self.device}")

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
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )
        return results[0]

    def extract_features(self, image, text_prompt: str = "object"):
        """
        Extracts deep features (embeddings) from the model for an image.

        Note: For GroundingDINO, these features are conditioned on the text_prompt.
        For general-purpose clustering, use a generic prompt like "object" or "beetle".

        Args:
            image: The input image.
            text_prompt (str): A text prompt to guide feature extraction. Defaults to "object".

        Returns:
            A tensor of features if successful, otherwise None.
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

        if hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
            # The output is of shape (batch_size, num_queries, hidden_dim).
            # We average across the queries to get a single vector per image.
            image_embeds = outputs.image_embeds
            pooled_features = torch.mean(image_embeds, dim=1)
            return pooled_features
        else:
            print("Could not extract image embeddings from GroundingDINO.")
            return None


@register_model
def grounding_dino_detect_model(pretrained: bool = True, **kwargs):
    """
    Factory function for the GroundingDINO beetle detector.
    """
    if not pretrained:
        print("Warning: `pretrained=False` has no effect. GroundingDINO is always loaded from pretrained weights.")
    model_id = kwargs.get("model_id", "IDEA-Research/grounding-dino-base")
    return GroundingDINOModel(model_id=model_id)