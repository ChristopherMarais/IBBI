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
            box_threshold=box_threshold,
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

        # FINAL FIX: Use the correct attribute name found in the debug output.
        if (
            hasattr(outputs, "encoder_last_hidden_state_vision")
            and outputs.encoder_last_hidden_state_vision is not None
        ):
            vision_features = outputs.encoder_last_hidden_state_vision
            # Pool the features across all patches to get a single vector for the image.
            # The shape is (batch_size, num_patches, hidden_dim), so we average over dim 1.
            pooled_features = torch.mean(vision_features, dim=1)
            return pooled_features
        else:
            print("Could not extract 'encoder_last_hidden_state_vision' from GroundingDINO output.")
            print(f"Available attributes in 'outputs': {dir(outputs)}")
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
