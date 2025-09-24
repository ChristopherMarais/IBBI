# src/ibbi/models/zero_shot.py

"""
This module provides models for zero-shot object detection. These models are capable of
detecting objects in images based on arbitrary text prompts, without being explicitly
trained on a predefined set of classes. This makes them highly flexible for a wide
range of detection tasks.

The module includes two primary wrapper classes for different zero-shot architectures:
- `GroundingDINOModel`: For the GroundingDINO model, which excels at open-set object detection.
- `YOLOWorldModel`: For the YOLOWorld model, which extends the YOLO architecture with zero-shot capabilities.

Additionally, it provides factory functions, decorated with `@register_model`, to easily
instantiate these models with pretrained weights.
"""

from io import BytesIO
from typing import Optional, Union

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import YOLOWorld

from ._registry import register_model


class GroundingDINOModel:
    """A wrapper class for the GroundingDINO zero-shot object detection model.

    This class provides a standardized interface for using the GroundingDINO model for
    detecting objects in an image based on a text prompt. It handles model and processor
    loading from the Hugging Face Hub, device placement, and provides methods for both
    prediction and feature extraction.

    Args:
        model_id (str, optional): The model identifier from the Hugging Face Hub.
                                Defaults to "IDEA-Research/grounding-dino-base".
    """

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base"):
        """Initializes the GroundingDINOModel.

        Args:
            model_id (str): The Hugging Face Hub model identifier for the GroundingDINO model.
        """
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.classes: list[str] = []
        print(f"GroundingDINO model loaded on device: {self.device}")

    def get_classes(self) -> list[str]:
        """Returns the classes the model is currently set to detect.

        For zero-shot models, this is determined by the last `text_prompt` used.

        Returns:
            list[str]: A list of the class names currently set for detection.
        """
        return self.classes

    def set_classes(self, classes: Union[list[str], str]):
        """Sets the classes for the model to detect.

        Args:
            classes (Union[list[str], str]): A list of class names or a single string
                                            with class names separated by " . ".
        """
        if isinstance(classes, str):
            self.classes = [c.strip() for c in classes.split(" . ")]
        else:
            self.classes = classes
        # print(f"GroundingDINO classes set to: {self.classes}")

    def predict(
        self,
        image,
        text_prompt: Optional[str] = None,
        box_threshold: float = 0.05,
        text_threshold: float = 0.05,
        verbose: bool = False,
    ):
        """Performs zero-shot object detection on an image given a text prompt.

        Args:
            image (Union[str, np.ndarray, Image.Image]): The input image. Can be a file path, URL,
                                                        numpy array, or PIL Image object.
            text_prompt (str, optional): The text prompt describing the object(s) to detect.
                                        If provided, this will set the detection classes for the model.
            box_threshold (float, optional): The confidence threshold for filtering bounding boxes.
                                            Defaults to 0.05.
            text_threshold (float, optional): The confidence threshold for filtering text labels.
                                            Defaults to 0.05.
            verbose (bool, optional): If True, prints detailed detection results. Defaults to False.

        Returns:
            dict: A dictionary containing the detection results with keys for 'scores',
                'labels', and 'boxes'.
        """
        if text_prompt:
            self.set_classes(text_prompt)

        if not self.classes:
            raise ValueError("No classes set for detection. Please provide a 'text_prompt' or call 'set_classes' first.")

        prompt = " . ".join(self.classes)
        # print(f"Running GroundingDINO detection for prompt: '{prompt}'...")

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

        inputs = self.processor(images=image_pil, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )

        result_dict = results[0]
        result_dict["labels"] = result_dict.pop("text_labels")

        if verbose:
            print("\n--- Detection Results ---")
            for score, label, box in zip(result_dict["scores"], result_dict["labels"], result_dict["boxes"]):
                print(f"- Label: '{label}', Confidence: {score:.4f}, Box: {[round(c, 2) for c in box.tolist()]}")
            print("-------------------------\n")

        result_dict["boxes"] = [box.tolist() for box in result_dict["boxes"]]

        return result_dict

    def extract_features(self, image, text_prompt: str = "object"):
        """Extracts deep features (embeddings) from the model for an image.

        Args:
            image (Union[str, np.ndarray, Image.Image]): The input image.
            text_prompt (str, optional): A text prompt to guide feature extraction.
                                    Defaults to "object".

        Returns:
            Optional[torch.Tensor]: A tensor containing the extracted feature embeddings,
                                    or None if features could not be extracted.
        """
        # print(f"Extracting features from GroundingDINO using prompt: '{text_prompt}'...")

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

        if hasattr(outputs, "encoder_last_hidden_state_vision") and outputs.encoder_last_hidden_state_vision is not None:
            vision_features = outputs.encoder_last_hidden_state_vision
            pooled_features = torch.mean(vision_features, dim=1)
            return pooled_features.detach()
        else:
            print("Could not extract 'encoder_last_hidden_state_vision' from GroundingDINO output.")
            print(f"Available attributes in 'outputs': {dir(outputs)}")
            return None


class YOLOWorldModel:
    """A wrapper class for the YOLOWorld zero-shot object detection model.

    This class provides a standardized interface for using the YOLOWorld model, which
    extends the YOLO architecture with zero-shot detection capabilities. It allows for
    setting detection classes dynamically and performs prediction and feature extraction.

    Args:
        model_path (str): The local file path to the YOLOWorld model's weights file.
    """

    def __init__(self, model_path: str):
        """Initializes the YOLOWorldModel.

        Args:
            model_path (str): Path to the YOLOWorld model weights file.
        """
        self.model = YOLOWorld(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"YOLO-World model loaded on device: {self.device}")

    def get_classes(self) -> list[str]:
        """Returns the classes the model is currently set to detect.

        Returns:
            list[str]: A list of the class names currently set for detection.
        """
        return list(self.model.names.values())

    def set_classes(self, classes: Union[list[str], str]):
        """Sets the classes for the model to detect.

        Args:
            classes (Union[list[str], str]): A list of class names or a single string
                                            with class names separated by " . ".
        """
        if isinstance(classes, str):
            class_list = [c.strip() for c in classes.split(". ")]
        else:
            class_list = classes

        with torch.no_grad():
            self.model.set_classes(class_list)

        print(f"YOLOWorld classes set to: {class_list}")

    def predict(self, image, text_prompt: Optional[str] = None, **kwargs):
        """Performs zero-shot object detection on an image.

        Note: Before calling `predict`, you should set the desired classes using `set_classes`.

        Args:
            image (Union[str, np.ndarray, Image.Image]): The input image.
            text_prompt (str, optional): The text prompt describing the object(s) to detect.
                                    If provided, this will set the detection classes for the model.
            **kwargs: Additional keyword arguments for the `ultralytics.YOLOWorld.predict` method.

        Returns:
            dict: A dictionary of detection results with keys for 'scores', 'labels', and 'boxes'.
        """
        if text_prompt:
            # Parse the new prompt into a list of classes
            new_classes = [c.strip() for c in text_prompt.split(". ")]

            # Get the currently set classes
            current_classes = self.get_classes()

            # Only update the model's classes if the new prompt is different
            if new_classes != current_classes:
                self.set_classes(new_classes)

        with torch.no_grad():
            results = self.model.predict(image, **kwargs)

        result_dict = {"scores": [], "labels": [], "boxes": []}

        if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
            for box in results[0].boxes:
                result_dict["scores"].append(box.conf.item())
                result_dict["labels"].append(self.model.names[int(box.cls)])
                result_dict["boxes"].append(box.xyxy[0].tolist())

        return result_dict

    def extract_features(self, image, **kwargs):
        """Extracts deep feature embeddings from an image.

        Args:
            image (Union[str, np.ndarray, Image.Image]): The input image.
            **kwargs: Additional keyword arguments for the `ultralytics.YOLOWorld.embed` method.
                      Also accepts 'text_prompt' to set classes before embedding.

        Returns:
            Optional[torch.Tensor]: A tensor of feature embeddings, or None.
        """
        # Check for and handle the 'text_prompt' argument from kwargs
        if "text_prompt" in kwargs:
            text_prompt = kwargs.pop("text_prompt")  # Remove it so it's not passed to `embed`
            new_classes = [c.strip() for c in text_prompt.split(". ")]
            current_classes = self.get_classes()

            # Only update if the classes have changed
            if new_classes != current_classes:
                self.set_classes(new_classes)

        with torch.no_grad():
            # Call embed with the remaining (valid) kwargs
            features = self.model.embed(image, **kwargs)

        return features[0] if features else None


@register_model
def grounding_dino_detect_model(pretrained: bool = True, **kwargs):
    """Factory function for the GroundingDINO beetle detector.

    Args:
        pretrained (bool, optional): This argument is ignored as the model is always loaded
                                    with pretrained weights. Defaults to True.
        **kwargs: Additional keyword arguments, such as `model_id` to specify a different
                GroundingDINO model from the Hugging Face Hub.

    Returns:
        GroundingDINOModel: An instance of the GroundingDINO model wrapper.
    """
    if not pretrained:
        print("Warning: `pretrained=False` has no effect. GroundingDINO is always loaded from pretrained weights.")
    model_id = kwargs.get("model_id", "IDEA-Research/grounding-dino-base")
    return GroundingDINOModel(model_id=model_id)


@register_model
def yoloworldv2_bb_detect_model(pretrained: bool = True, **kwargs):
    """Factory function for the YOLOWorld beetle detector.

    Args:
        pretrained (bool, optional): If True, loads the default 'yolov8x-worldv2.pt' weights.
                                    This argument is effectively always True for this model.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        YOLOWorldModel: An instance of the YOLOWorld model wrapper.
    """
    local_weights_path = "yolov8x-worldv2.pt"
    return YOLOWorldModel(model_path=local_weights_path)
