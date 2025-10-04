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
        include_full_probabilities: bool = False,
        **kwargs,
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
            include_full_probabilities (bool, optional): If True, includes a 'full_results' key in the
                                                         output with detailed probabilities for each class.
                                                         Defaults to False.

        Returns:
            dict: A dictionary containing the detection results with keys for 'scores',
                'labels', and 'boxes'.
        """
        if text_prompt:
            self.set_classes(text_prompt)

        if not self.classes:
            raise ValueError("No classes set for detection. Please provide a 'text_prompt' or call 'set_classes' first.")

        prompt = " . ".join(self.classes)

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

        result_dict = {"scores": [], "labels": [], "boxes": []}
        if include_full_probabilities:
            result_dict["full_results"] = []
            result_dict["class_names"] = self.classes

        if results and results[0]["scores"].nelement() > 0:
            for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
                result_dict["scores"].append(score.item())
                result_dict["labels"].append(label)
                bbox = box.tolist()
                result_dict["boxes"].append(bbox)

                if include_full_probabilities:
                    # Create a simple proxy probability distribution
                    probabilities = np.zeros(len(self.classes))
                    # Clean the label and the class list for robust, case-insensitive matching
                    cleaned_label = label.strip().lower()
                    cleaned_classes = [c.strip().lower() for c in self.classes]
                    class_id = -1
                    if cleaned_label in cleaned_classes:
                        class_id = cleaned_classes.index(cleaned_label)
                        probabilities[class_id] = score.item()

                    result_dict["full_results"].append(
                        {
                            "predicted_class": label,
                            "predicted_class_id": class_id,
                            "confidence": score.item(),
                            "class_probabilities": probabilities.tolist(),
                            "bbox": bbox,
                        }
                    )

        if verbose:
            print("\n--- Detection Results ---")
            for score, label, box in zip(result_dict["scores"], result_dict["labels"], result_dict["boxes"]):
                print(f"- Label: '{label}', Confidence: {score:.4f}, Box: {[round(c, 2) for c in box]}")
            print("-------------------------\n")

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

        # Perform a minimal warm-up by setting a dummy class. This initializes
        # the text encoder's weights and state without running a full prediction,
        # which was causing state conflicts.
        print("Performing one-time warm-up for YOLOWorld text encoder...")
        try:
            with torch.no_grad():
                self.set_classes(["warm-up"])
            print("Warm-up complete.")
        except Exception as e:
            print(f"Warning: YOLOWorld warm-up failed with an error: {e}")

    def get_classes(self) -> list[str]:
        """Returns the classes the model is currently set to detect.

        Returns:
            list[str]: A list of the class names currently set for detection.
        """
        return list(self.model.names.values())

    def set_classes(self, classes: Union[list[str], str]):
        """Sets the classes for the model to detect.

        This method now includes a targeted fix to ensure the internal CLIP model
        is in the correct evaluation state before processing text.

        Args:
            classes (Union[list[str], str]): A list of class names or a single string
                                            with class names separated by " . ".
        """
        if isinstance(classes, str):
            class_list = [c.strip() for c in classes.split(".") if c.strip()]
        else:
            class_list = classes

        # The root cause of the error is the internal state of the CLIP text encoder.
        # Explicitly setting the clip_model to eval() mode here ensures that its
        # parameters are not tracking gradients, which resolves the 'version counter'
        # conflict even when this method is called multiple times.
        if hasattr(self.model, "clip_model") and self.model.clip_model is not None:
            self.model.clip_model.eval()

        with torch.no_grad():
            self.model.set_classes(class_list)

    def predict(self, image, text_prompt: Optional[str] = None, include_full_probabilities: bool = False, **kwargs):
        """Performs zero-shot object detection on an image.

        Args:
            image (Union[str, np.ndarray, Image.Image]): The input image.
            text_prompt (str, optional): The text prompt describing the object(s) to detect.
            **kwargs: Additional keyword arguments for the `ultralytics.YOLOWorld.predict` method.

        Returns:
            dict: A dictionary of detection results.
        """
        with torch.no_grad():
            if text_prompt:
                new_classes = [c.strip() for c in text_prompt.split(".") if c.strip()]
                if new_classes != self.get_classes():
                    self.set_classes(new_classes)

            results = self.model.predict(image, **kwargs)

        result_dict = {"scores": [], "labels": [], "boxes": []}
        if include_full_probabilities:
            result_dict["full_results"] = []
            result_dict["class_names"] = self.get_classes()

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
                    # Create a proxy probability distribution
                    probabilities = np.zeros(len(self.get_classes()))
                    if label in self.get_classes():
                        class_id_in_list = self.get_classes().index(label)
                        probabilities[class_id_in_list] = confidence

                    result_dict["full_results"].append(
                        {
                            "predicted_class": label,
                            "predicted_class_id": self.get_classes().index(label) if label in self.get_classes() else -1,
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
            **kwargs: Additional arguments, including 'text_prompt'.

        Returns:
            Optional[torch.Tensor]: A tensor of feature embeddings.
        """
        with torch.no_grad():
            if "text_prompt" in kwargs:
                text_prompt = kwargs.pop("text_prompt")
                new_classes = [c.strip() for c in text_prompt.split(". ")]
                if new_classes != self.get_classes():
                    self.set_classes(new_classes)

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
