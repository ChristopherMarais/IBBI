# src/ibbi/xai/shap.py

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import shap
from datasets import Dataset
from PIL import Image as PILImage

from ibbi.models import GroundingDINOModel, ModelType
from ibbi.utils.hub import get_model_config_from_hub

# --- Setup logging ---
logger = logging.getLogger(__name__)


def get_background_images(dataset: Dataset, num_samples: int) -> Tuple[np.ndarray, Sequence[Dict[str, Any]]]:
    """
    Get a sample of background images from the dataset.
    """
    samples = dataset.shuffle().select(range(num_samples))
    images = [sample["image"].convert("RGB") for sample in samples]  # type: ignore
    return np.array(images), [dict(s) for s in samples]


def get_example_images(dataset: Dataset, num_samples: int) -> Tuple[np.ndarray, Sequence[Dict[str, Any]]]:
    """
    Get a sample of example images from the dataset for explanation.
    """
    samples = dataset.shuffle().select(range(num_samples))
    images = [sample["image"].convert("RGB") for sample in samples]  # type: ignore
    return np.array(images), [dict(s) for s in samples]


def _prediction_wrapper(model: ModelType, text_prompt: Optional[str] = None) -> Callable:
    """
    Creates a prediction function that is compatible with the SHAP explainer.
    """

    def predict(image_array: np.ndarray) -> np.ndarray:
        images = [PILImage.fromarray(image) for image in image_array]
        predictions: List[np.ndarray] = []

        if isinstance(model, GroundingDINOModel):
            if text_prompt is None:
                raise ValueError("text_prompt cannot be None for GroundingDINOModel")
            dino_classes = [s.strip() for s in text_prompt.split(".")]
            dino_num_classes = len(dino_classes)
            results = model.predict(images, text_prompt=text_prompt)

            for res in results:
                scores = np.zeros(dino_num_classes)
                if res and res.confidence is not None and len(res.confidence) > 0:
                    if res.labels is not None:
                        for i, label_str in enumerate(res.labels):
                            try:
                                label_idx = dino_classes.index(label_str)
                                scores[label_idx] = max(scores[label_idx], res.confidence[i])
                            except ValueError:
                                continue
                predictions.append(scores)

        else:  # Handle other YOLO-based models
            model_name = getattr(model.model, "name", None)
            if not isinstance(model_name, str):
                raise ValueError("Could not determine model name for config lookup.")

            config = get_model_config_from_hub(model_name)
            id2label = config.get("id2label", {})
            class_names = [id2label[str(i)] for i in sorted(int(k) for k in id2label.keys())]
            num_classes = len(class_names)
            results = model.predict(images, verbose=False)

            for res in results:
                scores = np.zeros(num_classes)
                if res.boxes is not None and len(res.boxes) > 0:
                    for box in res.boxes:
                        class_idx = int(box.cls)
                        if class_idx < num_classes:
                            conf_value = float(box.conf.item())
                            scores[class_idx] = max(scores[class_idx], conf_value)
                predictions.append(scores)

        return np.array(predictions)

    return predict


def explain_model(
    model: ModelType,
    background_dataset: Dataset,
    explain_dataset: Dataset,
    num_background_samples: int = 10,
    num_explain_samples: int = 3,
    max_evals: int = 500,
    text_prompt: Optional[str] = None,
) -> shap.Explanation:
    """
    Generates SHAP explanations for a given model.
    """
    logger.info("Starting SHAP explanation generation...")

    background_np, _ = get_background_images(background_dataset, num_background_samples)
    images_to_explain_np, _ = get_example_images(explain_dataset, num_explain_samples)

    prediction_fn = _prediction_wrapper(model, text_prompt=text_prompt)

    masker = shap.maskers.Image("inpaint_telea", background_np[0].shape)  # type: ignore

    if isinstance(model, GroundingDINOModel):
        if text_prompt is None:
            raise ValueError("text_prompt is required for explaining GroundingDINOModel")
        output_names = [s.strip() for s in text_prompt.split(".")]
    else:
        model_name = getattr(model.model, "name", None)
        if not isinstance(model_name, str):
            raise ValueError("Could not determine model name for config lookup.")

        config = get_model_config_from_hub(model_name)
        id2label = config.get("id2label", {})
        output_names = [id2label[str(i)] for i in sorted(int(k) for k in id2label.keys())]

    explainer = shap.Explainer(prediction_fn, masker, output_names=output_names)

    logger.info(f"Generating SHAP values with max_evals={max_evals}. This may take a while...")
    shap_values = explainer(
        images_to_explain_np,
        max_evals=max_evals,  # type: ignore
        main_effects=False,
    )
    logger.info("SHAP explanation generation complete.")

    return shap_values
