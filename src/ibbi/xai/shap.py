# src/ibbi/xai/shap.py

"""
SHAP-based model explainability for IBBI models.
"""

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import shap
from PIL import Image
from shap.maskers import Image as ImageMasker

# Import specific model types to handle them differently
from ..models import ModelType
from ..models.zero_shot_detection import GroundingDINOModel


def _prepare_image_for_shap(image_array: np.ndarray) -> np.ndarray:
    """
    Normalizes image array to have pixel values between 0 and 1.
    """
    if image_array.max() > 1.0:
        image_array = image_array.astype(np.float32) / 255.0
    return image_array


def _prediction_wrapper(model: ModelType, text_prompt: Optional[str] = None) -> Callable:
    """
    Creates a prediction function compatible with the SHAP explainer.
    """

    def predict(image_array: np.ndarray) -> np.ndarray:
        # This function remains the same as it correctly handles model predictions
        if isinstance(model, GroundingDINOModel):
            if not text_prompt:
                raise ValueError("A 'text_prompt' is required for explaining a GroundingDINOModel.")
            num_classes = 1
            predictions = np.zeros((image_array.shape[0], num_classes))
            images_to_predict = [Image.fromarray((img * 255).astype(np.uint8)) for img in image_array]
            results = [model.predict(img, text_prompt=text_prompt) for img in images_to_predict]
            for i, res in enumerate(results):
                if res["scores"].nelement() > 0:
                    predictions[i, 0] = res["scores"].max().item()
        else:
            class_names = model.get_classes()
            num_classes = len(class_names)
            predictions = np.zeros((image_array.shape[0], num_classes))
            images_to_predict = [Image.fromarray((img * 255).astype(np.uint8)) for img in image_array]
            results = model.predict(images_to_predict, verbose=False)
            for i, res in enumerate(results):
                if hasattr(res, "boxes") and res.boxes is not None:
                    for box in res.boxes:
                        class_idx = int(box.cls)
                        confidence = box.conf.item()
                        predictions[i, class_idx] = max(predictions[i, class_idx], confidence)
        return predictions

    return predict


def explain_with_shap(
    model: ModelType,
    explain_dataset: list,
    background_dataset: list,
    num_explain_samples: int,
    num_background_samples: int,
    max_evals: int = 1000,
    batch_size: int = 50,
    image_size: tuple = (640, 640),
    text_prompt: Optional[str] = None,
) -> shap.Explanation:
    """
    Generates SHAP explanations for a given model.
    This function is computationally intensive.
    """
    prediction_fn = _prediction_wrapper(model, text_prompt=text_prompt)

    if isinstance(model, GroundingDINOModel):
        if not text_prompt:
            raise ValueError("A 'text_prompt' is required for explaining a GroundingDINOModel.")
        output_names = [text_prompt]
    else:
        output_names = model.get_classes()

    background_pil_images = [background_dataset[i]["image"].resize(image_size) for i in range(num_background_samples)]
    background_images = [np.array(img) for img in background_pil_images]
    background_images_norm = np.stack([_prepare_image_for_shap(img) for img in background_images])

    background_summary = np.median(background_images_norm, axis=0)

    images_to_explain_pil = [explain_dataset[i]["image"].resize(image_size) for i in range(num_explain_samples)]
    images_to_explain = [np.array(img) for img in images_to_explain_pil]
    images_to_explain_norm = [_prepare_image_for_shap(img) for img in images_to_explain]
    images_to_explain_array = np.array(images_to_explain_norm)

    masker = ImageMasker(background_summary, shape=images_to_explain_array[0].shape)
    explainer = shap.Explainer(prediction_fn, masker, output_names=output_names)

    # FIX: Reformatted to fix ruff error E501 (line too long).
    # Ignoring the arg-type error which is due to incorrect type hints in the shap library
    shap_explanation = explainer(images_to_explain_array, max_evals=max_evals, batch_size=batch_size)  # type: ignore[arg-type]
    shap_explanation.data = np.array(images_to_explain)
    return shap_explanation


def plot_shap_explanation(
    shap_explanation_for_single_image: shap.Explanation,
    model: ModelType,
    top_k: int = 5,
    text_prompt: Optional[str] = None,
) -> None:
    """
    Plots SHAP explanations for a SINGLE image.

    Args:
        shap_explanation_for_single_image: A SHAP Explanation object for a SINGLE image.
            To get this, index the output of explain_with_shap (e.g., `shap_explanation[0]`).
        model: The model that was explained.
        top_k: The number of top predicted classes to visualize.
        text_prompt: The text prompt, if a GroundingDINOModel was used.
    """
    print("\n--- Generating Explanations for Image ---")

    image_for_plotting = shap_explanation_for_single_image.data
    shap_values = shap_explanation_for_single_image.values
    class_names = np.array(shap_explanation_for_single_image.output_names)

    prediction_fn = _prediction_wrapper(model, text_prompt=text_prompt)
    image_norm = _prepare_image_for_shap(np.array(image_for_plotting))
    prediction_scores_batch = prediction_fn(np.expand_dims(image_norm, axis=0))
    prediction_scores = prediction_scores_batch[0]

    top_indices = np.argsort(prediction_scores)[-top_k:][::-1]

    plt.figure(figsize=(5, 5))
    plt.imshow(image_for_plotting)
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    for class_idx in top_indices:
        if prediction_scores[class_idx] > 0:
            class_name = class_names[class_idx]
            score = prediction_scores[class_idx]
            print(f"Explanation for '{class_name}' (Prediction Score: {score:.3f})")

            # FIX 2: Added `# type: ignore` to suppress the incorrect slicing error from pyright.
            # The slicing logic is correct for the shape of the shap_values array.
            shap_values_for_class = shap_values[:, :, :, class_idx]  # type: ignore[misc]

            shap.image_plot(
                shap_values=[shap_values_for_class],
                pixel_values=np.array(image_for_plotting),
                show=True,
            )
