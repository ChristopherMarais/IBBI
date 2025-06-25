# src/ibbi/xai/shap.py

"""
SHAP-based model explainability for IBBI models.
"""

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import shap
from PIL import Image

# Import specific model types to handle them differently
from ..models import ModelType
from ..models.zero_shot_detection import GroundingDINOModel


def _prepare_image_for_shap(image_array: np.ndarray) -> np.ndarray:
    if image_array.max() > 1.0:
        image_array = image_array.astype(np.float32) / 255.0
    return image_array


def _prediction_wrapper(model: ModelType, text_prompt: Optional[str] = None) -> Callable:
    def predict(image_array: np.ndarray) -> np.ndarray:
        # Handle GroundingDINOModel which has a different API
        if isinstance(model, GroundingDINOModel):
            if not text_prompt:
                raise ValueError("A 'text_prompt' is required for explaining a GroundingDINOModel.")
            # For zero-shot models, the "class" is the text prompt itself
            class_names = [text_prompt]
            num_classes = 1
            predictions = np.zeros((image_array.shape[0], num_classes))
            images_to_predict = [Image.fromarray((img * 255).astype(np.uint8)) for img in image_array]

            # Call predict without 'verbose' and with 'text_prompt'
            results = [model.predict(img, text_prompt=text_prompt) for img in images_to_predict]

            for i, res in enumerate(results):
                # GroundingDINO returns a dictionary with scores
                if res["scores"].nelement() > 0:
                    predictions[i, 0] = res["scores"].max().item()
        else:
            # This logic works for the other detection models
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


def explain_model(
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

    # Get class names based on the model type
    if isinstance(model, GroundingDINOModel):
        if not text_prompt:
            raise ValueError("A 'text_prompt' is required for explaining a GroundingDINOModel.")
        output_names = [text_prompt]
    else:
        output_names = model.get_classes()

    images_to_explain_pil = [explain_dataset[i]["image"].resize(image_size) for i in range(num_explain_samples)]
    images_to_explain = [np.array(img) for img in images_to_explain_pil]
    images_to_explain_norm = [_prepare_image_for_shap(img) for img in images_to_explain]
    images_to_explain_array = np.array(images_to_explain_norm)

    # FIX: Suppress false positive pyright error for this line.
    masker = shap.maskers.Image("blur(128,128)", shape=images_to_explain_array[0].shape)  # type: ignore
    explainer = shap.Explainer(prediction_fn, masker, output_names=output_names)

    # FIX: Suppress false positive pyright errors for this line.
    shap_explanation = explainer(images_to_explain_array, max_evals=max_evals, batch_size=batch_size)  # type: ignore
    shap_explanation.data = np.array(images_to_explain)
    return shap_explanation


def plot_explanations(
    shap_explanation: shap.Explanation, model: ModelType, top_k: int = 5, text_prompt: Optional[str] = None
) -> None:
    """
    Plots SHAP explanations for the top_k predicted classes for each image.
    """
    print(f"Displaying SHAP explanations for top {top_k} predictions...")

    images_for_plotting = shap_explanation.data
    class_names = np.array(shap_explanation.output_names)

    # Pass text_prompt to the wrapper for GroundingDINO
    prediction_fn = _prediction_wrapper(model, text_prompt=text_prompt)
    images_norm = np.array([_prepare_image_for_shap(img) for img in images_for_plotting])
    prediction_scores = prediction_fn(images_norm)

    for i in range(len(images_for_plotting)):
        print(f"\n--- Explanations for Image {i+1} ---")

        top_indices = np.argsort(prediction_scores[i])[-top_k:][::-1]

        plt.figure(figsize=(5, 5))
        plt.imshow(images_for_plotting[i])
        plt.title("Original Image")
        plt.axis("off")
        plt.show()

        for class_idx in top_indices:
            if prediction_scores[i, class_idx] > 0:
                class_name = class_names[class_idx]
                score = prediction_scores[i, class_idx]
                print(f"Explanation for '{class_name}' (Prediction Score: {score:.3f})")

                # FIX: Suppress false positive pyright error for this line.
                shap_values_for_class = shap_explanation.values[i, :, :, :, class_idx]  # type: ignore
                image_for_plot = images_for_plotting[i]
                shap.image_plot(shap_values=shap_values_for_class, pixel_values=image_for_plot, show=True)
