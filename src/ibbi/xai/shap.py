# src/ibbi/xai/shap.py

"""
SHAP-based model explainability for IBBI models using PartitionExplainer.
"""

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import shap
from PIL import Image
from shap import maskers

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
    Creates a prediction function compatible with SHAP explainers that
    pass a batch of images as a 4D array.
    """

    def predict(image_batch: np.ndarray) -> np.ndarray:
        # PartitionExplainer provides a 4D array of shape (num_samples, height, width, channels)
        num_images = image_batch.shape[0]

        if image_batch.max() <= 1.0:
            image_batch = (image_batch * 255).astype(np.uint8)

        images_to_predict = [Image.fromarray(img) for img in image_batch]

        if isinstance(model, GroundingDINOModel):
            if not text_prompt:
                raise ValueError("A 'text_prompt' is required for explaining a GroundingDINOModel.")
            num_classes = 1
            predictions = np.zeros((num_images, num_classes))
            results = [model.predict(img, text_prompt=text_prompt) for img in images_to_predict]
            for i, res in enumerate(results):
                if res["scores"].nelement() > 0:
                    predictions[i, 0] = res["scores"].max().item()
        else:  # Covers YOLOWorld and other standard models
            class_names = model.get_classes()
            num_classes = len(class_names)
            predictions = np.zeros((num_images, num_classes))
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
    max_evals: int = 1000,
    image_size: tuple = (224, 224),
    text_prompt: Optional[str] = None,
    **kwargs,  # Absorb unused kwargs
) -> shap.Explanation:
    """
    Generates SHAP explanations for a given model using the PartitionExplainer.
    This function is computationally more efficient than KernelExplainer for images.
    """
    prediction_fn = _prediction_wrapper(model, text_prompt=text_prompt)

    if isinstance(model, GroundingDINOModel):
        if not text_prompt:
            raise ValueError("A 'text_prompt' is required for explaining a GroundingDINOModel.")
        output_names = [text_prompt]
    else:
        output_names = model.get_classes()

    # --- Prepare Datasets ---
    background_pil_images = [d["image"] for d in background_dataset]
    background_images = [np.array(img) for img in background_pil_images]
    background_images_norm = np.stack([_prepare_image_for_shap(img) for img in background_images])

    images_to_explain_pil = [explain_dataset[i]["image"].resize(image_size) for i in range(num_explain_samples)]
    images_to_explain_np = np.array([np.array(img) for img in images_to_explain_pil])
    images_to_explain_norm = _prepare_image_for_shap(images_to_explain_np.copy())

    # --- Initialize and Run PartitionExplainer ---
    masker = maskers.Image("blur(16, 16)", images_to_explain_norm[0].shape)

    print(f"Using a background dataset of {len(background_images_norm)} images.")
    explainer = shap.PartitionExplainer(prediction_fn, masker)

    print(f"Generating SHAP explanations for {num_explain_samples} images...")
    shap_values = explainer(images_to_explain_norm, max_evals=max_evals, main_effects=False)

    # Assign correct metadata to the explanation object
    shap_values.data = images_to_explain_np
    shap_values.output_names = output_names

    return shap_values


def plot_shap_explanation(
    shap_explanation_for_single_image: shap.Explanation,
    model: ModelType,
    top_k: int = 5,
    text_prompt: Optional[str] = None,
) -> None:
    """
    Plots SHAP explanations for a SINGLE image. This function is compatible
    with the output from PartitionExplainer.
    """
    print("\n--- Generating Explanations for Image ---")

    image_for_plotting = shap_explanation_for_single_image.data
    shap_values = shap_explanation_for_single_image.values
    class_names = np.array(shap_explanation_for_single_image.output_names)

    prediction_fn = _prediction_wrapper(model, text_prompt=text_prompt)
    image_norm = _prepare_image_for_shap(np.array(image_for_plotting))

    prediction_scores = prediction_fn(image_norm[np.newaxis, ...])[0]

    if len(prediction_scores) > 1:
        top_indices = np.argsort(prediction_scores)[-top_k:][::-1]
    else:
        top_indices = [0]

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

            shap_values_for_class = shap_values[:, :, :, class_idx]  # type: ignore

            if np.all(shap_values_for_class == 0):
                print("⚠️  Warning: SHAP values are all zero. The plot will be empty.")
                print("   This can happen if the model's prediction is not sensitive to the masking.")

            shap.image_plot(
                shap_values=[shap_values_for_class],
                pixel_values=image_for_plotting,
                show=True,
            )
