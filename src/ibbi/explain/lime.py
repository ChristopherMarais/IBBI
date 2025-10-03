# src/ibbi/explain/lime.py

"""
Highly optimized LIME-based model explainability for IBBI models,
featuring batched predictions and faster segmentation.
"""

from typing import Callable, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from lime import lime_image
from PIL import Image
from skimage.segmentation import slic
from skimage.transform import resize

from ..models import ModelType
from ..models.feature_extractors import HuggingFaceFeatureExtractor
from ..models.zero_shot import GroundingDINOModel


def _prediction_wrapper(model: ModelType, text_prompt: Optional[str] = None) -> Callable:
    """Creates a highly efficient, batched prediction function for LIME.

    This function acts as a factory, returning a `predict` function that is compatible
    with LIME's expectation of processing a numpy array of perturbed images. It handles
    different model types and ensures predictions are returned in the required format
    (a numpy array of class probabilities).

    Args:
        model (ModelType): The instantiated `ibbi` model to be explained.
        text_prompt (Optional[str], optional): A text prompt required for zero-shot models
                                               like GroundingDINO. Defaults to None.

    Returns:
        Callable: A `predict` function that takes a batch of images as a numpy array
                  and returns a numpy array of prediction probabilities.
    """

    def predict(image_array: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Batched prediction function for LIME.

        Args:
            image_array (np.ndarray): A numpy array representing a batch of images,
                                      with shape (num_samples, height, width, channels).
            verbose (bool, optional): Whether to print prediction details. Defaults to False.

        Returns:
            np.ndarray: A 2D numpy array of shape (num_samples, num_classes) containing the
                        prediction probabilities for each image and class.
        """
        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array, 0)

        # --- Handle different model types ---
        if isinstance(model, GroundingDINOModel):
            if not text_prompt:
                raise ValueError("A 'text_prompt' is required for explaining a GroundingDINOModel.")
            images_to_predict = [Image.fromarray(img) for img in image_array]
            # GroundingDINO predict is not batched, so we iterate
            predictions = np.zeros((image_array.shape[0], 1))
            for i, img in enumerate(images_to_predict):
                res = model.predict(img, text_prompt=text_prompt)
                if res["scores"]:
                    predictions[i, 0] = max(res["scores"])

        elif isinstance(model, HuggingFaceFeatureExtractor):
            # This model type doesn't support batching in the same way.
            # It also doesn't have classes or predict scores, so this is a fallback.
            print("Warning: LIME is not designed for pure feature extractors. Returning zero scores.")
            return np.zeros((image_array.shape[0], 1))

        else:  # Covers standard detection models like YOLO, RT-DETR
            image_tensor = torch.from_numpy(image_array).permute(0, 3, 1, 2).float() / 255.0
            device = next(model.model.parameters()).device
            image_tensor = image_tensor.to(device)

            class_names = model.get_classes()
            num_classes = len(class_names)
            predictions = np.zeros((image_array.shape[0], num_classes))

            results = model.model(image_tensor, verbose=verbose)

            for i, res in enumerate(results):
                if hasattr(res, "boxes") and res.boxes is not None:
                    for box in res.boxes:
                        class_idx = int(box.cls)
                        confidence = box.conf.item()
                        predictions[i, class_idx] = max(predictions[i, class_idx], confidence)
        return predictions

    return predict


def explain_with_lime(
    model: ModelType,
    image: Image.Image,
    text_prompt: Optional[str] = None,
    image_size: tuple[int, int] = (640, 640),
    batch_size: int = 50,
    num_samples: int = 1000,
    top_labels: int = 5,
    num_features: int = 100000,
) -> tuple[lime_image.ImageExplanation, Image.Image]:
    """Generates LIME explanations for a single image.

    This function uses the LIME (Local Interpretable Model-agnostic Explanations) algorithm
    to explain a model's prediction on a single image. It identifies which parts (superpixels)
    of the image were most influential in the model's decision-making process.

    Args:
        model (ModelType): The instantiated `ibbi` model to explain.
        image (Image.Image): The PIL Image to be explained.
        text_prompt (Optional[str], optional): A text prompt required for zero-shot models
                                               like GroundingDINO. Defaults to None.
        image_size (tuple[int, int], optional): The size to which the image will be resized for the explanation.
                                                Defaults to (640, 640).
        batch_size (int, optional): The number of perturbed images to process in a single batch. Defaults to 50.
        num_samples (int, optional): The number of perturbed images to generate for the LIME algorithm. Defaults to 1000.
        top_labels (int, optional): The number of top predicted classes to generate explanations for. Defaults to 5.
        num_features (int, optional): The maximum number of superpixels to include in the explanation. Defaults to 100000.

    Returns:
        tuple[lime_image.ImageExplanation, Image.Image]: A tuple containing:
            - The LIME explanation object from the `lime` library.
            - The original, unmodified PIL Image.
    """
    original_image = image
    image_to_explain = image.resize(image_size)
    prediction_fn = _prediction_wrapper(model, text_prompt=text_prompt)
    explainer = lime_image.LimeImageExplainer()
    image_np = np.array(image_to_explain)

    def segmentation_fn(x: np.ndarray) -> np.ndarray:
        """Faster `slic` segmentation."""
        return slic(x, n_segments=50, compactness=30, sigma=3)

    explanation = explainer.explain_instance(
        image_np,
        prediction_fn,
        top_labels=top_labels,
        hide_color=0,
        num_samples=num_samples,
        num_features=num_features,
        batch_size=batch_size,
        segmentation_fn=segmentation_fn,
    )
    return explanation, original_image


def plot_lime_explanation(explanation: lime_image.ImageExplanation, image: Image.Image, top_k: int = 1, alpha: float = 0.6) -> None:
    """Plots a detailed LIME explanation with a red-to-green overlay.

    This function visualizes the output of `explain_with_lime`. It overlays the original
    image with a heatmap where green areas indicate features that positively contributed
    to the prediction, and red areas indicate negative contributions.

    Args:
        explanation (lime_image.ImageExplanation): The explanation object generated by `explain_with_lime`.
        image (Image.Image): The original image that was explained.
        top_k (int, optional): The number of top classes to display explanations for. Defaults to 1.
        alpha (float, optional): The transparency of the color overlay. Defaults to 0.6.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    segments = explanation.segments

    for label in explanation.top_labels[:top_k]:  # type: ignore[attr-defined]
        print(f"\n--- Explanation for Class Index: {label} ---")

        exp_for_label = explanation.local_exp.get(label)
        if not exp_for_label:
            print(f"No explanation available for class {label}.")
            continue

        weight_map = np.zeros(segments.shape, dtype=np.float32)
        for feature, weight in exp_for_label:
            weight_map[segments == feature] = weight

        max_abs_weight = np.max(np.abs(weight_map))
        if max_abs_weight == 0:
            print(f"No significant features found for class {label}.")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image)
            ax.set_title(f"LIME: No features for class {label}")
            ax.axis("off")
            plt.show()
            continue

        norm = mcolors.Normalize(vmin=-max_abs_weight, vmax=max_abs_weight)
        cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]

        colored_overlay_rgba = cmap(norm(weight_map))
        original_size = image.size
        colored_overlay_resized = resize(
            colored_overlay_rgba,
            (original_size[1], original_size[0]),
            anti_aliasing=True,
            mode="constant",
        )

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(image)
        ax.imshow(colored_overlay_resized, alpha=alpha)  # type: ignore[arg-type]

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Feature Weight (Green: Positive, Red: Negative)", rotation=270, labelpad=20)

        ax.set_title(f"LIME Explanation for Class Index: {label}")
        ax.axis("off")
        plt.show()
