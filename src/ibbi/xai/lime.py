# src/ibbi/xai/lime.py

"""
Highly optimized LIME-based model explainability for IBBI models,
featuring batched predictions and faster segmentation.
"""

from typing import Callable, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from lime import lime_image
from PIL import Image
from skimage.segmentation import slic
from skimage.transform import resize

from ..models import ModelType
from ..models.zero_shot_detection import GroundingDINOModel


def _prediction_wrapper(model: ModelType, text_prompt: Optional[str] = None) -> Callable:
    """
    Creates a highly efficient, batched prediction function for LIME.
    This version processes the entire batch of images as a single tensor.
    """

    def predict(image_array: np.ndarray) -> np.ndarray:
        """
        Batched prediction function for LIME.

        Args:
            image_array (np.ndarray): A numpy array of perturbed images from LIME,
                                    with shape (num_images, height, width, channels).

        Returns:
            np.ndarray: A numpy array of prediction probabilities for each class.
        """
        # --- Tensor-based Batching for Maximum Efficiency ---
        # The model expects a PyTorch tensor in the format (batch, channels, height, width)
        # and values scaled to [0, 1].
        # This check is needed because LIME can sometimes send a single image (ndim=3)
        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array, 0)

        # 1. Convert NumPy array to a PyTorch tensor.
        # 2. Permute dimensions from (B, H, W, C) to (B, C, H, W).
        # 3. Convert to float and scale pixel values from [0, 255] to [0, 1].
        image_tensor = torch.from_numpy(image_array).permute(0, 3, 1, 2).float() / 255.0

        # Move tensor to the same device as the model for GPU acceleration
        device = next(model.model.parameters()).device
        image_tensor = image_tensor.to(device)

        # For GroundingDINOModel (requires a different prediction logic)
        if isinstance(model, GroundingDINOModel):
            if not text_prompt:
                raise ValueError("A 'text_prompt' is required for explaining a GroundingDINOModel.")
            # This model's predict is not optimized for tensors, so we still use a list
            images_to_predict = [Image.fromarray(img) for img in image_array]
            results = model.predict(images_to_predict, text_prompt=text_prompt)
            num_classes = 1
            predictions = np.zeros((image_array.shape[0], num_classes))
            for i, res in enumerate(results):
                if res["scores"].nelement() > 0:
                    predictions[i, 0] = res["scores"].max().item()

        # For standard detection models (like YOLO)
        else:
            class_names = model.get_classes()
            num_classes = len(class_names)
            predictions = np.zeros((image_array.shape[0], num_classes))

            # --- Perform a single, efficient batched prediction on the tensor ---
            results = model.model(image_tensor)

            # Process the batch of results
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
    image_size: Tuple[int, int] = (640, 640),
    batch_size: int = 50,
    num_samples: int = 1000,
    top_labels: int = 5,
    num_features: int = 100000,
) -> Tuple[lime_image.ImageExplanation, Image.Image]:
    """
    Generates LIME explanations for a single image using batched predictions
    and a faster segmentation algorithm.

    Args:
        model (ModelType): The model to explain.
        image (Image.Image): The input image to explain.
        text_prompt (Optional[str], optional): Text prompt for GroundingDINOModel. Defaults to None.
        batch_size (int, optional): The number of samples to predict on in a single batch.
                                    Defaults to 50.
        num_samples (int, optional): The number of perturbations to generate for LIME.
                                    Defaults to 1000.
        top_labels (int, optional): The number of top labels to consider. Defaults to 5.
        num_features (int, optional): The number of superpixels to generate. Defaults to 100000.

    Returns:
        Tuple[lime_image.ImageExplanation, Image.Image]: The LIME explanation object and the original image.
    """
    original_image = image  # Keep the original for plotting
    image_to_explain = image.resize(image_size)
    # Convert the image to a numpy array and normalize it
    prediction_fn = _prediction_wrapper(model, text_prompt=text_prompt)
    explainer = lime_image.LimeImageExplainer()
    image_np = np.array(image_to_explain)

    # FIX: Replaced lambda with a nested `def` to fix ruff error E731.
    def segmentation_fn(x: np.ndarray) -> np.ndarray:
        """Faster `slic` segmentation."""
        return slic(x, n_segments=50, compactness=30, sigma=3)

    # Generate the explanation with all optimizations
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


def plot_lime_explanation(
    explanation: lime_image.ImageExplanation, image: Image.Image, top_k: int = 1, alpha: float = 0.6
) -> None:
    """
    Plots a detailed LIME explanation with a red-to-green overlay.

    This function generates a visualization for each of the top 'k' predicted
    classes. Positive contributions (features that support the prediction)
    are colored green, while negative contributions are colored red.
    The intensity of the color corresponds to the weight of the feature.

    Args:
        explanation (lime_image.ImageExplanation): The LIME explanation object.
        image (Image.Image): The original, full-resolution input image.
        top_k (int, optional): The number of top predicted classes to visualize.
                                Defaults to 1.
        alpha (float, optional): The transparency of the color overlay.
                                Defaults to 0.6.
    """
    # 1. Display the original image first for reference
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    # Get the segmentation map from the explanation
    segments = explanation.segments

    # 2. Iterate through the top 'k' predictions to explain each one
    # FIX: Suppress pyright error as `top_labels` exists on the explanation object at runtime.
    for label in explanation.top_labels[:top_k]:  # type: ignore[attr-defined]
        print(f"\n--- Explanation for Class Index: {label} ---")

        # Get the feature weights for the current class
        exp_for_label = explanation.local_exp.get(label)
        if not exp_for_label:
            print(f"No explanation available for class {label}.")
            continue

        # 3. Create a "weight map" with the same dimensions as the segmentation
        weight_map = np.zeros(segments.shape, dtype=np.float32)
        for feature, weight in exp_for_label:
            weight_map[segments == feature] = weight

        # Find the maximum absolute weight for normalization
        max_abs_weight = np.max(np.abs(weight_map))
        if max_abs_weight == 0:
            print(f"No significant features found for class {label}.")
            # Optionally, still show the image without an overlay
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image)
            ax.set_title(f"LIME: No features for class {label}")
            ax.axis("off")
            plt.show()
            continue

        # 4. Normalize the weights to the range [-1, 1] for coloring
        norm = mcolors.Normalize(vmin=-max_abs_weight, vmax=max_abs_weight)

        # Use a diverging colormap: Red -> White/Yellow -> Green.
        # FIX: Suppress pyright error as `RdYlGn` is a valid attribute of `plt.cm`.
        cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]

        # 5. Create the color overlay and resize it to the original image size
        # The colormap returns RGBA values (0-1 range)
        colored_overlay_rgba = cmap(norm(weight_map))

        # Resize overlay to match the original image's dimensions
        original_size = image.size
        colored_overlay_resized = resize(
            colored_overlay_rgba,
            (original_size[1], original_size[0]),  # (height, width)
            anti_aliasing=True,
            mode="constant",
        )

        # 6. Plot the original image, overlay the explanation, and add a colorbar
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(image)
        # Display the overlay. We don't need the return value of this call.
        ax.imshow(colored_overlay_resized, alpha=alpha)  # type: ignore[arg-type]

        # Create a ScalarMappable that understands the normalization and colormap.
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # Set a dummy array, as it's not used for drawing the colorbar.
        sm.set_array([])

        # Add the colorbar to the plot, using the ScalarMappable.
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Feature Weight (Green: Positive, Red: Negative)", rotation=270, labelpad=20)

        ax.set_title(f"LIME Explanation for Class Index: {label}")
        ax.axis("off")
        plt.show()
