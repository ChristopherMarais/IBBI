# src/ibbi/explain/__init__.py

"""
Provides the high-level Explainer class for model interpretability using LIME and SHAP.
"""

from ..models import ModelType
from .lime import explain_with_lime, plot_lime_explanation
from .shap import explain_with_shap, plot_shap_explanation


class Explainer:
    """A wrapper for LIME and SHAP explainability methods.

    This class provides a simple interface to generate model explanations using
    either LIME (Local Interpretable Model-agnostic Explanations) or SHAP
    (SHapley Additive exPlanations). It is designed to work with any model
    created using `ibbi.create_model`.

    Args:
        model (ModelType): An instantiated model from `ibbi.create_model`.
    """

    def __init__(self, model: ModelType):
        """A wrapper for LIME and SHAP explainability methods.

        This class provides a simple interface to generate model explanations using
        either LIME (Local Interpretable Model-agnostic Explanations) or SHAP
        (SHapley Additive exPlanations). It is designed to work with any model
        created using `ibbi.create_model`.

        Args:
            model (ModelType): An instantiated model from `ibbi.create_model`.
        """
        self.model = model

    def with_lime(self, image, **kwargs):
        """Generates a LIME explanation for a single image.

        LIME provides a local, intuitive explanation by showing which parts of an image
        contributed most to a specific prediction. This method is a wrapper around
        the `explain_with_lime` function.

        Args:
            image (PIL.Image.Image): The single image to be explained.
            **kwargs: Additional keyword arguments to be passed to the underlying
                      `ibbi.explain.lime.explain_with_lime` function. Common arguments
                      include `image_size`, `batch_size`, `num_samples`, `top_labels`,
                      and `num_features`.

        Returns:
            tuple[lime_image.ImageExplanation, PIL.Image.Image]: A tuple containing the LIME
            explanation object and the original image. The explanation object can be
            visualized using `ibbi.plot_lime_explanation`.
        """
        return explain_with_lime(self.model, image, **kwargs)

    def with_shap(self, explain_dataset, background_dataset, **kwargs):
        """Generates SHAP explanations for a set of images.

        SHAP (SHapley Additive exPlanations) provides robust, theoretically-grounded
        explanations by attributing a model's prediction to its input features. This
        method is a wrapper around the `explain_with_shap` function and requires a
        background dataset to integrate out features.

        Args:
            explain_dataset (list): A list of dictionaries, where each dictionary
                                    represents an image to be explained (e.g., `[{'image': img1}, {'image': img2}]`).
            background_dataset (list): A list of dictionaries representing a background dataset,
                                       used by SHAP to simulate feature absence.
            **kwargs: Additional keyword arguments to be passed to the underlying
                      `ibbi.explain.shap.explain_with_shap` function. Common arguments
                      include `num_explain_samples`, `max_evals`, `image_size`, and `text_prompt`.

        Returns:
            shap.Explanation: A SHAP Explanation object containing the SHAP values for each
                              image and each class. This object can be visualized using
                              `ibbi.plot_shap_explanation`.
        """
        return explain_with_shap(self.model, explain_dataset, background_dataset, **kwargs)


__all__ = [
    "Explainer",
    "plot_lime_explanation",
    "plot_shap_explanation",
]
