# src/ibbi/explain/__init__.py

"""
Provides the high-level Explainer class for model interpretability using LIME and SHAP.
"""

from ..models import ModelType
from .lime import explain_with_lime, plot_lime_explanation
from .shap import explain_with_shap, plot_shap_explanation


class Explainer:
    """
    A wrapper for LIME and SHAP explainability methods.

    Args:
        model (ModelType): An instantiated model from `ibbi.create_model`.
    """

    def __init__(self, model: ModelType):
        self.model = model

    def with_lime(self, image, **kwargs):
        """
        Generates a LIME explanation for a single image.
        """
        return explain_with_lime(self.model, image, **kwargs)

    def with_shap(self, explain_dataset, background_dataset, **kwargs):
        """
        Generates SHAP explanations for a set of images.
        """
        return explain_with_shap(self.model, explain_dataset, background_dataset, **kwargs)


__all__ = [
    "Explainer",
    "plot_lime_explanation",
    "plot_shap_explanation",
]
