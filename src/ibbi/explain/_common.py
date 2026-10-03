# src/ibbi/explain/_common.py

"""Prediction function shared by the LIME and SHAP explainers."""

from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image


def prediction_function(model: Any, text_prompt: str | None = None) -> Callable[[np.ndarray], np.ndarray]:
    """Returns f(images [N, H, W, 3] uint8 or float in [0, 1]) -> class scores [N, n_classes] built on `model.predict_proba`.

    Detectors score each class by their highest detection confidence, classifiers by the calibrated species
    probability, zero-shot detectors by the highest score per prompt (`text_prompt` sets the prompts first).
    """
    if text_prompt is not None and hasattr(model, "set_classes"):
        model.set_classes(text_prompt)
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} has no predict_proba(); it cannot be explained with LIME or SHAP.")

    def predict(image_array: np.ndarray, **kwargs) -> np.ndarray:
        arr = np.asarray(image_array)
        if arr.ndim == 3:
            arr = arr[None]
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return model.predict_proba([Image.fromarray(a) for a in arr])

    return predict


def class_names(model: Any) -> list[str]:
    return list(model.get_classes())
