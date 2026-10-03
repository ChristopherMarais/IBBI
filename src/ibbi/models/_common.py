# src/ibbi/models/_common.py

"""Shared helpers for the model wrappers: image loading, device choice and output conventions.

Every detector's `predict` returns, per image, a dictionary with
    "boxes"  : [[x1, y1, x2, y2], ...] absolute pixels
    "scores" : [float, ...] confidence of each box
    "labels" : [str, ...] class name (species, "arthropod", or the matching text prompt)
and every model implements `get_classes()`, `predict_proba(images) -> np.ndarray [N, n_classes]` (used by LIME and
SHAP) and `extract_features(image)`.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

ImageInput = Union[str, Path, np.ndarray, Image.Image]


def resolve_device(device: str | torch.device | None = None) -> str:
    """Returns `device` if given, else "cuda" when available, else "mps" on Apple silicon, else "cpu"."""
    if device is not None:
        return str(device)
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(image: ImageInput) -> Image.Image:
    """Loads a file path, URL, numpy array (RGB, HxWx3) or PIL image as an RGB PIL image (EXIF orientation applied)."""
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.dtype != np.uint8:
            image = np.clip(image * (255.0 if image.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")
    s = str(image)
    if s.startswith(("http://", "https://")):
        import requests

        r = requests.get(s, timeout=60)
        r.raise_for_status()
        im = Image.open(BytesIO(r.content))
    else:
        im = Image.open(s)
    im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":
        if im.mode.startswith("I;16") or im.mode in ("I", "F"):
            arr = np.asarray(im, dtype=np.float32)
            arr = (255 * (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6)).astype(np.uint8)
            im = Image.fromarray(arr)
        im = im.convert("RGB")
    return im


def is_batch(images: Any) -> bool:
    """True for a list/tuple of images (a single image may itself be a numpy array)."""
    return isinstance(images, (list, tuple))


def empty_result() -> dict[str, list]:
    return {"boxes": [], "scores": [], "labels": []}


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Class-agnostic non-maximum suppression; returns kept indices."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    import torchvision

    keep = torchvision.ops.nms(torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(scores, dtype=torch.float32), iou)
    return keep.numpy()
