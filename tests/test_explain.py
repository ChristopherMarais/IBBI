"""LIME / SHAP explainers through predict_proba (offline, tiny models)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from PIL import Image

import ibbi
from ibbi.explain._common import class_names, prediction_function


class _Proba:
    """predict_proba returns the mean red value as the score of class 'red' and its complement for 'other'."""

    def predict_proba(self, images, **kw):
        r = np.array([np.asarray(im, dtype=float)[..., 0].mean() / 255 for im in images])
        return np.stack([r, 1 - r], axis=1)

    def get_classes(self):
        return ["red", "other"]


def test_prediction_function_accepts_uint8_and_float():
    f = prediction_function(_Proba())
    img = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    img[0, ..., 0] = 255
    assert np.allclose(f(img)[:, 0], [1.0, 0.0])
    assert np.allclose(f(img.astype(np.float32) / 255)[:, 0], [1.0, 0.0])
    assert f(img[0]).shape == (1, 2)
    assert class_names(_Proba()) == ["red", "other"]


def test_prediction_function_sets_prompts():
    class ZS(_Proba):
        def set_classes(self, c):
            self.c = c

    m = ZS()
    prediction_function(m, text_prompt="beetle . fly")
    assert m.c == "beetle . fly"
    with pytest.raises(TypeError):
        prediction_function(object())


def test_lime_on_tiny_classifier(tiny_classifier):
    img = Image.fromarray((np.random.default_rng(0).random((48, 48, 3)) * 255).astype(np.uint8))
    exp, original = ibbi.Explainer(tiny_classifier).with_lime(img, image_size=(32, 32), num_samples=20, batch_size=10, top_labels=2)
    assert len(exp.top_labels) == 2 and original is img
    ibbi.plot_lime_explanation(exp, img, top_k=1)


def test_shap_on_tiny_classifier(tiny_classifier):
    img = Image.fromarray((np.random.default_rng(1).random((32, 32, 3)) * 255).astype(np.uint8))
    background = [{"image": Image.new("RGB", (32, 32))}]
    values = ibbi.Explainer(tiny_classifier).with_shap([{"image": img}], background, num_explain_samples=1, image_size=(32, 32), max_evals=40)
    assert values.values.shape[:3] == (1, 32, 32)
    assert values.values.shape[-1] == len(tiny_classifier.get_classes())
    ibbi.plot_shap_explanation(values[0], tiny_classifier, top_k=2)
