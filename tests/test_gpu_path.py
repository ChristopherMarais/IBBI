"""Fast path helpers (models/_gpu.py) against the reference implementations they replace. Run on the CPU here; the
same code runs on CUDA."""

import numpy as np
import pytest
import torch
from PIL import Image

from ibbi.models._common import load_image
from ibbi.models._gpu import _exif_orient, crop_letterbox, load_tensor, ultralytics_letterbox
from ibbi.models.classifiers import _letterbox
from ibbi.pipeline import IdentificationPipeline


def _rand_img(h, w, seed=0):
    rng = np.random.default_rng(seed)
    # smooth image (JPEG-like content) so resize rounding differences stay small
    base = rng.random((h // 8 + 2, w // 8 + 2, 3))
    return np.asarray(Image.fromarray((base * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR))


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_matches_pillow(tmp_path, orientation):
    arr = _rand_img(40, 64)
    exif = Image.Exif()
    exif[0x0112] = orientation
    p = tmp_path / f"o{orientation}.jpg"
    Image.fromarray(arr).save(p, quality=95, exif=exif)
    ref = np.asarray(load_image(p))  # PIL decode + exif_transpose
    decoded = np.asarray(Image.open(p).convert("RGB"))  # raw decode, no orientation
    got = _exif_orient(torch.from_numpy(decoded.copy()).permute(2, 0, 1), p).permute(1, 2, 0).numpy()
    assert got.shape == ref.shape
    assert np.array_equal(got, ref)
    t = load_tensor(p, "cpu")  # torchvision decode with orientation
    assert tuple(t.shape[1:]) == ref.shape[:2]
    assert np.abs(t.permute(1, 2, 0).numpy().astype(int) - ref.astype(int)).mean() < 2


def test_load_tensor_inputs():
    arr = _rand_img(30, 50)
    for x in (arr, Image.fromarray(arr), torch.from_numpy(arr.copy()).permute(2, 0, 1)):
        t = load_tensor(x, "cpu")
        assert t.dtype == torch.uint8 and tuple(t.shape) == (3, 30, 50)
        assert np.array_equal(t.permute(1, 2, 0).numpy(), arr)


@pytest.mark.parametrize("hw", [(480, 640), (3456, 5184), (853, 640), (640, 640), (100, 1500)])
def test_letterbox_matches_ultralytics(hw):
    from ultralytics.data.augment import LetterBox

    arr = _rand_img(*hw)
    ref = LetterBox(1024, auto=True, stride=32)(image=arr)  # HWC uint8
    got = ultralytics_letterbox(torch.from_numpy(arr.copy()).permute(2, 0, 1), 1024, 32)
    assert got.shape[2:] == ref.shape[:2]
    diff = np.abs(np.round(got[0].permute(1, 2, 0).numpy() * 255).astype(int) - ref.astype(int))
    assert diff.mean() < 0.5 and diff.max() <= 4  # rounding of the bilinear resize only


def test_crop_letterbox_matches_classifier_crop():
    arr = _rand_img(600, 900)
    boxes = [[100.0, 50.0, 300.0, 400.0], [700.0, 500.0, 890.0, 595.0], [0.0, 0.0, 1.0, 1.0]]
    pad, res, fill = 0.05, 64, (124, 116, 104)
    got = crop_letterbox(torch.from_numpy(arr.copy()).permute(2, 0, 1), boxes, pad, res, fill)
    assert tuple(got.shape) == (3, 3, res, res)

    class _C:  # the reference crop of HierarchicalClassifier, without loading a model
        from ibbi.models.classifiers import HierarchicalClassifier as H

        crop = H.crop
        pad = 0.05

    im = Image.fromarray(arr)
    for k, b in enumerate(boxes):
        ref = np.asarray(_letterbox(_C().crop(im, b), res, fill)).astype(int)
        diff = np.abs(got[k].permute(1, 2, 0).numpy().astype(int) - ref)
        assert diff.mean() < 1.5, (k, diff.mean())


def test_fast_pipeline_matches_reference(fake_box_detector, tiny_classifier):
    img = Image.fromarray(_rand_img(96, 128, seed=3))
    ref = IdentificationPipeline(fake_box_detector, tiny_classifier, det_conf=0.5, fast=False).predict([img, img])
    fast = IdentificationPipeline(fake_box_detector, tiny_classifier, det_conf=0.5, fast=True, batch_size=4).predict([img, img])
    assert len(fast) == 2
    for a, b in zip(ref, fast):
        assert a["boxes"] == b["boxes"] and a["det_scores"] == b["det_scores"]
        assert len(b["classifications"]) == len(a["classifications"]) == 3
        for ra, rb in zip(a["classifications"], b["classifications"]):
            assert abs(ra["species"]["prob"] - rb["species"]["prob"]) < 0.05
    single = IdentificationPipeline(fake_box_detector, tiny_classifier, det_conf=0.5, fast=True).predict(img)
    assert single["boxes"] == ref[0]["boxes"]
    assert IdentificationPipeline(fake_box_detector, tiny_classifier).fast is False  # CPU classifier: reference path


def test_detector_fast_path_on_cpu(tmp_path):
    """The Ultralytics GPU letterbox path also runs on the CPU (untrained YOLO, just geometry and mapping)."""
    from ibbi.models.detectors import UltralyticsDetector

    det_ref = UltralyticsDetector("yolov8n.yaml", config={"imgsz": 320}, device="cpu", fast=False)
    det_fast = UltralyticsDetector("yolov8n.yaml", config={"imgsz": 320}, device="cpu", fast=True)
    det_fast.model = det_ref.model
    img = Image.fromarray(_rand_img(240, 400))
    a = det_ref.predict(img, conf=0.0, max_det=20)
    b = det_fast.predict(img, conf=0.0, max_det=20)
    assert set(a) == set(b)
    if a["boxes"] and b["boxes"]:
        ba, bb = np.asarray(a["boxes"]), np.asarray(b["boxes"])
        assert ba[:, [0, 2]].max() <= 400.5 and bb[:, [0, 2]].max() <= 400.5
