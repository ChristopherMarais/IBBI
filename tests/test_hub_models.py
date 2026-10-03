"""Slow tests: download every model from the Hugging Face Hub and run it on a benchmark image.

    pytest --run-slow tests/test_hub_models.py

Checks that the public repositories, their configs and the wrappers fit together. SAM 3 is gated and is skipped when the
logged-in account has no access.
"""

import numpy as np
import pytest

import ibbi

pytestmark = pytest.mark.slow

DETECTORS = [
    "yolov8x_species_detector",
    "yolov9e_species_detector",
    "yolov10x_species_detector",
    "yolo11x_species_detector",
    "yolo12x_species_detector",
    "rtdetrx_species_detector",
    "yolo11x_arthropod_detector",
]
ZERO_SHOT = ["grounding_dino_zero_shot_detector", "owlv2_zero_shot_detector", "yoloworld_zero_shot_detector", "sam3_zero_shot_detector"]
CLASSIFIERS = ["dinov3_hierarchical_classifier", "bioclip2_hierarchical_classifier"]


@pytest.fixture(scope="module")
def sample():
    """One iid_test image (single file download) and its scored specimen box (xyxy)."""
    import json

    from huggingface_hub import hf_hub_download

    from ibbi.utils.data import BENCHMARK_REPO_ID, BENCHMARK_REVISION, _load_rgb

    ann = json.load(
        open(hf_hub_download(BENCHMARK_REPO_ID, "detection/annotations_coco/iid_test.json", repo_type="dataset", revision=BENCHMARK_REVISION))
    )
    a = next(a for a in ann["annotations"] if not a["iscrowd"])
    im = next(i for i in ann["images"] if i["id"] == a["image_id"])
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    p = hf_hub_download(
        BENCHMARK_REPO_ID, f"detection/images/iid_test/{im['file_name'].split('/')[-1]}", repo_type="dataset", revision=BENCHMARK_REVISION
    )
    x, y, w, h = a["bbox"]
    return _load_rgb(p, (im["width"], im["height"])), [x, y, x + w, y + h], cats[a["category_id"]]


@pytest.mark.parametrize("name", DETECTORS)
def test_detector(name, sample):
    img, box, species = sample
    m = ibbi.create_model(name)
    r = m.predict(img)
    assert len(r["boxes"]) >= 1, "a specimen photo should give at least one detection"
    best = int(np.argmax(r["scores"]))
    bx = r["boxes"][best]
    iw, ih = max(0, min(bx[2], box[2]) - max(bx[0], box[0])), max(0, min(bx[3], box[3]) - max(bx[1], box[1]))
    inter = iw * ih
    union = (bx[2] - bx[0]) * (bx[3] - bx[1]) + (box[2] - box[0]) * (box[3] - box[1]) - inter
    assert inter / union > 0.5, "the top detection should cover the annotated specimen"
    if m.is_species_level:
        assert len(m.get_classes()) == 65 and all(lbl in m.get_classes() for lbl in r["labels"])
    else:
        assert m.get_classes() == ["arthropod"] and 0 < m.operating_conf < 1
    assert m.predict_proba([img]).shape == (1, len(m.get_classes()))
    assert m.extract_features(img) is not None


@pytest.mark.parametrize("name", CLASSIFIERS)
def test_classifier(name, sample):
    img, box, species = sample
    clf = ibbi.create_model(name)
    assert len(clf.get_classes()) == 65 and species in clf.get_classes()
    rec = clf.predict(img, boxes=[box])[0]
    assert rec["subfamily"]["taxon"] in {"Scolytinae", "Platypodinae"}
    assert rec["depth"] >= 1, "a known species on a curated photo should be recognised at least at subfamily level"
    assert clf.extract_features(clf.crop(img, box)).shape[0] == 1


@pytest.mark.parametrize("name", ZERO_SHOT)
def test_zero_shot(name, sample):
    img, box, _ = sample
    try:
        m = ibbi.create_model(name, tile=0)
    except OSError as e:
        if "gated" in str(e) or "sam3" in name:
            pytest.skip(f"no access to the gated model: {e}")
        raise
    r = m.predict(img)
    assert len(r["boxes"]) >= 1 and all(lbl in m.get_classes() for lbl in r["labels"])


def test_pipeline_and_aliases(sample):
    img, box, species = sample
    pipe = ibbi.create_pipeline()
    r = pipe.predict(img)
    assert len(r["boxes"]) >= 1 and len(r["classifications"]) == len(r["boxes"])
    for alias in ("beetle_detector", "species_detector", "feature_extractor"):
        assert ibbi.create_model(alias) is not None


def test_small_benchmark_split():
    """get_dataset downloads a real split (inat_test: 74 images) and the evaluator scores a real model on it."""
    ds = ibbi.get_dataset("inat_test")
    assert len(ds) == 74
    res = ibbi.Evaluator(ibbi.create_model("species_detector")).benchmark(splits=["inat_test"])
    assert 0 <= res["headline"]["inat_test.AP_50"] <= 1
