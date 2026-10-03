"""Model wrappers with local stand-ins (no downloads): classifier maths and records, Ultralytics wrapper, zero-shot
tiling, the pipeline, the registry and the hub override."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

import ibbi
from ibbi.models import classifiers as C
from ibbi.models._common import load_image, nms, resolve_device
from ibbi.pipeline import IdentificationPipeline

LEVELS = C.LEVELS


# --- common helpers -----------------------------------------------------------------------------------------------------
def test_load_image_inputs(tmp_path):
    arr = (np.random.default_rng(0).random((20, 30, 3)) * 255).astype(np.uint8)
    p = tmp_path / "a.png"
    Image.fromarray(arr).save(p)
    for x in (arr, Image.fromarray(arr), str(p), p, arr.astype(np.float32) / 255.0, arr[..., 0]):
        im = load_image(x)
        assert im.mode == "RGB" and im.size == (30, 20)


def test_nms_and_device():
    b = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=float)
    keep = nms(b, np.array([0.9, 0.8, 0.7]), 0.5)
    assert sorted(keep.tolist()) == [0, 2]
    assert len(nms(np.zeros((0, 4)), np.zeros(0), 0.5)) == 0
    assert resolve_device("cpu") == "cpu"
    assert resolve_device() in ("cuda", "mps", "cpu")


def test_hub_override(monkeypatch, tmp_path):
    from ibbi.utils import hub

    (tmp_path / "ibbi_x").mkdir()
    (tmp_path / "ibbi_x" / "config.json").write_text(json.dumps({"a": 1}))
    monkeypatch.setenv("IBBI_MODELS_DIR", str(tmp_path))
    assert hub.get_model_config_from_hub("IBBI-bio/ibbi_x") == {"a": 1}


def test_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("IBBI_CACHE_DIR", str(tmp_path / "c"))
    assert ibbi.get_cache_dir() == tmp_path / "c" and (tmp_path / "c").exists()
    ibbi.clean_cache()
    assert not (tmp_path / "c").exists()


# --- registry -------------------------------------------------------------------------------------------------------------
def test_create_model_resolves_alias(monkeypatch):
    seen = {}
    monkeypatch.setitem(ibbi.model_registry, "dinov3_hierarchical_classifier", lambda pretrained=True, **k: seen.setdefault("k", (pretrained, k)))
    ibbi.create_model("species_classifier", device="cpu")
    assert seen["k"] == (True, {"device": "cpu"})


def test_classifiers_require_weights():
    with pytest.raises(ValueError, match="pretrained=True"):
        ibbi.create_model("dinov3_hierarchical_classifier", pretrained=False)


# --- hierarchical classifier ------------------------------------------------------------------------------------------
def test_taxonomy_maths_consistent(tiny_classifier):
    tax = tiny_classifier.tax
    torch.manual_seed(1)
    z = {lvl: torch.randn(6, tax.n[lvl]) * 3 for lvl in LEVELS}
    for temps in (dict.fromkeys(LEVELS, 1.0), dict.fromkeys(LEVELS, 2.5)):
        m = {k: v.exp() for k, v in tax.marginals(z, temps).items()}
        for lvl in LEVELS:
            assert torch.allclose(m[lvl].sum(1), torch.ones(6), atol=1e-5)
        for s, p in enumerate(tax.path.tolist()):  # P(parent) >= P(child) along every lineage
            for li in range(3):
                assert torch.all(m[LEVELS[li]][:, p[li]] + 1e-6 >= m[LEVELS[li + 1]][:, p[li + 1]])
    tbl = tax.table()
    assert list(tbl.columns) == [*LEVELS, "scientificName"] and len(tbl) == tax.n["species"]


def test_classifier_records(tiny_classifier):
    img = Image.fromarray((np.random.default_rng(0).random((60, 80, 3)) * 255).astype(np.uint8))
    rec = tiny_classifier.predict(img)
    assert set(rec) >= {*LEVELS, "depth", "reported", "depth_by_op"}
    assert 0 <= rec["depth"] <= 4
    assert set(rec["depth_by_op"]) == {"0.90", "0.95", "0.99", "gallery"}
    # top-down: each taxon lies inside its predicted parent
    tbl = tiny_classifier.taxonomy_table
    row = tbl[tbl["scientificName"] == rec["species"]["taxon"]].iloc[0]
    for lvl in ("subfamily", "tribe", "genus"):
        assert row[lvl] == rec[lvl]["taxon"]
    for lvl in LEVELS:
        r = rec[lvl]
        assert 0 <= r["prob"] <= 1 and 0 <= r["score"] <= 1 and r["known"] == (r["score"] >= r["threshold"])
        assert len(r["top3"]) >= 1
    # the reported depth is the longest prefix of known levels
    d = 0
    for lvl in LEVELS:
        if not rec[lvl]["known"]:
            break
        d += 1
    assert rec["depth"] == d
    assert rec["reported"] == C.describe(rec, d)


def test_classifier_batch_boxes_proba_features(tiny_classifier):
    img = Image.fromarray((np.random.default_rng(1).random((90, 120, 3)) * 255).astype(np.uint8))
    recs = tiny_classifier.predict(img, boxes=[[10, 10, 50, 60], [60, 20, 110, 80]])
    assert len(recs) == 2
    assert len(tiny_classifier.predict([img, img])) == 2
    p = tiny_classifier.predict_proba([img, img])
    assert p.shape == (2, len(tiny_classifier.get_classes())) and np.allclose(p.sum(1), 1, atol=1e-4)
    pg = tiny_classifier.predict_proba([img], level="genus")
    assert pg.shape[1] == tiny_classifier.tax.n["genus"]
    assert tuple(tiny_classifier.extract_features(img).shape) == (1, 384)


def test_operating_points(tiny_classifier):
    img = Image.fromarray((np.random.default_rng(2).random((40, 40, 3)) * 255).astype(np.uint8))
    lenient = tiny_classifier.predict(img, operating_point="0.99")
    strict = tiny_classifier.predict(img, operating_point="0.90")
    assert lenient["depth"] >= strict["depth"]  # lower thresholds never reduce the depth
    with pytest.raises(KeyError):
        tiny_classifier.predict(img, operating_point="0.5")


def test_unknown_default_operating_point(tiny_classifier_files):
    cfg = json.loads((tiny_classifier_files / "config.json").read_text())
    with pytest.raises(ValueError):
        C.HierarchicalClassifier(
            cfg,
            str(tiny_classifier_files / "model.safetensors"),
            str(tiny_classifier_files / "deploy.safetensors"),
            device="cpu",
            operating_point="0.42",
        )


def test_crop_padding(tiny_classifier):
    img = Image.new("RGB", (200, 100))
    assert tiny_classifier.crop(img, (50, 20, 150, 80)).size == (110, 66)  # box + 5% each side
    assert tiny_classifier.crop(img, (0, 0, 200, 100)).size == (200, 100)  # clipped to the image
    assert tiny_classifier.crop(img, (10, 10, 10.5, 10.5)).size == (200, 100)  # degenerate box -> whole image


def test_describe_all_depths():
    rec = {lvl: {"taxon": t} for lvl, t in zip(LEVELS, ["Scolytinae", "Xyleborini", "Xyleborus", "Xyleborus volvulus"], strict=True)}
    assert C.describe(rec, 0).startswith("unrecognised")
    assert C.describe(rec, 1) == "Scolytinae (tribe undetermined)"
    assert C.describe(rec, 2) == "Xyleborini (genus undetermined)"
    assert C.describe(rec, 3) == "Xyleborus sp. (species undetermined)"
    assert C.describe(rec, 4) == "Xyleborus volvulus"


def test_load_through_registry(monkeypatch, tiny_classifier_files):
    """The registered factory reads config / weights / deploy files by repository name."""
    monkeypatch.setenv("IBBI_MODELS_DIR", str(tiny_classifier_files.parent))
    clf = C._load_classifier("tiny", tiny_classifier_files.name, "cpu", None, None)
    assert (
        clf.get_classes()
        == C.HierarchicalClassifier(
            json.loads((tiny_classifier_files / "config.json").read_text()),
            str(tiny_classifier_files / "model.safetensors"),
            str(tiny_classifier_files / "deploy.safetensors"),
            device="cpu",
        ).get_classes()
    )


# --- Ultralytics wrapper ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_species_detector(tmp_path_factory, known_species):
    """An untrained YOLOv8n built from its yaml (no download), with species class names."""
    import yaml
    from ultralytics import YOLO
    from ultralytics.utils import ROOT

    from ibbi.models.detectors import SpeciesDetector

    d = tmp_path_factory.mktemp("det")
    cfg = yaml.safe_load((ROOT / "cfg" / "models" / "v8" / "yolov8.yaml").read_text())
    cfg["nc"] = len(known_species)
    (d / "yolov8n.yaml").write_text(yaml.safe_dump(cfg))
    m = YOLO(str(d / "yolov8n.yaml"))
    m.model.names = dict(enumerate(known_species))
    p = d / "tiny.pt"
    m.save(str(p))
    return SpeciesDetector(str(p), config={"imgsz": 64, "inference_defaults": {"conf": 0.0, "iou": 0.7, "max_det": 10}}, device="cpu", name="tiny")


def test_detector_outputs(tiny_species_detector, known_species):
    img = Image.fromarray((np.random.default_rng(0).random((64, 96, 3)) * 255).astype(np.uint8))
    r = tiny_species_detector.predict(img)
    assert set(r) >= {"boxes", "scores", "labels", "class_ids", "species"}
    assert len(r["boxes"]) == len(r["scores"]) == len(r["labels"]) <= 10
    assert all(lbl in known_species for lbl in r["labels"])
    assert isinstance(tiny_species_detector.predict([img, img]), list)
    g = tiny_species_detector.predict(img, level="genus")
    assert g["labels"] == [s.split()[0] for s in g["species"]]
    p = tiny_species_detector.predict_proba([img])
    assert p.shape == (1, len(known_species)) and (p >= 0).all() and (p <= 1).all()
    assert tiny_species_detector.get_classes() == known_species
    assert tiny_species_detector.is_species_level and tiny_species_detector.benchmark_kwargs["conf"] == 0.001


def test_ultralytics_version_warning(monkeypatch):
    import ultralytics

    from ibbi.models import detectors

    monkeypatch.setattr(ultralytics, "__version__", "8.4.0")
    with pytest.warns(UserWarning, match="8.3"):
        detectors._check_ultralytics_version()


# --- zero-shot base class -----------------------------------------------------------------------------------------------
class _FakeZeroShot(ibbi.models.ZeroShotDetector):
    """Finds a 'beetle' at a fixed place of every window it is given."""

    def __init__(self, **kw):
        super().__init__(prompts=["beetle", "fly"], **kw)
        self.windows = []

    def _detect(self, img, conf):
        self.windows.append(img.size)
        return np.array([[2.0, 2.0, 12.0, 12.0]]), np.array([0.8]), np.array([0])


def test_zero_shot_whole_image_and_prompts():
    zs = _FakeZeroShot(tile=0, device="cpu")
    r = zs.predict(Image.new("RGB", (300, 200)))
    assert r["labels"] == ["beetle"] and r["boxes"] == [[2.0, 2.0, 12.0, 12.0]]
    zs.set_classes("moth . bee")
    assert zs.get_classes() == ["moth", "bee"]
    assert zs.predict(Image.new("RGB", (50, 50)), text_prompt=["ant"])["labels"] == ["ant"]
    assert zs.predict_proba([Image.new("RGB", (50, 50))]).shape == (1, 1)
    with pytest.raises(NotImplementedError):
        zs.extract_features(Image.new("RGB", (5, 5)))


def test_zero_shot_tiling_offsets():
    zs = _FakeZeroShot(tile=100, device="cpu")
    r = zs.predict(Image.new("RGB", (250, 120)))
    assert len(zs.windows) > 1 and (250, 120) in zs.windows  # windows plus the whole image
    xs = sorted(b[0] for b in r["boxes"])
    assert xs[0] == 2.0 and xs[-1] > 100  # detections shifted back into image coordinates
    assert all(0 <= b[2] <= 250 and 0 <= b[3] <= 120 for b in r["boxes"])
    small = _FakeZeroShot(tile=1000, device="cpu")
    small.predict(Image.new("RGB", (250, 120)))
    assert small.windows == [(250, 120)]  # image smaller than a tile: whole image only


# --- pipeline ------------------------------------------------------------------------------------------------------------
def test_pipeline_with_fakes(fake_box_detector, tiny_classifier):
    pipe = IdentificationPipeline(fake_box_detector, tiny_classifier, det_conf=0.5)
    img = Image.fromarray((np.random.default_rng(3).random((96, 128, 3)) * 255).astype(np.uint8))
    r = pipe.predict(img)
    assert len(r["boxes"]) == len(r["labels"]) == len(r["species"]) == len(r["classifications"]) == 3
    assert all(abs(s - d * c["species"]["prob"]) < 1e-6 for s, d, c in zip(r["scores"], r["det_scores"], r["classifications"], strict=True))
    assert pipe.predict(img, det_conf=0.85)["det_scores"] == [0.9]
    assert len(pipe.predict([img, img])) == 2
    assert pipe.predict_proba([img]).shape == (1, len(tiny_classifier.get_classes()))
    assert pipe.get_classes() == tiny_classifier.get_classes()
    assert pipe.is_species_level and "det_conf" in pipe.benchmark_kwargs


def test_pipeline_no_detections(tiny_classifier):
    class Empty:
        def predict(self, image, **k):
            return {"boxes": [], "scores": [], "labels": []}

    r = IdentificationPipeline(Empty(), tiny_classifier).predict(Image.new("RGB", (32, 32)))
    assert r["boxes"] == [] and r["classifications"] == []
    assert IdentificationPipeline(Empty(), tiny_classifier).predict_proba([Image.new("RGB", (32, 32))]).sum() == 0
