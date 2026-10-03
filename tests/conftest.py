"""Shared fixtures.

Fast tests (the default) use tiny local stand-ins: a synthetic four-split benchmark, a tiny ViT hierarchical classifier
with the same file layout as the released ones, and fake models. Tests marked `slow` download the real weights from
the Hugging Face Hub and run them; enable them with `pytest --run-slow` (a GPU is recommended).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

LEVELS = ("subfamily", "tribe", "genus", "species")


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run tests that download models from the Hub")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: downloads model weights or data from the Hugging Face Hub")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def taxonomy():
    from ibbi.utils.data import get_taxonomy

    return get_taxonomy()


@pytest.fixture(scope="session")
def known_species(taxonomy):
    """Four trainable species: two congeners, one from another genus of the same tribe if possible, one from another tribe."""
    t = taxonomy[taxonomy["benchmark_role"] == "trainable"].sort_values(["tribe", "genus", "scientificName"])
    g = t.groupby("genus").filter(lambda d: len(d) >= 2)
    a, b = g.iloc[0], g.iloc[1]
    other_tribe = t[t["tribe"] != a["tribe"]].iloc[0]
    other_genus = t[(t["tribe"] == a["tribe"]) & (t["genus"] != a["genus"])]
    c = other_genus.iloc[0] if len(other_genus) else t[t["genus"] != a["genus"]].iloc[0]
    rows = [a, b, c, other_tribe]
    return [r["scientificName"] for r in rows]


@pytest.fixture(scope="session")
def held_out_species(taxonomy):
    t = taxonomy[taxonomy["benchmark_role"] == "semantic_ood"]
    return [t[t["distance_band_vs_trainable"] == b].iloc[0]["scientificName"] for b in ("near_genus", "mid_tribe", "far_tribe")]


# ----------------------------------------------------------------------------------------------------------------------
def _write_split(root: Path, split: str, images: list, anns: list, cats: list):
    (root / "detection" / "annotations_coco").mkdir(parents=True, exist_ok=True)
    img_dir = root / "detection" / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    for im in images:
        rng = np.random.default_rng(im["id"])
        arr = (rng.random((im["height"], im["width"], 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(img_dir / im["file_name"])
    coco = {"images": images, "annotations": anns, "categories": cats}
    (root / "detection" / "annotations_coco" / f"{split}.json").write_text(json.dumps(coco))


@pytest.fixture(scope="session")
def tiny_benchmark(tmp_path_factory, taxonomy, known_species, held_out_species):
    """A four-split benchmark with the real layout: 4 known species (train / iid_test / inat_test) and 3 held-out ones.

    Every image is 128 x 96 with one scored specimen at [20, 20, 40, 30] (x, y, w, h); iid_test images also carry a
    crowd specimen at [80, 50, 30, 30].
    """
    import shutil
    from importlib import resources

    root = tmp_path_factory.mktemp("benchmark")
    shutil.copy(resources.files("ibbi.data").joinpath("species_taxonomy.csv"), root / "species_taxonomy.csv")
    all_species = known_species + held_out_species
    cat_id = {s: i + 1 for i, s in enumerate(all_species)}
    known_cats = [{"id": cat_id[s], "name": s} for s in known_species]
    ood_cats = [{"id": cat_id[s], "name": s} for s in held_out_species]
    aid = 1

    def make(split, species, n_per, crowd=False, start=1):
        nonlocal aid
        images, anns = [], []
        iid = start
        for s in species:
            for _ in range(n_per):
                images.append({"id": iid, "file_name": f"{split}_{iid}.jpg", "width": 128, "height": 96})
                anns.append({"id": aid, "image_id": iid, "category_id": cat_id[s], "bbox": [20, 20, 40, 30], "area": 1200, "iscrowd": 0})
                aid += 1
                if crowd:
                    anns.append({"id": aid, "image_id": iid, "category_id": cat_id[s], "bbox": [80, 50, 30, 30], "area": 900, "iscrowd": 1})
                    aid += 1
                iid += 1
        return images, anns

    for split, species, n, crowd, cats in (
        ("train", known_species, 2, False, known_cats),
        ("iid_test", known_species, 2, True, known_cats),
        ("inat_test", known_species[:2], 1, False, known_cats[:2]),
        ("semantic_ood", held_out_species, 2, False, ood_cats),
    ):
        images, anns = make(split, species, n, crowd)
        _write_split(root, split, images, anns, cats)
    return root


# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tiny_classifier_files(tmp_path_factory, taxonomy, known_species):
    """A tiny DINO-style hierarchical classifier saved exactly like the released repositories (random weights)."""
    from safetensors.torch import save_file

    from ibbi.models.classifiers import _HierNet

    t = taxonomy.drop_duplicates("scientificName").set_index("scientificName")
    rows = sorted([(t.at[s, "subfamily"], t.at[s, "tribe"], t.at[s, "genus"], s) for s in known_species])
    nodes = {lvl: [] for lvl in LEVELS}
    parent = {lvl: [] for lvl in LEVELS}
    index = {lvl: {} for lvl in LEVELS}
    path = []
    for names in rows:
        idx = []
        for li, lvl in enumerate(LEVELS):
            key = names[: li + 1]
            if key not in index[lvl]:
                index[lvl][key] = len(nodes[lvl])
                nodes[lvl].append(names[li])
                parent[lvl].append(idx[li - 1] if li else -1)
            idx.append(index[lvl][key])
        path.append(idx)
    sizes = [len(nodes[lvl]) for lvl in LEVELS]
    spec = {"kind": "timm", "name": "vit_tiny_patch16_224", "patch": 16, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    res = 32
    torch.manual_seed(0)
    net = _HierNet(spec, res, 384, sizes)
    d = tmp_path_factory.mktemp("tiny_classifier") / "ibbi_tiny_hierarchical_classifier"
    d.mkdir()
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, str(d / "model.safetensors"))
    members = {"subfamily": ["msp"], "tribe": ["msp"], "genus": ["msp"], "species": ["msp", "knn5"]}
    bank = torch.nn.functional.normalize(torch.randn(20, 384), dim=1).half()
    tensors = {"knn_bank": bank}
    rng = np.random.default_rng(0)
    for lvl in LEVELS:
        for m in members[lvl]:
            tensors[f"valsorted.{m}.{lvl}"] = torch.from_numpy(np.sort(rng.random(50)).astype(np.float32))
    save_file(tensors, str(d / "deploy.safetensors"))
    # operating point keys as in the released configs ("0.9", not "0.90")
    thr = {op: dict.fromkeys(LEVELS, v) for op, v in (("0.9", 0.1), ("0.95", 0.05), ("0.99", 0.01))}
    thr["gallery"] = {lvl: (0.01 if lvl == "subfamily" else 0.05) for lvl in LEVELS}
    cfg = {
        "ibbi_model_type": "hierarchical_classifier",
        "backbone": spec,
        "res": res,
        "embedding_dim": 384,
        "crop_pad": 0.05,
        "taxonomy": {"nodes": nodes, "parent": parent, "path": path},
        "temperatures": dict.fromkeys(LEVELS, 1.0),
        "novelty": {"members": members, "thresholds": thr, "default_op": "gallery"},
    }
    (d / "config.json").write_text(json.dumps(cfg))
    return d


@pytest.fixture(scope="session")
def tiny_classifier(tiny_classifier_files):
    from ibbi.models.classifiers import HierarchicalClassifier

    d = tiny_classifier_files
    cfg = json.loads((d / "config.json").read_text())
    return HierarchicalClassifier(cfg, str(d / "model.safetensors"), str(d / "deploy.safetensors"), device="cpu", name="tiny")


# ----------------------------------------------------------------------------------------------------------------------
class FakeSpeciesDetector:
    """Predicts the true box of the benchmark fixture with a fixed species (always the first known species)."""

    is_species_level = True
    benchmark_kwargs = {"conf": 0.001}

    def __init__(self, species):
        self.species = species
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return {"boxes": [[20.0, 20.0, 60.0, 50.0]], "scores": [0.9], "labels": [self.species], "species": [self.species]}

    def get_classes(self):
        return [self.species]


class FakeBoxDetector:
    """Class-agnostic: one box on the scored specimen, one on the crowd region, one false alarm."""

    is_species_level = False
    benchmark_kwargs = {"conf": 0.01}
    operating_conf = 0.5

    def predict(self, image, conf=0.25, **kwargs):
        boxes = [[20.0, 20.0, 60.0, 50.0], [80.0, 50.0, 110.0, 80.0], [0.0, 0.0, 10.0, 10.0]]
        scores = [0.9, 0.8, 0.6]
        keep = [i for i, s in enumerate(scores) if s >= conf]
        return {"boxes": [boxes[i] for i in keep], "scores": [scores[i] for i in keep], "labels": ["arthropod"] * len(keep)}

    def predict_proba(self, images, **kwargs):
        return np.full((len(images), 1), 0.9, dtype=np.float32)

    def get_classes(self):
        return ["arthropod"]


@pytest.fixture()
def fake_species_detector(known_species):
    return FakeSpeciesDetector(known_species[0])


@pytest.fixture()
def fake_box_detector():
    return FakeBoxDetector()
