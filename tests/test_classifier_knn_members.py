"""Novelty members 'knn<k>' with any k (used by dinov3_400species_hierarchical_classifier: knn1) and a 'strict' operating point."""

import json
import shutil

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

LEVELS = ("subfamily", "tribe", "genus", "species")


def _variant(tiny_classifier_files, tmp_path, members, extra_op=None):
    d = tmp_path / "variant"
    d.mkdir()
    shutil.copy(tiny_classifier_files / "model.safetensors", d / "model.safetensors")
    dep = load_file(str(tiny_classifier_files / "deploy.safetensors"))
    tensors = {"knn_bank": dep["knn_bank"]}
    rng = np.random.default_rng(1)
    for lvl in LEVELS:
        for m in members[lvl]:
            tensors[f"valsorted.{m}.{lvl}"] = torch.from_numpy(np.sort(rng.random(50)).astype(np.float32))
    save_file(tensors, str(d / "deploy.safetensors"))
    cfg = json.loads((tiny_classifier_files / "config.json").read_text(encoding="utf-8"))
    cfg["novelty"]["members"] = members
    if extra_op:
        cfg["novelty"]["thresholds"][extra_op] = dict.fromkeys(LEVELS, 0.5)
        cfg["novelty"]["default_op"] = extra_op
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d, cfg


def _load(d, cfg):
    from ibbi.models.classifiers import HierarchicalClassifier

    return HierarchicalClassifier(cfg, str(d / "model.safetensors"), str(d / "deploy.safetensors"), device="cpu", name="variant")


def test_knn1_matches_nearest_neighbour(tiny_classifier_files, tmp_path):
    members = {"subfamily": ["knn1"], "tribe": ["msp"], "genus": ["msp", "knn3"], "species": ["knn1"]}
    d, cfg = _variant(tiny_classifier_files, tmp_path, members)
    clf = _load(d, cfg)
    crop = Image.fromarray(np.random.default_rng(0).integers(0, 255, (48, 40, 3), dtype=np.uint8))
    emb, z = clf._forward([crop])
    marg = {lvl: v.numpy() for lvl, v in clf.tax.marginals(z, clf.temps).items()}
    nov = clf._novelty(emb, marg)
    sim = torch.nn.functional.normalize(emb, dim=1) @ clf.bank.T
    k1 = float(sim.max(1).values[0])
    vs = clf.valsorted[("knn1", "species")]
    assert np.isclose(nov["species"][0], np.searchsorted(vs, k1, side="right") / len(vs))
    rec = clf.predict(crop)
    assert 0 <= rec["depth"] <= 4 and set(LEVELS) <= set(rec)


def test_strict_operating_point_is_default(tiny_classifier_files, tmp_path):
    members = {"subfamily": ["msp"], "tribe": ["msp"], "genus": ["msp"], "species": ["msp", "knn1"]}
    d, cfg = _variant(tiny_classifier_files, tmp_path, members, extra_op="strict")
    clf = _load(d, cfg)
    assert clf.operating_point == "strict"
    crop = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
    rec = clf.predict(crop)
    assert set(rec["depth_by_op"]) >= {"strict", "gallery"}
