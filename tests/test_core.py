"""Fast CPU tests: no network, no model weights."""

import json

import numpy as np
import pytest
import torch

import ibbi
from ibbi.evaluate.benchmark import evaluate_class_agnostic, to_coco_results
from ibbi.evaluate.hierarchical import evaluate_hierarchical_records
from ibbi.models.classifiers import LEVELS, Taxonomy, describe
from ibbi.utils.data import BenchmarkDataset, get_taxonomy, taxonomic_distance_matrix

EXPECTED_MODELS = {
    "yolov8x_species_detector",
    "yolov9e_species_detector",
    "yolov10x_species_detector",
    "yolo11x_species_detector",
    "yolo12x_species_detector",
    "rtdetrx_species_detector",
    "yolo11x_arthropod_detector",
    "codino_arthropod_detector",
    "grounding_dino_zero_shot_detector",
    "owlv2_zero_shot_detector",
    "yoloworld_zero_shot_detector",
    "sam3_zero_shot_detector",
    "dinov3_hierarchical_classifier",
    "bioclip2_hierarchical_classifier",
}


def test_registry_and_aliases():
    assert set(ibbi.model_registry) == EXPECTED_MODELS
    for alias, target in ibbi.MODEL_ALIASES.items():
        assert target in ibbi.model_registry, alias
    with pytest.raises(KeyError):
        ibbi.create_model("not_a_model")


def test_list_models_matches_registry():
    df = ibbi.list_models(as_df=True)
    assert set(df["Model Name"]) == EXPECTED_MODELS


def test_taxonomy_table():
    t = get_taxonomy()
    assert len(t) == 175
    assert (t["benchmark_role"] == "trainable").sum() == 65
    d = taxonomic_distance_matrix(["Xyleborus volvulus", "Xyleborus ferrugineus", "Ips acuminatus"])
    assert d.loc["Xyleborus volvulus", "Xyleborus ferrugineus"] == 1
    assert d.loc["Xyleborus volvulus", "Xyleborus volvulus"] == 0
    assert d.loc["Xyleborus volvulus", "Ips acuminatus"] == 3  # same subfamily, different tribe


def _tiny_tax():
    # two subfamilies; subfamily 0 has two tribes; species path = (subfamily, tribe, genus, species)
    return Taxonomy(
        {
            "nodes": {"subfamily": ["A", "B"], "tribe": ["a1", "a2", "b1"], "genus": ["g1", "g2", "g3"], "species": ["s1", "s2", "s3", "s4"]},
            "parent": {"subfamily": [-1, -1], "tribe": [0, 0, 1], "genus": [0, 1, 2], "species": [0, 0, 1, 2]},
            "path": [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 2], [1, 2, 2, 3]],
        }
    )


def test_hierarchical_marginals_are_coherent():
    tax = _tiny_tax()
    torch.manual_seed(0)
    logits = {lvl: torch.randn(5, tax.n[lvl]) for lvl in LEVELS}
    marg = tax.marginals(logits, dict.fromkeys(LEVELS, 1.0))
    for lvl in LEVELS:
        assert torch.allclose(marg[lvl].exp().sum(1), torch.ones(5), atol=1e-5)
    # P(genus) >= P(species) along every path
    p_g, p_s = marg["genus"].exp(), marg["species"].exp()
    for s, path in enumerate(tax.path.tolist()):
        assert torch.all(p_g[:, path[2]] + 1e-6 >= p_s[:, s])


def test_describe():
    rec = {lvl: {"taxon": t} for lvl, t in zip(LEVELS, ["Scolytinae", "Xyleborini", "Xyleborus", "Xyleborus volvulus"])}
    assert describe(rec, 4) == "Xyleborus volvulus"
    assert describe(rec, 3).startswith("Xyleborus sp.")
    assert describe(rec, 0).startswith("unrecognised")


@pytest.fixture()
def tiny_benchmark(tmp_path):
    ann = tmp_path / "detection" / "annotations_coco"
    img_dir = tmp_path / "detection" / "images" / "iid_test"
    ann.mkdir(parents=True)
    img_dir.mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (100, 100), (200, 200, 200)).save(img_dir / "a.jpg")
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 7, "name": "Xyleborus volvulus"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 7, "bbox": [60, 60, 20, 20], "area": 400, "iscrowd": 1},
        ],
    }
    (ann / "iid_test.json").write_text(json.dumps(coco))
    return tmp_path


def test_benchmark_dataset(tiny_benchmark):
    ds = BenchmarkDataset(tiny_benchmark, "iid_test")
    assert len(ds) == 1
    item = ds[0]
    assert item["image"].size == (100, 100)
    assert item["objects"]["iscrowd"] == [0, 1]
    assert item["objects"]["genus"][0] == "Xyleborus"


def test_class_agnostic_scoring_ignores_crowd(tiny_benchmark):
    preds = to_coco_results(1, [[10, 10, 30, 30], [60, 60, 80, 80]], [0.9, 0.8])
    res = evaluate_class_agnostic({"iid_test": preds}, tiny_benchmark, operating_conf=0.5)
    r = res["iid_test"]
    assert r["n_gt"] == 1  # the crowd specimen is not a target
    assert r["recall_50_at_op"] == 1.0
    assert r["precision_50_at_op"] == 1.0  # the detection on the crowd region is neither TP nor FP
    assert r["fp_per_image_at_op"] == 0.0


def test_hierarchical_metrics_over_commit():
    known = get_taxonomy()
    known = known[known["benchmark_role"] == "trainable"][["subfamily", "tribe", "genus", "scientificName"]]
    sp = known.iloc[0]
    rec = {lvl: {"taxon": sp[lvl if lvl != "species" else "scientificName"], "prob": 0.9, "score": 0.9} for lvl in LEVELS}
    rec["depth"] = 4
    held_out = get_taxonomy()
    held_out = held_out[(held_out["benchmark_role"] == "semantic_ood") & (held_out["distance_band_vs_trainable"] == "near_genus")].iloc[0]
    res = evaluate_hierarchical_records(
        [rec, rec], [{"split": "iid_test", "species": sp["scientificName"]}, {"split": "semantic_ood", "species": held_out["scientificName"]}], known
    )
    assert res["per_split"]["iid_test"]["acc_species"] == 1.0
    assert res["per_split"]["semantic_ood"]["over_commit_rate"] == 1.0  # depth 4 on a species it cannot know
    assert np.isfinite(res["novelty"]["species"]["auroc"])
