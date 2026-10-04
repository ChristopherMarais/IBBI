"""ibbi.Evaluator and ibbi.evaluate.* on the synthetic benchmark (offline)."""

import json
import warnings

import numpy as np
import pytest

import ibbi
from ibbi.evaluate.benchmark import evaluate_class_agnostic, evaluate_predictions, headline_metrics, to_coco_results
from ibbi.evaluate.hierarchical import evaluate_hierarchical_records


def test_to_coco_results():
    r = to_coco_results(7, [[10, 20, 30, 60]], [0.5], [3])
    assert r == [{"image_id": 7, "category_id": 3, "bbox": [10.0, 20.0, 20.0, 40.0], "score": 0.5}]
    assert to_coco_results(1, [[0, 0, 1, 1]], [0.1])[0]["category_id"] == 1


def test_reference_evaluator_is_vendored_unchanged():
    from importlib import resources

    src = resources.files("ibbi.evaluate").joinpath("_reference_evaluator.py").read_text()
    assert src.count("_n_real_gt") >= 4  # the dataset card's check for a crowd-aware evaluator
    assert "sys.exit(1)" not in src.split("def main")[0]


def test_species_benchmark_end_to_end(tiny_benchmark, fake_species_detector, known_species):
    ev = ibbi.Evaluator(fake_species_detector)
    res = ev.benchmark(dataset_dir=tiny_benchmark)
    h = res["headline"]
    # one detection per image on the scored specimen: every specimen is found; only the first species is named right
    assert h["iid_test.detection_recall"] == 1.0
    assert h["iid_test.acc_species_given_det"] == pytest.approx(1 / len(known_species))
    assert h["semantic_ood.detection_recall"] == 1.0
    assert 0 <= h["iid_test.AP_50"] <= 1
    assert all(c == {"conf": 0.001} for c in fake_species_detector.calls)  # the model's benchmark defaults were used
    assert set(res["per_split"]) == {"iid_test", "inat_test", "semantic_ood"}


def test_species_benchmark_writes_outputs(tiny_benchmark, fake_species_detector, tmp_path):
    ibbi.Evaluator(fake_species_detector).benchmark(splits=["iid_test"], dataset_dir=tiny_benchmark, output_dir=tmp_path, model_name="fake")
    assert (tmp_path / "iid_test_predictions.json").exists()
    assert (tmp_path / "fake_summary.json").exists() and (tmp_path / "fake_report.txt").exists()
    preds = json.loads((tmp_path / "iid_test_predictions.json").read_text())
    assert len(preds) == 8 and all(p["category_id"] >= 1 for p in preds)


def test_unknown_species_labels_are_dropped(tiny_benchmark, fake_species_detector):
    fake_species_detector.species = "Not a benchmark species"
    preds = ibbi.Evaluator(fake_species_detector).predict_split(ibbi.get_dataset("iid_test", local_dir=tiny_benchmark, download=False))
    assert preds == []


def test_class_agnostic_benchmark(tiny_benchmark, fake_box_detector):
    res = ibbi.Evaluator(fake_box_detector).benchmark(splits=["iid_test", "semantic_ood"], dataset_dir=tiny_benchmark)
    iid = res["iid_test"]
    assert iid["n_gt"] == 8  # crowd specimens are not targets
    assert iid["recall_50_at_op"] == 1.0
    # per image: one TP, one detection on a crowd region (ignored), one false alarm
    assert iid["precision_50_at_op"] == pytest.approx(0.5)
    assert iid["fp_per_image_at_op"] == pytest.approx(1.0)
    # semantic_ood has no crowd regions: the crowd-region box is a false alarm there
    assert res["semantic_ood"]["precision_50_at_op"] == pytest.approx(1 / 3)
    assert "iid_test.class_agnostic_AP_50" in res["headline"]


def test_evaluate_predictions_direct(tiny_benchmark):
    ds = ibbi.get_dataset("iid_test", local_dir=tiny_benchmark, download=False)
    perfect = []
    for r in ds.records():
        o = r["objects"]
        for b, c, crowd in zip(o["bbox"], o["category_id"], o["iscrowd"], strict=True):
            if not crowd:
                perfect.append({"image_id": r["image_id"], "category_id": c, "bbox": b, "score": 0.99})
    res = evaluate_predictions({"iid_test": perfect}, tiny_benchmark)
    assert res["headline"]["iid_test.AP_50"] == pytest.approx(1.0)
    assert res["headline"]["iid_test.acc_species_given_det"] == 1.0
    assert headline_metrics(res) == res["headline"]
    ca = evaluate_class_agnostic({"iid_test": perfect, "inat_test": []}, tiny_benchmark, operating_conf=0.5)
    assert ca["iid_test"]["AP_50"] == pytest.approx(1.0) and ca["inat_test"] == {"n_predictions": 0.0}


def test_hierarchical_classification(tiny_benchmark, tiny_classifier):
    res = ibbi.Evaluator(tiny_classifier).hierarchical_classification(dataset_dir=tiny_benchmark, batch_size=4)
    ps = res["per_split"]
    assert ps["iid_test"]["n"] == 8 and ps["semantic_ood"]["n"] == 6
    assert set(ps["semantic_ood"]["by_band"]) == {"near_genus", "mid_tribe", "far_tribe"}
    assert all(0 <= ps[s]["over_commit_rate"] <= 1 for s in ps)
    assert ps["iid_test"]["over_commit_rate"] == 0.0  # nothing can be finer than species for a known species
    assert {"genus", "species"} <= set(res["novelty"])
    with_crowd = ibbi.Evaluator(tiny_classifier).hierarchical_classification(splits=["iid_test"], dataset_dir=tiny_benchmark, include_crowd=True)
    assert with_crowd["per_split"]["iid_test"]["n"] == 16


def test_hierarchical_classification_through_pipeline(tiny_benchmark, tiny_classifier, fake_box_detector):
    pipe = ibbi.IdentificationPipeline(fake_box_detector, tiny_classifier)
    res = ibbi.Evaluator(pipe).hierarchical_classification(splits=["inat_test"], dataset_dir=tiny_benchmark)
    assert res["per_split"]["inat_test"]["n"] == 2


def test_hierarchical_classification_needs_classifier(fake_box_detector, tiny_benchmark):
    with pytest.raises(TypeError):
        ibbi.Evaluator(fake_box_detector).hierarchical_classification(dataset_dir=tiny_benchmark)


def test_hierarchical_metrics_known_vs_unknown(taxonomy, known_species, held_out_species, tiny_classifier):
    known = tiny_classifier.taxonomy_table
    sp = known.iloc[0]

    def rec(depth, score):
        r = {
            lvl: {"taxon": sp[lvl if lvl != "species" else "scientificName"], "prob": 0.8, "score": score}
            for lvl in ("subfamily", "tribe", "genus", "species")
        }
        r["depth"] = depth
        return r

    # a held-out species whose genus the (tiny) classifier knows: ideal depth 3
    t = taxonomy[(taxonomy["benchmark_role"] == "semantic_ood") & (taxonomy["genus"].isin(set(known["genus"])))]
    if t.empty:
        pytest.skip("no held-out congener of the tiny classifier's species")
    near = t.iloc[0]["scientificName"]
    res = evaluate_hierarchical_records(
        [rec(4, 0.9), rec(3, 0.2), rec(4, 0.9)],
        [
            {"split": "iid_test", "species": sp["scientificName"]},
            {"split": "semantic_ood", "species": near},
            {"split": "semantic_ood", "species": near},
        ],
        known,
    )
    so = res["per_split"]["semantic_ood"]
    assert so["over_commit_rate"] == 0.5
    assert res["rows"]["ideal_depth"].tolist() == [4, 3, 3]
    assert res["novelty"]["species"]["auroc"] == pytest.approx(0.75)


def test_embeddings(tiny_benchmark, tiny_classifier):
    ds = ibbi.get_dataset("iid_test", local_dir=tiny_benchmark, download=False)
    res = ibbi.Evaluator(tiny_classifier).embeddings(ds, use_umap=False, min_cluster_size=2)
    assert "internal_cluster_validation" in res and "external_cluster_validation" in res
    assert len(res["sample_results"]) == 16  # scored + crowd specimens
    if "mantel_correlation" in res:
        assert -1 <= res["mantel_correlation"]["r"] <= 1 and res["mantel_correlation"]["distance"] == "taxonomic"
    img_level = ibbi.Evaluator(tiny_classifier).embeddings(ds, evaluation_level="image", use_umap=False, min_cluster_size=2)
    assert len(img_level["sample_results"]) == 8
    with pytest.raises(ValueError):
        ibbi.Evaluator(tiny_classifier).embeddings(ds, evaluation_level="pixel")


def test_object_classification_deprecated(tiny_benchmark, fake_box_detector):
    ds = ibbi.get_dataset("iid_test", local_dir=tiny_benchmark, download=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = ibbi.Evaluator(fake_box_detector).object_classification(ds)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert "iid_test" in res
    with pytest.raises(TypeError):
        ibbi.Evaluator(fake_box_detector).object_classification([{"image": None}])


def test_list_models():
    df = ibbi.list_models(as_df=True)
    assert set(df["Model Name"]) == set(ibbi.model_registry)
    assert df["Licence"].notna().all()
    assert np.isfinite(df.loc[df["Model Name"] == "yolo11x_arthropod_detector", "Parameters (M)"]).all()
