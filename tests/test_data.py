"""ibbi.utils.data: benchmark loading, taxonomy, deprecations and download retries (offline)."""

import warnings

import numpy as np
import pytest
from PIL import Image

import ibbi
from ibbi.utils import data as D


def test_split_items(tiny_benchmark, known_species):
    ds = ibbi.get_dataset("iid_test", local_dir=tiny_benchmark, download=False)
    assert len(ds) == 8
    item = ds[0]
    assert item["image"].mode == "RGB" and item["image"].size == (128, 96)
    o = item["objects"]
    assert o["iscrowd"] == [0, 1]
    assert o["category"][0] in known_species
    assert o["bbox"][0] == [20.0, 20.0, 40.0, 30.0]
    assert o["genus"][0] == o["category"][0].split()[0]
    assert set(o) >= {"bbox", "category", "category_id", "iscrowd", "subfamily", "tribe", "genus"}


def test_views(tiny_benchmark):
    ds = D.BenchmarkDataset(tiny_benchmark, "iid_test")
    assert len(ds[2:5]) == 3
    assert ds.select([3, 1])[0]["image_id"] == ds[3]["image_id"]
    assert sorted(r["image_id"] for r in ds.shuffle(seed=1).records()) == sorted(r["image_id"] for r in ds.records())
    assert ds[-1]["image_id"] == ds[len(ds) - 1]["image_id"]
    assert len(list(iter(ds))) == len(ds)
    assert "iid_test" in repr(ds)


def test_unknown_split_and_missing_files(tmp_path):
    with pytest.raises(ValueError):
        D.BenchmarkDataset(tmp_path, "val")
    with pytest.raises(FileNotFoundError):
        D.BenchmarkDataset(tmp_path, "iid_test")


def test_old_repo_rejected():
    with pytest.raises(ValueError, match="no longer supported"):
        ibbi.get_dataset(repo_id="IBBI-bio/ibbi_test_data")


def test_get_ood_dataset_is_deprecated(tiny_benchmark):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = ibbi.get_ood_dataset(local_dir=tiny_benchmark, download=False)
    assert ds.split == "semantic_ood"
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_download_benchmark_patterns(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(D, "snapshot_download", lambda *a, **k: calls.append(k))
    root = D.download_benchmark("inat_test", local_dir=tmp_path)
    assert root == tmp_path
    pats = calls[0]["allow_patterns"]
    assert "detection/images/inat_test/*" in pats and "detection/images/train/*" not in pats
    assert calls[0]["revision"] == D.BENCHMARK_REVISION
    D.download_benchmark(local_dir=tmp_path, images=False)
    assert not any(p.startswith("detection/images/") for p in calls[1]["allow_patterns"])
    with pytest.raises(ValueError):
        D.download_benchmark("validation", local_dir=tmp_path)


def test_download_retries_on_rate_limit(monkeypatch, tmp_path):
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    R = lambda: httpx.Response(429, request=httpx.Request("GET", "https://huggingface.co"))  # noqa: E731

    n = {"calls": 0}

    def flaky(*a, **k):
        n["calls"] += 1
        if n["calls"] < 3:
            raise HfHubHTTPError("rate limited", response=R())

    monkeypatch.setattr(D, "snapshot_download", flaky)
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    D._snapshot_with_retry(tmp_path, "rev", ["x"], max_attempts=5, wait_s=0)
    assert n["calls"] == 3


def test_download_does_not_retry_other_errors(monkeypatch, tmp_path):
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    R = lambda: httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co"))  # noqa: E731

    def broken(*a, **k):
        raise HfHubHTTPError("not found", response=R())

    monkeypatch.setattr(D, "snapshot_download", broken)
    with pytest.raises(HfHubHTTPError):
        D._snapshot_with_retry(tmp_path, "rev", ["x"], max_attempts=5, wait_s=0)


def test_shap_background_from_train(monkeypatch, tiny_benchmark):
    monkeypatch.setattr(D, "_default_root", lambda revision: tiny_benchmark)
    bg = ibbi.get_shap_background_dataset(image_size=(32, 24), n_images=3)
    assert len(bg) == 3 and bg[0]["image"].size == (32, 24)


def test_taxonomy_and_distance(taxonomy, known_species):
    assert len(taxonomy) == 175
    assert taxonomy["scientificName"].is_unique
    assert set(taxonomy["benchmark_role"]) == {"trainable", "semantic_ood"}
    d = D.taxonomic_distance_matrix(known_species)
    assert np.allclose(d.values, d.values.T) and (np.diag(d.values) == 0).all()
    assert d.loc[known_species[0], known_species[1]] == 1  # congeners
    assert d.values.max() <= 4
    with pytest.raises(KeyError):
        D.taxonomic_distance_matrix(["Not a species"])


def test_load_rgb_handles_16bit(tmp_path):
    arr = (np.arange(64 * 48, dtype=np.uint16).reshape(48, 64) * 20).astype(np.uint16)
    p = tmp_path / "x16.png"
    Image.fromarray(arr).save(p)
    im = D._load_rgb(p, (64, 48))
    assert im.mode == "RGB" and im.size == (64, 48)
