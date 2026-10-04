# src/ibbi/evaluate/__init__.py

"""
Provides the high-level `Evaluator` class for assessing IBBI models on the Bark and Ambrosia Beetle Detection Benchmark.

* `Evaluator.benchmark` runs a detector (species-level, arthropod or zero-shot) or a detector + classifier pipeline on
  the benchmark splits and scores the predictions with the benchmark's crowd-aware reference evaluator.
* `Evaluator.hierarchical_classification` scores a hierarchical classifier on the ground-truth specimen crops:
  per-level accuracy and calibration on known species, per-level novelty separation on the held-out species, and the
  reported taxonomic depth against the deepest level that could be right.
* `Evaluator.embeddings` measures how well a model's embeddings cluster by species and how well embedding distances
  follow taxonomic distance.
"""

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ..utils.data import BENCHMARK_REVISION, BenchmarkDataset, download_benchmark, get_dataset, taxonomic_distance_matrix
from .benchmark import evaluate_class_agnostic, evaluate_predictions, to_coco_results
from .embeddings import EmbeddingEvaluator
from .hierarchical import evaluate_hierarchical_records

DEFAULT_SPLITS = ("iid_test", "inat_test", "semantic_ood")


class Evaluator:
    """A unified evaluator for IBBI models.

    Args:
        model: Any model created with `ibbi.create_model` or `ibbi.create_pipeline`.
    """

    def __init__(self, model: Any):
        self.model = model

    # ------------------------------------------------------------------------------------------------------------
    def predict_split(
        self,
        dataset: BenchmarkDataset,
        predict_kwargs: dict[str, Any] | None = None,
        max_images: int | None = None,
        progress: bool = True,
    ) -> list[dict[str, Any]]:
        """Runs the model on every image of a benchmark split and returns COCO result dicts.

        Species-level models write the benchmark's global category id of each predicted species; class-agnostic models
        write category 1.
        """
        kwargs = dict(getattr(self.model, "benchmark_kwargs", {}) or {})
        kwargs.update(predict_kwargs or {})
        species_level = bool(getattr(self.model, "is_species_level", False))
        name_to_cat = _benchmark_category_ids(dataset.root) if species_level else {}
        n = len(dataset) if max_images is None else min(max_images, len(dataset))
        preds: list[dict[str, Any]] = []
        # models that pool work across images (the pipeline's fast path) get several images per call
        chunk = int(getattr(self.model, "predict_batch_images", 1)) if getattr(self.model, "fast", False) else 1
        bar = tqdm(total=n, desc=f"{dataset.split}", disable=not progress)
        for start in range(0, n, chunk):
            items = [dataset[i] for i in range(start, min(n, start + chunk))]
            results = self.model.predict([it["image"] for it in items], **kwargs) if chunk > 1 else [self.model.predict(items[0]["image"], **kwargs)]
            bar.update(len(items))
            for item, res in zip(items, results):
                preds.extend(self._to_coco(item, res, species_level, name_to_cat))
        bar.close()
        return preds

    @staticmethod
    def _to_coco(item: dict[str, Any], res: dict[str, Any], species_level: bool, name_to_cat: dict[str, int]) -> list[dict[str, Any]]:
        """COCO result dicts of one image (species-level models: benchmark category ids; others: category 1)."""
        labels = res.get("species", res.get("labels", []))
        cats = None
        if species_level:
            cats = [name_to_cat.get(lbl, -1) for lbl in labels]
            keep = [j for j, c in enumerate(cats) if c != -1]
            res = {k: [res[k][j] for j in keep] for k in ("boxes", "scores")}
            cats = [cats[j] for j in keep]
        return to_coco_results(item["image_id"], res["boxes"], res["scores"], cats)

    def benchmark(
        self,
        splits: Sequence[str] = DEFAULT_SPLITS,
        dataset_dir: str | Path | None = None,
        predict_kwargs: dict[str, Any] | None = None,
        max_images: int | None = None,
        output_dir: str | Path | None = None,
        model_name: str | None = None,
        operating_conf: float | None = None,
        revision: str = BENCHMARK_REVISION,
    ) -> dict[str, Any]:
        """Runs the model on the benchmark splits and scores it with the benchmark's crowd-aware evaluator.

        Species-level models (the species detectors and detector + classifier pipelines) get the full reference
        evaluation (COCO suite, detection / identification decomposition, taxonomic degradation on unseen species,
        calibration, novelty). Class-agnostic detectors (arthropod and zero-shot detectors) get class-agnostic AP/AR
        plus recall and precision at their operating confidence.

        Args:
            splits (Sequence[str]): Splits to evaluate. Defaults to iid_test, inat_test and semantic_ood.
            dataset_dir (str | Path | None): Benchmark root; downloaded to the ibbi cache when omitted.
            predict_kwargs (dict | None): Extra arguments for `model.predict` (override the model's benchmark defaults,
                e.g. the low confidence floor used for AP).
            max_images (int | None): Evaluate only the first N images of each split (quick checks; metrics are then
                not comparable to published numbers).
            output_dir (str | Path | None): Write predictions and evaluator reports here.
            model_name (str | None): Name used for output files.
            operating_conf (float | None): Operating confidence for class-agnostic recall/precision. Defaults to the
                model's `operating_conf` attribute when it has one.
            revision (str): Benchmark revision. Defaults to the pinned v2.0.1 commit.

        Returns:
            dict: Evaluator output with a flat `"headline"` dict of the key metrics.
        """
        root = Path(dataset_dir) if dataset_dir is not None else None
        if root is None:
            root = download_benchmark(list(splits), revision=revision)
        preds = {}
        for split in splits:
            ds = get_dataset(split, local_dir=root, revision=revision)
            preds[split] = self.predict_split(ds, predict_kwargs=predict_kwargs, max_images=max_images)
        name = model_name or getattr(self.model, "name", type(self.model).__name__)
        if output_dir is not None:
            import json

            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for split, p in preds.items():
                (out / f"{split}_predictions.json").write_text(json.dumps(p))
        if getattr(self.model, "is_species_level", False):
            return evaluate_predictions(preds, root, output_dir=output_dir, model_name=name)
        op = operating_conf if operating_conf is not None else getattr(self.model, "operating_conf", None)
        res = evaluate_class_agnostic(preds, root, operating_conf=op)
        if output_dir is not None:
            import json

            (Path(output_dir) / f"{name}_class_agnostic.json").write_text(json.dumps(res, indent=1))
        return res

    # ------------------------------------------------------------------------------------------------------------
    def hierarchical_classification(
        self,
        splits: Sequence[str] = DEFAULT_SPLITS,
        dataset_dir: str | Path | None = None,
        include_crowd: bool = False,
        operating_point: str | None = None,
        max_images: int | None = None,
        batch_size: int = 32,
        revision: str = BENCHMARK_REVISION,
    ) -> dict[str, Any]:
        """Scores a hierarchical classifier on ground-truth specimen crops of the benchmark.

        Every annotated specimen is cropped at its box (grown by the classifier's padding) and classified. Metrics, per
        level (subfamily, tribe, genus, species): accuracy and expected calibration error on taxa the classifier knows;
        AUROC and FPR at 95% TPR of the novelty score for "known at this level" versus "unknown at this level" (the
        held-out species are unknown at species level, and also at genus and tribe level when their genus or tribe is
        absent from training); the reported depth against the ideal depth, including the over-commit rate (naming a
        taxon below the deepest level that could be right).

        Args:
            splits (Sequence[str]): Splits to use. Defaults to iid_test, inat_test and semantic_ood.
            dataset_dir (str | Path | None): Benchmark root; downloaded to the ibbi cache when omitted.
            include_crowd (bool): Also classify the unscored crowd specimens of iid_test / inat_test (8x more known
                specimens; the benchmark standard is the scored ones only). Defaults to False.
            operating_point (str | None): Classifier operating point ("0.90", "0.95", "0.99", "gallery"). Defaults to
                the classifier's default.
            max_images (int | None): Use only the first N images per split.
            batch_size (int): Crops per forward pass.
            revision (str): Benchmark revision.

        Returns:
            dict: `{"per_split": ..., "novelty": ..., "headline": {...}}`.
        """
        clf = getattr(self.model, "classifier", self.model)
        if not hasattr(clf, "classify_crops"):
            raise TypeError("hierarchical_classification needs a hierarchical classifier (or a pipeline that contains one).")
        root = Path(dataset_dir) if dataset_dir is not None else download_benchmark(list(splits), revision=revision)
        records, truths = [], []
        for split in splits:
            ds = get_dataset(split, local_dir=root, revision=revision)
            n = len(ds) if max_images is None else min(max_images, len(ds))
            crops, meta = [], []
            for i in tqdm(range(n), desc=f"{split} crops"):
                item = ds[i]
                o = item["objects"]
                for j, (x, y, w, h) in enumerate(o["bbox"]):
                    if o["iscrowd"][j] and not include_crowd:
                        continue
                    if w < 2 or h < 2:
                        continue
                    crops.append(clf.crop(item["image"], (x, y, x + w, y + h)))
                    meta.append({"split": split, "species": o["category"][j], "crowd": int(o["iscrowd"][j])})
                if len(crops) >= 512 or (i == n - 1 and crops):
                    recs = clf.classify_crops(crops, operating_point=operating_point, batch_size=batch_size)
                    records.extend(recs)
                    truths.extend(meta)
                    crops, meta = [], []
        return evaluate_hierarchical_records(records, truths, clf.taxonomy_table)

    # ------------------------------------------------------------------------------------------------------------
    def embeddings(
        self,
        dataset,
        evaluation_level: str = "object",
        use_umap: bool = True,
        extract_kwargs: dict[str, Any] | None = None,
        batch_size: int = 32,
        include_crowd: bool = True,
        **kwargs,
    ):
        """Evaluates the quality of the model's feature embeddings.

        Embeddings are extracted per specimen crop ("object", default) or per image ("image"), clustered (optionally
        after UMAP) and compared with the species labels (ARI, NMI, purity, internal cluster indices). A Mantel test
        compares between-species embedding distances with taxonomic distance (0 same species, 1 same genus, 2 same tribe,
        3 same subfamily, 4 otherwise).

        Args:
            dataset: A `BenchmarkDataset` (or any iterable of items with "image" and "objects").
            evaluation_level (str): "object" or "image".
            use_umap (bool): Reduce with UMAP before clustering. Defaults to True.
            extract_kwargs (dict | None): Arguments for `model.extract_features`.
            batch_size (int): Batch size for the distance computation of the Mantel test.
            include_crowd (bool): Use unscored crowd specimens too (they carry true labels). Defaults to True.
            **kwargs: Passed to `EmbeddingEvaluator`.

        Returns:
            dict: Clustering metrics, sample-level results and the Mantel correlation.
        """
        if extract_kwargs is None:
            extract_kwargs = {}
        if evaluation_level not in ["image", "object"]:
            raise ValueError("evaluation_level must be either 'image' or 'object'.")
        print(f"Extracting embeddings for evaluation at the '{evaluation_level}' level...")
        embeddings_list, label_names = [], []
        for item in tqdm(dataset):
            objs = item.get("objects", {})
            if evaluation_level == "image":
                emb = self.model.extract_features(item["image"], **extract_kwargs)
                if emb is not None and objs.get("category"):
                    embeddings_list.append(emb)
                    label_names.append(objs["category"][0])
                continue
            for j, (x, y, w, h) in enumerate(objs.get("bbox", [])):
                if w <= 1 or h <= 1 or (not include_crowd and objs.get("iscrowd", [0] * (j + 1))[j]):
                    continue
                emb = self.model.extract_features(item["image"].crop((x, y, x + w, y + h)), **extract_kwargs)
                if emb is not None:
                    embeddings_list.append(emb)
                    label_names.append(objs["category"][j])
        if not embeddings_list:
            print("Warning: Could not extract any valid embeddings from the dataset.")
            return {}
        embeddings = np.array([np.asarray(e.detach().cpu() if hasattr(e, "detach") else e, dtype=np.float32).flatten() for e in embeddings_list])
        names = sorted(set(label_names))
        name_to_idx = {n: i for i, n in enumerate(names)}
        idx_to_name = dict(enumerate(names))
        true_labels = np.array([name_to_idx[n] for n in label_names])
        evaluator = EmbeddingEvaluator(embeddings, use_umap=use_umap, **kwargs)
        results: dict[str, Any] = {"internal_cluster_validation": evaluator.evaluate_cluster_structure()}
        results["external_cluster_validation"] = evaluator.evaluate_against_truth(true_labels)
        results["sample_results"] = evaluator.get_sample_results(true_labels, label_map=idx_to_name)
        if len(names) >= 3:
            try:
                ext = taxonomic_distance_matrix([n for n in names if n in _known_species()])
                mantel_eval = EmbeddingEvaluator(embeddings, use_umap=False)
                r, p, n_items, per_class = mantel_eval.compare_to_distance_matrix(
                    true_labels, label_map=idx_to_name, ext_distance=ext, batch_size=batch_size
                )
                results["mantel_correlation"] = {"r": r, "p_value": p, "n_items": n_items, "distance": "taxonomic"}
                results["per_class_centroids"] = per_class
            except (ImportError, KeyError, ValueError) as e:
                print(f"Could not run Mantel test: {e}")
        return results

    # ------------------------------------------------------------------------------------------------------------
    def object_classification(self, dataset, **kwargs):
        """Deprecated: use `Evaluator.benchmark`, which scores with the benchmark's crowd-aware evaluator."""
        warnings.warn(
            "Evaluator.object_classification() is deprecated; use Evaluator.benchmark(splits=[...]) instead. "
            "The old evaluator did not honour the benchmark's crowd regions.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(dataset, BenchmarkDataset):
            raise TypeError("object_classification now only accepts a BenchmarkDataset from ibbi.get_dataset().")
        return self.benchmark(splits=[dataset.split], dataset_dir=dataset.root, predict_kwargs=kwargs.get("predict_kwargs"))


_CAT_CACHE: dict[str, dict[str, int]] = {}


def _benchmark_category_ids(root: Path) -> dict[str, int]:
    """species name -> global category id, read from the benchmark's COCO files (cached)."""
    import json

    key = str(root)
    if key not in _CAT_CACHE:
        m: dict[str, int] = {}
        for p in sorted((Path(root) / "detection" / "annotations_coco").glob("*.json")):
            with open(p) as f:
                for c in json.load(f).get("categories", []):
                    m[c["name"]] = int(c["id"])
        _CAT_CACHE[key] = m
    return _CAT_CACHE[key]


def _known_species() -> set:
    from ..utils.data import get_taxonomy

    return set(get_taxonomy()["scientificName"])


__all__ = ["Evaluator"]
