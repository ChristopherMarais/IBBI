# src/ibbi/evaluate/benchmark.py

"""
Crowd-aware scoring of predictions on the Bark and Ambrosia Beetle Detection Benchmark.

Two entry points:

* `evaluate_predictions` scores species-level predictions (category ids are the benchmark's global ids, 1..175) with
  the benchmark's own reference evaluator (`_reference_evaluator.py`, vendored unchanged). It reports the COCO suite,
  per-species AP, class-agnostic detection, the detection / identification decomposition (detection recall, species
  accuracy given detection, end-to-end species recall), taxonomic graceful degradation on unseen species,
  calibration and known-vs-novel separation.
* `evaluate_class_agnostic` scores detectors that only localise ("arthropod", zero-shot prompts): class-agnostic
  COCO AP/AR plus recall and precision at IoU 0.5 at an operating confidence. Crowd regions are ignored in both.

Predictions are COCO results: `{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}` with
absolute-pixel boxes in the frame of the benchmark's COCO files.
"""

import contextlib
import io
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import _reference_evaluator as ref

# The headline numbers reported in the package's model summary (see docs/benchmark.md).
SPECIES_HEADLINE = {
    "iid_test": [
        ("coco", "AP_50_95"),
        ("coco", "AP_50"),
        ("decomposition", "detection_recall"),
        ("decomposition", "acc_species_given_det"),
        ("decomposition", "species_recall"),
        ("decomposition", "acc_genus_given_det"),
        ("class_agnostic", "class_agnostic_AR_100"),
        ("calibration", "ECE"),
    ],
    "inat_test": [
        ("coco", "AP_50"),
        ("decomposition", "detection_recall"),
        ("decomposition", "species_recall"),
    ],
    "semantic_ood": [
        ("class_agnostic", "class_agnostic_AR_100"),
        ("decomposition", "detection_recall"),
        ("decomposition", "acc_genus_given_det"),
        ("decomposition", "acc_tribe_given_det"),
    ],
}


def _write_predictions(tmp: Path, predictions: dict[str, list]) -> None:
    for split, preds in predictions.items():
        with open(tmp / f"{split}_predictions.json", "w") as f:
            json.dump(preds, f)


def evaluate_predictions(
    predictions: dict[str, list[dict[str, Any]]],
    benchmark_dir: str | Path,
    iou: float = 0.5,
    quiet: bool = True,
    output_dir: str | Path | None = None,
    model_name: str = "model",
) -> dict[str, Any]:
    """Scores species-level predictions with the benchmark's reference evaluator.

    Args:
        predictions (dict[str, list]): split name -> COCO results list. Splits: "iid_test", "inat_test", "semantic_ood".
        benchmark_dir (str | Path): Benchmark root (see `ibbi.download_benchmark`); needs the annotations of every
            split (only the annotation files are read).
        iou (float): IoU threshold for matching-based metrics. Defaults to 0.5.
        quiet (bool): Suppress the evaluator's progress output. Defaults to True.
        output_dir (str | Path | None): If given, the evaluator's files (summary JSON/CSV, per-species CSV, report)
            are written there.
        model_name (str): File-name prefix for `output_dir`.

    Returns:
        dict: `{"per_split": {...}, "summary": {...}, "headline": {...}}`. `headline` holds the flat metrics listed in
        `SPECIES_HEADLINE`.
    """
    benchmark_dir = Path(benchmark_dir)
    splits = list(predictions)
    with tempfile.TemporaryDirectory() as tmp:
        _write_predictions(Path(tmp), predictions)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
            results = ref.run_evaluation(benchmark_dir, Path(tmp), splits=splits, iou=iou, quiet=quiet)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ref.write_per_species_csv(results, out, model_name)
        ref.write_detections_csv(results, out, model_name)
    ref._strip_private(results)
    if output_dir is not None:
        out = Path(output_dir)
        ref.write_summary_json(results, out, model_name)
        ref.write_summary_csv(results, out, model_name)
        (out / f"{model_name}_report.txt").write_text(ref.render_report(results, model_name))
    results["headline"] = headline_metrics(results)
    return results


def headline_metrics(results: dict[str, Any]) -> dict[str, float]:
    """Flattens the headline metrics of `evaluate_predictions` output to `{"<split>.<metric>": value}`."""
    flat: dict[str, float] = {}
    for split, keys in SPECIES_HEADLINE.items():
        sr = results.get("per_split", {}).get(split)
        if not sr:
            continue
        for block, key in keys:
            v = sr.get(block, {}).get(key)
            if v is not None:
                flat[f"{split}.{key}"] = float(v)
    for k in ("novelty_det_AUROC", "novelty_det_FPR@95TPR", "photography_robustness_gap_AP_50", "species_generalization_gap_AR_100"):
        if k in results.get("summary", {}):
            flat[f"summary.{k}"] = float(results["summary"][k])
    return flat


def _class_agnostic_gt(coco_path: Path):
    from pycocotools.coco import COCO

    with open(coco_path) as f:
        d = json.load(f)
    d = {
        "info": d.get("info", {}),
        "licenses": d.get("licenses", []),
        "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
        "images": d["images"],
        "annotations": [{**a, "category_id": 1} for a in d["annotations"]],
    }
    gt = COCO()
    gt.dataset = d
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def evaluate_class_agnostic(
    predictions: dict[str, list[dict[str, Any]]],
    benchmark_dir: str | Path,
    operating_conf: float | None = None,
    max_dets: int = 100,
) -> dict[str, Any]:
    """Class-agnostic, crowd-aware detection metrics for every split in `predictions`.

    Reported per split: COCO AP (IoU 0.50:0.95), AP50, AP75, AR@100 (pycocotools, crowd regions ignored), and at IoU
    0.5 with at most `max_dets` detections per image: maximum recall (every score), and recall, precision and false
    positives per image at `operating_conf` (when given). Detections on crowd regions are neither true nor false
    positives.

    Args:
        predictions (dict[str, list]): split -> COCO results (category ids are ignored).
        benchmark_dir (str | Path): Benchmark root.
        operating_conf (float | None): Score threshold of the model's operating point.
        max_dets (int): Detections per image considered. Defaults to 100.

    Returns:
        dict: `{split: {metric: value}}` and a flat `"headline"` dict.
    """
    from pycocotools.cocoeval import COCOeval

    benchmark_dir = Path(benchmark_dir)
    out: dict[str, Any] = {}
    flat: dict[str, float] = {}
    for split, preds in predictions.items():
        gt = _class_agnostic_gt(benchmark_dir / "detection" / "annotations_coco" / f"{split}.json")
        n_img = len(gt.getImgIds())
        res: dict[str, float] = {"n_predictions": float(len(preds))}
        if not preds:
            out[split] = res
            continue
        dts = [{**p, "category_id": 1} for p in preds]
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(dts)
            ev = COCOeval(gt, dt, "bbox")
            ev.params.maxDets = [1, 10, max_dets]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        s = ev.stats
        res.update({"AP_50_95": float(s[0]), "AP_50": float(s[1]), "AP_75": float(s[2]), f"AR_{max_dets}": float(s[8])})
        # matching at IoU 0.5 (index 0 of params.iouThrs), all areas, max_dets detections per image
        tps, fps, scores, n_gt = [], [], [], 0
        for e in ev.evalImgs:
            if e is None or e["aRng"] != ev.params.areaRng[0] or e["maxDet"] != max_dets:
                continue
            n_gt += int(np.sum(~np.asarray(e["gtIgnore"], dtype=bool)))
            dtm = np.asarray(e["dtMatches"])[0]
            dig = np.asarray(e["dtIgnore"], dtype=bool)[0]
            sc = np.asarray(e["dtScores"])
            keep = ~dig
            tps.append((dtm[keep] > 0).astype(np.int8))
            fps.append((dtm[keep] == 0).astype(np.int8))
            scores.append(sc[keep])
        tp, fp, sc = (np.concatenate(v) if v else np.zeros(0) for v in (tps, fps, scores))
        res["n_gt"] = float(n_gt)
        res["max_recall_50"] = float(tp.sum() / n_gt) if n_gt else float("nan")
        if operating_conf is not None:
            m = sc >= operating_conf
            n_tp, n_fp = float(tp[m].sum()), float(fp[m].sum())
            res["operating_conf"] = float(operating_conf)
            res["recall_50_at_op"] = n_tp / n_gt if n_gt else float("nan")
            res["precision_50_at_op"] = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else float("nan")
            res["fp_per_image_at_op"] = n_fp / n_img if n_img else float("nan")
        out[split] = res
        for k, v in res.items():
            if k not in ("n_predictions", "n_gt", "operating_conf"):
                flat[f"{split}.class_agnostic_{k}"] = v
    out["headline"] = flat
    return out


def to_coco_results(
    image_id: int,
    boxes_xyxy: Sequence[Sequence[float]],
    scores: Sequence[float],
    category_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Converts one image's xyxy boxes to COCO result dicts (xywh). `category_ids=None` writes category 1."""
    out = []
    for i, (b, s) in enumerate(zip(boxes_xyxy, scores)):
        x0, y0, x1, y1 = (float(v) for v in b)
        out.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_ids[i]) if category_ids is not None else 1,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": float(s),
            }
        )
    return out
