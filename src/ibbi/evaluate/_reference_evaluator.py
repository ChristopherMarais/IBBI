# ruff: noqa
# pyright: basic
# Vendored copy of `evaluation/evaluate.py` from the Bark and Ambrosia Beetle Detection Benchmark v2.0.1
# (https://huggingface.co/datasets/IBBI-bio/bark-ambrosia-beetle-benchmark, revision
# 8dc5a58e09c429e7d91077437846f424f5ab0c4e). It is the crowd-aware reference evaluator the benchmark's published
# results were scored with. Only change: the pycocotools import no longer calls sys.exit() when the library is
# missing (pycocotools is a dependency of ibbi). Do not edit; use `ibbi.evaluate.benchmark` instead.
#!/usr/bin/env python3
"""
============================================================================
Bark & Ambrosia Beetle Detection Benchmark — Standard Evaluation Harness
============================================================================

This script is the canonical evaluation tool for the IBBI-bio bark &
ambrosia beetle detection benchmark. It is framework-agnostic: it accepts
predictions in standard COCO results format and produces a consistent set
of metrics regardless of which model produced them.

WHAT IT EVALUATES
-----------------
The benchmark has three test splits and this script runs the appropriate
metrics on each:

  iid_test     : in-distribution test — same species and similar photography
                 as the training set. Establishes the model's ceiling
                 performance.

  inat_test    : photography-OOD — same species as training, but field
                 photos from iNaturalist instead of institutional photos.
                 Measures robustness to photography environment shift.

  semantic_ood : species-OOD — species absent from training. The model
                 cannot predict the correct species ID for these, so
                 standard mAP will be near zero. The harness instead runs
                 class-agnostic detection (did the model find ANY beetle?),
                 taxonomic graceful-degradation (was its best guess at
                 least the right genus/tribe?), and calibration analysis
                 (does the model become less confident on unfamiliar
                 species?).

USAGE
-----
Required directory structure for predictions:
    <predictions-dir>/
        iid_test_predictions.json
        inat_test_predictions.json
        semantic_ood_predictions.json

Each file is a COCO-results list:
    [
        {
            "image_id":    <int>,    # MUST match the integer id field in
                                     # the benchmark's COCO file for that split
            "category_id": <int>,    # category id as listed in the benchmark's
                                     # categories array (global IDs 1..175)
            "bbox":        [x, y, w, h],  # absolute pixels, COCO order
            "score":       <float>   # confidence in [0, 1]
        },
        ...
    ]

Run:
    python evaluate.py \\
        --benchmark-dir   ./bark-ambrosia-beetle-benchmark \\
        --predictions-dir ./my_predictions \\
        --output-dir      ./eval_results \\
        --model-name      my_model_v1

OUTPUT
------
    <output-dir>/
        <model_name>_summary.json    — all metrics, machine-readable
        <model_name>_summary.csv     — flat metrics table for leaderboards
        <model_name>_per_species.csv — per-species AP for diagnosing weak
                                       categories (small dataset species,
                                       visually-similar pairs, etc.)
        <model_name>_report.txt      — human-readable report (also printed)

The summary.json schema is stable. Headline metrics that should be reported
when comparing models:

    iid_test.coco.AP_50_95           — in-distribution detection quality
    inat_test.coco.AP_50_95          — photography-shift robustness
    semantic_ood.class_agnostic.AR_100 — species-shift detection ability
    summary.photography_robustness_gap — iid_test.AP_50 − inat_test.AP_50
    summary.species_generalization_gap — iid_test class-agnostic AR_100 −
                                         semantic_ood class-agnostic AR_100
    summary.confidence_drop_OOD      — should be positive (calibrated
                                       uncertainty on unseen species)

Dependencies: numpy, pandas, pycocotools (install: pip install pycocotools)
============================================================================
"""

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ============================================================================
# CONSTANTS
# ============================================================================

# Splits this harness knows how to evaluate. `train` is excluded by default
# because evaluating on training data is not informative for generalization.
DEFAULT_SPLITS = ["iid_test", "inat_test", "semantic_ood"]

# Splits where standard COCO mAP is meaningful (the model can predict
# the correct category IDs because they were in the training categories).
IN_DISTRIBUTION_SPECIES_SPLITS = {"iid_test", "inat_test"}

# Splits where standard mAP will be ~0 because the model wasn't trained
# on these category IDs. Class-agnostic and hierarchical metrics apply.
OUT_OF_DISTRIBUTION_SPECIES_SPLITS = {"semantic_ood"}

# Default IoU threshold for non-COCO-suite metrics (matching, ECE, etc.).
# 0.5 is the PASCAL VOC default and is what most downstream consumers use.
DEFAULT_IOU = 0.5

# Number of bins for Expected Calibration Error.
ECE_BINS = 10


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a detection model on the bark & ambrosia beetle benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--benchmark-dir", required=True, type=Path,
                   help="Path to the unpacked benchmark dataset (contains "
                        "detection/annotations_coco/, species_taxonomy.csv).")
    p.add_argument("--predictions-dir", required=True, type=Path,
                   help="Path to directory containing <split>_predictions.json files.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to write evaluation outputs.")
    p.add_argument("--model-name", default="model",
                   help="Identifier used in output filenames and reports.")
    p.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS,
                   choices=["iid_test", "inat_test", "semantic_ood", "train"],
                   help="Which splits to evaluate (default: all three test splits).")
    p.add_argument("--iou", type=float, default=DEFAULT_IOU,
                   help=f"IoU threshold for matching-based metrics (default: {DEFAULT_IOU}).")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress pycocotools' per-split printout.")
    return p.parse_args()


# ============================================================================
# LOADING + VALIDATION
# ============================================================================

def load_benchmark_coco(benchmark_dir: Path, split: str) -> COCO:
    """Load the COCO ground-truth file for a split."""
    coco_path = benchmark_dir / "detection" / "annotations_coco" / f"{split}.json"
    if not coco_path.exists():
        raise FileNotFoundError(f"Benchmark COCO not found: {coco_path}")
    return COCO(str(coco_path))


def load_taxonomy(benchmark_dir: Path) -> pd.DataFrame:
    """Load species_taxonomy.csv, which gives subfamily/tribe/genus for every
    benchmark species plus the benchmark_role and distance_band_vs_trainable
    columns we use for OOD breakdowns."""
    tax_path = benchmark_dir / "species_taxonomy.csv"
    if not tax_path.exists():
        raise FileNotFoundError(f"Taxonomy CSV not found: {tax_path}")
    return pd.read_csv(tax_path)


def build_global_category_lookup(benchmark_dir: Path) -> dict:
    """Build a cat_id -> species_name lookup that spans ALL benchmark splits.

    Needed because on semantic_ood, model predictions use TRAINABLE category
    IDs (the model can only emit classes it was trained on), but the
    semantic_ood COCO file only knows OOD category IDs. To compute
    hierarchical metrics we need to resolve a trainable predicted id to its
    species name even when evaluating an OOD split.

    Returns: {global_cat_id: species_name}
    """
    cat_lookup = {}
    coco_dir = benchmark_dir / "detection" / "annotations_coco"
    for coco_file in coco_dir.glob("*.json"):
        with open(coco_file) as f:
            data = json.load(f)
        for c in data.get("categories", []):
            cat_lookup[int(c["id"])] = c["name"]
    return cat_lookup


def load_train_image_counts(benchmark_dir: Path) -> dict:
    """Return per-species training image counts read from train.json.

    Used to bin species into head/medium/tail tiers for class-imbalance
    tolerance analysis. Returns {} if train.json is unavailable, in which
    case class imbalance metrics are skipped."""
    train_path = benchmark_dir / "detection" / "annotations_coco" / "train.json"
    if not train_path.exists():
        return {}
    with open(train_path) as f:
        data = json.load(f)
    cat_to_name = {c["id"]: c["name"] for c in data["categories"]}
    species_imgs = defaultdict(set)
    for ann in data["annotations"]:
        sp = cat_to_name.get(ann["category_id"])
        if sp is not None:
            species_imgs[sp].add(ann["image_id"])
    return {sp: len(imgs) for sp, imgs in species_imgs.items()}


def load_train_object_counts(benchmark_dir: Path):
    """[FIX 2] Per-species training OBJECT (annotation) counts + objects-per-image,
    straight from train.json. train.json already carries this; the harness only
    ever surfaced image counts. Enables AP-vs-data and the images-vs-objects
    question with NO change to the dataset. Returns ({}, {}) if unavailable."""
    train_path = benchmark_dir / "detection" / "annotations_coco" / "train.json"
    if not train_path.exists():
        return {}, {}
    with open(train_path) as f:
        data = json.load(f)
    cat_to_name = {c["id"]: c["name"] for c in data["categories"]}
    imgs = defaultdict(set)
    objs = defaultdict(int)
    for ann in data["annotations"]:
        sp = cat_to_name.get(ann["category_id"])
        if sp is None:
            continue
        imgs[sp].add(ann["image_id"])
        objs[sp] += 1
    n_obj = dict(objs)
    opi = {sp: (objs[sp] / len(imgs[sp]) if imgs[sp] else float("nan")) for sp in objs}
    return n_obj, opi


def load_predictions(predictions_dir: Path, split: str) -> list:
    """Load a model's predictions for a split. Returns [] if file missing."""
    pred_path = predictions_dir / f"{split}_predictions.json"
    if not pred_path.exists():
        print(f"  [warn] no predictions found at {pred_path}; skipping {split}")
        return []
    with open(pred_path) as f:
        preds = json.load(f)
    if not isinstance(preds, list):
        raise ValueError(f"{pred_path}: expected a list of COCO results, got {type(preds)}")
    return preds


def validate_predictions(coco_gt: COCO, predictions: list, split: str) -> tuple:
    """Validate prediction schema, drop malformed entries, return (cleaned, warnings).

    A robust eval harness must not crash on minor schema issues — instead, it
    drops bad entries with a warning so the user can fix their pipeline."""
    warnings = []
    cleaned = []
    valid_img_ids = set(coco_gt.getImgIds())
    valid_cat_ids = set(coco_gt.getCatIds())
    required_fields = ["image_id", "category_id", "bbox", "score"]

    n_bad_schema = n_bad_img = n_bad_cat = 0
    for i, pred in enumerate(predictions):
        if not isinstance(pred, dict) or not all(k in pred for k in required_fields):
            n_bad_schema += 1
            continue
        if pred["image_id"] not in valid_img_ids:
            n_bad_img += 1
            continue
        # Note: for OOD splits, predictions may use trainable category IDs that
        # are NOT in the OOD GT's categories. That's expected — the model can
        # only output classes it was trained on. We keep them; class-agnostic
        # and hierarchical metrics will use them; standard mAP will simply
        # contribute zero AP for those classes.
        if split in IN_DISTRIBUTION_SPECIES_SPLITS and pred["category_id"] not in valid_cat_ids:
            n_bad_cat += 1
            continue
        if len(pred["bbox"]) != 4:
            n_bad_schema += 1
            continue
        cleaned.append({
            "image_id":    int(pred["image_id"]),
            "category_id": int(pred["category_id"]),
            "bbox":        [float(v) for v in pred["bbox"]],
            "score":       float(pred["score"]),
        })

    if n_bad_schema:
        warnings.append(f"{n_bad_schema} predictions with malformed schema were dropped")
    if n_bad_img:
        warnings.append(f"{n_bad_img} predictions referenced unknown image_ids and were dropped")
    if n_bad_cat:
        warnings.append(f"{n_bad_cat} predictions used invalid category_ids and were dropped "
                        f"(only relevant for IID splits)")
    return cleaned, warnings


# ============================================================================
# BBOX HELPERS
# ============================================================================

def bbox_iou(box_a: list, box_b: list) -> float:
    """Intersection-over-Union for two boxes in COCO format [x, y, w, h]."""
    ax1, ay1, aw, ah = box_a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = box_b
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _n_real_gt(coco_gt, cat_id=None) -> int:
    """Count GT annotations, skipping crowd regions. Crowd marks specimens
    that are present but not scored: they cannot be recalled, so they must
    never enter a recall denominator."""
    ids = (coco_gt.getAnnIds(catIds=[cat_id]) if cat_id is not None
           else coco_gt.getAnnIds())
    return sum(1 for a in coco_gt.loadAnns(ids) if not a.get("iscrowd", 0))


def match_predictions_to_gt(coco_gt: COCO, predictions: list, iou_threshold: float) -> list:
    """Greedy per-image matching: for each prediction in descending confidence
    order, claim the unmatched GT box with the highest IoU >= threshold.

    Returns a list of dicts:
      {pred, gt (or None if unmatched), iou}

    This matching is class-agnostic — it answers "is this prediction a true
    detection of SOMETHING?" not "is this prediction the right class?".
    Class correctness is handled by downstream metrics (hierarchical etc.)."""
    by_img = defaultdict(list)
    for p in predictions:
        by_img[p["image_id"]].append(p)

    matches = []
    for img_id, img_preds in by_img.items():
        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        all_gts = coco_gt.loadAnns(gt_ann_ids)
        gts    = [g for g in all_gts if not g.get("iscrowd", 0)]
        crowds = [g for g in all_gts if g.get("iscrowd", 0)]
        claimed = set()  # GT ann_ids already matched
        for p in sorted(img_preds, key=lambda x: -x["score"]):
            best_iou, best_gt = 0.0, None
            for gt in gts:
                if gt["id"] in claimed:
                    continue
                iou = bbox_iou(p["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_gt = iou, gt
            if best_gt is not None and best_iou >= iou_threshold:
                matches.append({"pred": p, "gt": best_gt, "iou": best_iou})
                claimed.add(best_gt["id"])
            else:
                # Prediction on a crowd region: the specimen is really there,
                # it just is not scored. Neither TP nor FP. Checked only AFTER
                # real-GT matching fails, so detection_recall is unchanged.
                if any(bbox_iou(p["bbox"], c["bbox"]) >= iou_threshold
                       for c in crowds):
                    continue
                matches.append({"pred": p, "gt": None, "iou": best_iou})
    return matches


# ============================================================================
# METRIC 1 — STANDARD COCO mAP SUITE
# ============================================================================
#
# WHAT IT MEASURES
# ----------------
# The COCO mAP suite is the de-facto standard for object detection. It runs
# the per-class precision-recall curve at multiple IoU thresholds and reports
# 12 numbers. The most important:
#
#   AP @ [0.5:0.95]   primary COCO metric, mean of AP at IoU = 0.50, 0.55,
#                     ..., 0.95 (10 evenly-spaced values). What papers
#                     usually mean by "mAP".
#   AP @ 0.5          PASCAL VOC threshold. Lenient localization.
#   AP @ 0.75         Strict localization.
#   AR @ 1 / 10 / 100 Recall when the model is allowed to emit 1, 10, or 100
#                     detections per image. Reveals confidence ranking
#                     quality independent of NMS settings.
#   AP_small/med/lg   AP at object area < 32², 32²–96², ≥ 96² pixels.
#
# HOW TO INTERPRET ON THIS BENCHMARK
# ----------------------------------
# iid_test     The "ceiling". Reflects model-quality on a well-curated test
#              with similar photography to training. Strong models reach
#              AP_50 in the 0.6-0.9 range; AP_50_95 typically 30-50% lower.
#
# inat_test    Same species but different photography. Expect a noticeable
#              drop versus iid_test. The smaller the drop, the better the
#              photography-environment robustness.
#
# semantic_ood Standard mAP here will be near zero. This is by design —
#              the model has no way to emit the correct OOD category IDs.
#              Read the class-agnostic and hierarchical metrics below
#              instead. (We still compute and report standard mAP here for
#              completeness and to detect bugs — if you see non-zero
#              numbers, something is wrong with your category ID mapping.)
# ============================================================================

def evaluate_coco_standard(coco_gt: COCO, predictions: list, quiet: bool = False) -> dict:
    if not predictions:
        return _zero_coco_metrics()
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    else:
        coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    s = coco_eval.stats
    return {
        "AP_50_95": float(s[0]), "AP_50": float(s[1]), "AP_75": float(s[2]),
        "AP_small": float(s[3]), "AP_medium": float(s[4]), "AP_large": float(s[5]),
        "AR_1": float(s[6]),     "AR_10": float(s[7]),    "AR_100": float(s[8]),
        "AR_small": float(s[9]), "AR_medium": float(s[10]), "AR_large": float(s[11]),
    }


def _zero_coco_metrics() -> dict:
    keys = ["AP_50_95","AP_50","AP_75","AP_small","AP_medium","AP_large",
            "AR_1","AR_10","AR_100","AR_small","AR_medium","AR_large"]
    return {k: 0.0 for k in keys}


# ============================================================================
# METRIC 2 — PER-SPECIES AP @ IoU = 0.5
# ============================================================================
#
# WHAT IT MEASURES
# ----------------
# AP for each individual species at a single IoU threshold. Diagnostic, not
# headline. Used to identify:
#   - Species the model fails on (data-poor categories, confused pairs)
#   - Whether failure is uniform or concentrated in a few hard classes
#   - Genus-level confusions when two congenerics both score low
#
# Reported as a CSV: model_name_per_species.csv with one row per species.
# ============================================================================

def evaluate_per_species(coco_gt: COCO, predictions: list, quiet: bool = True) -> dict:
    """Returns: {category_id: {name, AP_50, AP_50_95, n_gt}}"""
    out = {}
    cat_ids = coco_gt.getCatIds()
    cats = {c["id"]: c for c in coco_gt.loadCats(cat_ids)}

    if not predictions:
        for cid in cat_ids:
            out[int(cid)] = {"name": cats[cid]["name"],
                             "AP_50": 0.0, "AP_50_95": 0.0,
                             "n_gt": _n_real_gt(coco_gt, cid)}
        return out

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate(); coco_eval.accumulate()

    # precision shape: (T, R, K, A, M) = (IoU thresholds, recall thresholds,
    # categories, area ranges, max detections). We use area="all", max_dets=100.
    precision = coco_eval.eval["precision"]  # (10, 101, K, 4, 3)
    iou_thresholds = coco_eval.params.iouThrs

    for k, cid in enumerate(coco_eval.params.catIds):
        # AP across all IoU thresholds (matches AP_50_95)
        p_all = precision[:, :, k, 0, 2]
        ap_5095 = float(p_all[p_all > -1].mean()) if (p_all > -1).any() else 0.0
        # AP at IoU = 0.5 specifically
        idx_50 = int(np.argmin(np.abs(iou_thresholds - 0.5)))
        p_50 = precision[idx_50, :, k, 0, 2]
        ap_50 = float(p_50[p_50 > -1].mean()) if (p_50 > -1).any() else 0.0
        out[int(cid)] = {
            "name": cats[cid]["name"],
            "AP_50": ap_50,
            "AP_50_95": ap_5095,
            "n_gt": _n_real_gt(coco_gt, cid),
        }
    return out


# ============================================================================
# METRIC 3 — CLASS-AGNOSTIC DETECTION
# ============================================================================
#
# WHAT IT MEASURES
# ----------------
# Detection performance when we don't care which class the model predicted
# — only whether it placed a box near a GT box. Implementation: re-label
# every GT annotation and every prediction to a single category "beetle",
# then run standard COCO eval.
#
# WHY IT MATTERS FOR THIS BENCHMARK
# ---------------------------------
# On semantic_ood, the model cannot predict the correct species ID by
# construction (those species weren't in training). But it can still
# correctly localize the beetle. Class-agnostic eval extracts that signal.
#
# HOW TO INTERPRET
# ----------------
# semantic_ood.class_agnostic_AR_100 is the headline OOD metric. Read it as
# "if I deploy this model in a setting where I'll see unfamiliar species,
# what fraction of them will I at least know are there?".
#
# READ AR, NOT AP.
# ----------------
# class_agnostic_AR_100 is the number to quote. class_agnostic_AP_50 is
# heavily depressed on this benchmark and is NOT a bug:
#
#   * AR is duplicate-immune. Extra boxes on a beetle cannot lower recall.
#   * AP is duplicate-sensitive. Inference ran conf=0.001 with ultralytics'
#     default agnostic_nms=False, so the predictions carry ~21 boxes per GT
#     beetle. Once relabelled to one class, ~20 of every 21 are false
#     positives. COCO scores that correctly, and AP collapses accordingly.
#     See split_results["prediction_density"] for the per-run figure.
#
# DO NOT compare class_agnostic_AP_50 against coco.AP_50 and treat
# class_agnostic < joint as a defect. It is NOT an invariant:
#
#     coco.AP_50            = MACRO average over 65 species (~10 GT each; a
#                             10-GT species can score high cheaply)
#     class_agnostic_AP_50  = MICRO average over ONE class, pooling all 650 GT
#                             and all ~13,200 predictions
#
# A macro-mean over 65 classes and a micro-mean over 1 class are not ordered.
# Chasing that non-invariant is what produced the (unsound) pre-NMS patch that
# [FIX 5B] removes. The sound invariant is: class_agnostic_AR_100 must track
# decomposition.detection_recall, since both answer "ignoring species, did the
# model box the beetle?" -- one via COCOeval, one via greedy matching.
# ============================================================================

def evaluate_class_agnostic(benchmark_dir: Path, split: str, predictions: list,
                            quiet: bool = True) -> dict:
    """Relabel GT and DT to a single class, then run standard COCO eval."""
    if not predictions:
        return {f"class_agnostic_{k}": 0.0 for k in _zero_coco_metrics()}

    coco_path = benchmark_dir / "detection" / "annotations_coco" / f"{split}.json"
    with open(coco_path) as f:
        gt_dict = json.load(f)

    # Strip extension fields, then relabel everything to category 1.
    gt_dict = {
        "info": gt_dict.get("info", {}),
        "licenses": gt_dict.get("licenses", []),
        "categories": [{"id": 1, "name": "beetle", "supercategory": "beetle"}],
        "images": gt_dict["images"],
        "annotations": [
            {**a, "category_id": 1} for a in gt_dict["annotations"]
        ],
    }

    coco_gt_ca = COCO()
    coco_gt_ca.dataset = gt_dict
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt_ca.createIndex()

    # ------------------------------------------------------------------
    # [FIX 5B] DO NOT PRE-NMS HERE. This line has been "fixed" once already
    # and the fix was wrong; the reasoning is recorded so it is not redone.
    #
    # The temptation: predictions carry ~21 boxes per ground-truth beetle
    # (inference ran conf=0.001 with ultralytics' default agnostic_nms=False,
    # which offsets boxes by class index so boxes of different species never
    # suppress one another). Relabelling them all to "beetle" turns those
    # cross-class duplicates into false positives and class-agnostic AP
    # collapses. The obvious repair is to run class-agnostic NMS first.
    #
    # That repair DESTROYS TRUE POSITIVES, provably:
    #
    #   (a) Deleting only FALSE POSITIVES can never lower COCO AP. COCO
    #       matches greedily by score, so removing an FP leaves the TP set and
    #       the recall identical while shortening every prefix of the ranked
    #       list -- precision weakly rises at every recall point. When
    #       class-agnostic NMS was applied here, AP@.5 FELL (0.249 -> 0.198).
    #       The only way that can happen is if TPs were being removed.
    #
    #   (b) Mechanism: greedy NMS keeps the highest-SCORING box in a spatial
    #       cluster; COCO matches on LOCALISATION (IoU >= 0.5). Those agree
    #       only when confidence predicts IoU. Here they are nearly
    #       independent -- class-agnostic AR@1 = 0.66 versus AR@100 = 0.89,
    #       i.e. the top-scoring box in an image is a correct localisation only
    #       two-thirds of the time. That is what the ~6x train/test linear
    #       object-scale shift produces (dense multi-specimen plates in train,
    #       single-specimen photographs in iid_test): the box-regression head
    #       is miscalibrated in scale, so confidence carries little
    #       localisation signal. Whenever a cluster's top-scoring box has
    #       IoU < 0.5 with the GT and suppresses a lower-scoring box that had
    #       IoU >= 0.5, that beetle becomes UNFINDABLE. Measured cost in two
    #       independent simulations calibrated to the real iid_test statistics:
    #       AR@100 0.723 -> 0.628 and 0.732 -> 0.627. ~10 recall points gone.
    #
    #   (c) class_agnostic_AR_100 is the HEADLINE OOD METRIC and the sole input
    #       to summary.species_generalization_gap_AR_100. Pre-NMS corrupts the
    #       benchmark's central result.
    #
    # The duplicates are real, but they are GENUINE false positives and COCO
    # already scores them as such. They cannot be removed post hoc, because no
    # offline rule can know which box localised best. The only sound way to
    # obtain duplicate-free class-agnostic AP is to re-run INFERENCE with
    # agnostic_nms=True. Until that is done, class-agnostic AP is reported
    # as-is and the inflation is disclosed via split_results["prediction_density"].
    # ------------------------------------------------------------------
    preds_ca = [{**p, "category_id": 1} for p in predictions]
    metrics = evaluate_coco_standard(coco_gt_ca, preds_ca, quiet=quiet)
    return {f"class_agnostic_{k}": v for k, v in metrics.items()}


# ============================================================================
# METRIC 4 — CALIBRATION (Expected Calibration Error)
# ============================================================================
#
# WHAT IT MEASURES
# ----------------
# Whether a model's confidence scores match its empirical accuracy. We bin
# predictions by their confidence, then in each bin compute:
#   - mean confidence in the bin
#   - true-positive rate in the bin (fraction matched to GT at IoU >= 0.5)
# ECE is the weighted average of |confidence - TP_rate| across bins.
#
# A perfectly-calibrated model has ECE = 0: when it says 0.8 confidence, it
# is right 80% of the time.
#
# WHY IT MATTERS FOR THIS BENCHMARK
# ---------------------------------
# Crucial on semantic_ood. A well-behaved model SHOULD become less confident
# on unfamiliar species. If a model produces the same confidence on OOD as
# it does on IID, it is overconfident — dangerous for deployment in
# open-world settings where users rely on score thresholds to filter noise.
#
# We compute ECE class-agnostically (matching is by IoU only). On IID
# splits, this is calibration of detection quality. On semantic_ood, this is
# calibration of "object presence" detection. We also report mean
# confidence per split so the OOD-confidence-drop is visible at a glance.
#
# HOW TO INTERPRET
# ----------------
# ECE < 0.05       very well calibrated
# ECE < 0.10       acceptable
# ECE > 0.15       poorly calibrated
# mean_confidence  on semantic_ood should be MEANINGFULLY LOWER than on
#                  iid_test for an OOD-aware model. Drop of 0.1+ is healthy.
# ============================================================================

def compute_calibration(coco_gt: COCO, predictions: list,
                        iou_threshold: float = DEFAULT_IOU,
                        n_bins: int = ECE_BINS,
                        matches: list = None) -> dict:
    # [FIX 5B] `matches` may be precomputed by the caller. match_predictions_to_gt
    # is O(n_preds x n_gt) per image and was previously run FOUR times per split
    # on identical arguments (calibration, hierarchical, decomposition, detection
    # records). On the dense semantic_ood plates that dominated re-score wall
    # time. Passing it in once is a pure speed-up: same arguments, same result.
    if not predictions:
        return {"ECE": 0.0, "mean_confidence": 0.0,
                "confidence_histogram": [0]*n_bins,
                "accuracy_per_bin": [0.0]*n_bins,
                "n_predictions": 0}

    if matches is None:
        matches = match_predictions_to_gt(coco_gt, predictions, iou_threshold)
    if not matches:
        return {"ECE": 0.0, "mean_confidence": 0.0,
                "confidence_histogram": [0]*n_bins,
                "accuracy_per_bin": [0.0]*n_bins,
                "n_predictions": 0}

    confs = np.array([m["pred"]["score"] for m in matches])
    is_tp = np.array([m["gt"] is not None for m in matches], dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    # np.digitize returns indices 1..n_bins for values in (edge[i-1], edge[i]]
    bin_ids = np.clip(np.digitize(confs, bin_edges[1:-1]), 0, n_bins - 1)

    ece = 0.0
    hist = [0] * n_bins
    acc_per_bin = [0.0] * n_bins
    total = len(confs)
    for b in range(n_bins):
        mask = bin_ids == b
        n_b = int(mask.sum())
        hist[b] = n_b
        if n_b == 0:
            continue
        bin_conf = confs[mask].mean()
        bin_acc = is_tp[mask].mean()
        acc_per_bin[b] = float(bin_acc)
        ece += (n_b / total) * abs(bin_conf - bin_acc)

    return {
        "ECE": float(ece),
        "mean_confidence": float(confs.mean()),
        "confidence_histogram": hist,
        "accuracy_per_bin": acc_per_bin,
        "n_predictions": total,
    }


# ============================================================================
# METRIC 5 — HIERARCHICAL / TAXONOMIC GRACEFUL DEGRADATION
# ============================================================================
#
# WHAT IT MEASURES
# ----------------
# For each prediction that overlaps a GT box at IoU >= threshold (regardless
# of class), check how close the prediction's class is to the GT class on
# the taxonomic tree:
#
#   correct_species   model predicted exactly the right species
#   correct_genus     wrong species but same genus
#   correct_tribe     wrong genus but same tribe
#   correct_subfamily wrong tribe but same subfamily
#   wrong             different subfamily
#
# WHY IT MATTERS FOR THIS BENCHMARK
# ---------------------------------
# On semantic_ood the model cannot output the correct species (it wasn't
# trained on those classes). But "it predicted Xylosandrus crassiusculus
# when the GT is Xylosandrus morigerus" is much better than predicting
# something from a different subfamily — it's correct at the genus level
# and tells downstream taxonomic-identification systems something useful.
#
# We also break this down by the distance band of the OOD species:
#   near_genus  the GT species' genus is also in training (best case for
#               the model — it has seen close relatives)
#   mid_tribe   the GT genus is new, but the tribe is shared with training
#   far_tribe   the GT tribe itself is new
#
# A model that does poorly on far_tribe but well on near_genus has not
# learned generic beetle features — it's pattern-matching on genus-level
# cues. A model that does similarly on all three bands is generalizing.
#
# HOW TO INTERPRET
# ----------------
# Read the four taxonomic-accuracy columns for semantic_ood. Higher
# correct_genus or correct_tribe values indicate useful taxonomic
# generalization. A model with low correct_species but high correct_genus
# on semantic_ood is a strong candidate for downstream
# expert-loop identification pipelines.
# ============================================================================

def evaluate_hierarchical(coco_gt: COCO, predictions: list, taxonomy: pd.DataFrame,
                          global_cat_lookup: dict,
                          iou_threshold: float = DEFAULT_IOU,
                          matches: list = None) -> dict:
    """For matched predictions, count agreement at species/genus/tribe/subfamily.

    `global_cat_lookup` is a {cat_id: species_name} map spanning ALL benchmark
    splits — required because predictions on OOD splits use trainable category
    IDs that aren't in the OOD GT's categories list."""
    if not predictions:
        return {"n_matched": 0, "correct_species": 0.0, "correct_genus": 0.0,
                "correct_tribe": 0.0, "correct_subfamily": 0.0,
                "per_band": {}}

    if matches is None:
        matches = match_predictions_to_gt(coco_gt, predictions, iou_threshold)
    tp_matches = [m for m in matches if m["gt"] is not None]

    tax_by_name = taxonomy.set_index("scientificName")[
        ["subfamily", "tribe", "genus"]].to_dict("index")

    def get_tax(species_name):
        return tax_by_name.get(species_name, {"subfamily": None, "tribe": None, "genus": None})

    total = len(tp_matches)
    if total == 0:
        return {"n_matched": 0, "correct_species": 0.0, "correct_genus": 0.0,
                "correct_tribe": 0.0, "correct_subfamily": 0.0, "per_band": {}}

    band_lookup = (taxonomy.set_index("scientificName")["distance_band_vs_trainable"]
                   if "distance_band_vs_trainable" in taxonomy.columns
                   else pd.Series(dtype=str))

    per_band_counts = defaultdict(lambda: {
        "n": 0, "correct_species": 0, "correct_genus": 0,
        "correct_tribe": 0, "correct_subfamily": 0
    })
    correct_s = correct_g = correct_t = correct_sf = 0

    for m in tp_matches:
        # Resolve names via the GLOBAL lookup (spans all splits' categories).
        # This is what makes OOD hierarchical metrics work: predictions on
        # semantic_ood use trainable IDs, GT uses OOD IDs — both resolve here.
        gt_cid = int(m["gt"]["category_id"])
        pred_cid = int(m["pred"]["category_id"])
        gt_name = global_cat_lookup.get(gt_cid)
        pred_name = global_cat_lookup.get(pred_cid)

        if gt_name is None:
            continue
        gt_tax = get_tax(gt_name)
        pred_tax = get_tax(pred_name) if pred_name else {"subfamily": None, "tribe": None, "genus": None}

        is_sp = pred_name is not None and pred_name == gt_name
        is_g  = pred_tax["genus"]     is not None and pred_tax["genus"]     == gt_tax["genus"]
        is_t  = pred_tax["tribe"]     is not None and pred_tax["tribe"]     == gt_tax["tribe"]
        is_sf = pred_tax["subfamily"] is not None and pred_tax["subfamily"] == gt_tax["subfamily"]

        correct_s  += int(is_sp)
        correct_g  += int(is_g)
        correct_t  += int(is_t)
        correct_sf += int(is_sf)

        band = band_lookup.get(gt_name, "")
        if band and isinstance(band, str) and band.strip():
            b = per_band_counts[band]
            b["n"] += 1
            b["correct_species"]   += int(is_sp)
            b["correct_genus"]     += int(is_g)
            b["correct_tribe"]     += int(is_t)
            b["correct_subfamily"] += int(is_sf)

    per_band = {}
    for band, c in per_band_counts.items():
        if c["n"] == 0:
            continue
        per_band[band] = {
            "n": c["n"],
            "correct_species":   c["correct_species"]   / c["n"],
            "correct_genus":     c["correct_genus"]     / c["n"],
            "correct_tribe":     c["correct_tribe"]     / c["n"],
            "correct_subfamily": c["correct_subfamily"] / c["n"],
        }

    return {
        "n_matched": total,
        "correct_species":   correct_s  / total,
        "correct_genus":     correct_g  / total,
        "correct_tribe":     correct_t  / total,
        "correct_subfamily": correct_sf / total,
        "per_band": per_band,
    }


# ============================================================================
# METRIC 5b — OOD PERFORMANCE BY TAXONOMIC DISTANCE BAND
# ============================================================================
#
# WHAT IT MEASURES
# ----------------
# For the semantic_ood split, breaks performance down by how far each
# held-out species sits from the training set's taxonomic neighborhood:
#     near_genus   the species' genus IS in the trainable set (closest)
#     mid_tribe    new genus, but the tribe is shared with training
#     far_tribe    the tribe itself is new (furthest)
#
# For each band we report:
#     class_agnostic_AP_50     Does the model FIND these beetles at all?
#                              (Relabel everything to "beetle" and score
#                              detection.) This is the headline detection
#                              capability per band.
#     class_agnostic_AR_100    Recall — fraction of GT boxes the model
#                              produced any localization for.
#     correct_genus/tribe/subfamily   Given a TP localization, how often
#                              is the predicted species in the same genus /
#                              tribe / subfamily as the GT?
#     mean_confidence_matched  Mean prediction confidence on TP detections.
#                              Tells you whether the model correctly
#                              expresses more uncertainty on more distant
#                              species (well-calibrated OOD = confidence
#                              should drop near_genus → far_tribe).
#
# The key derived signals are the gaps between near_genus and far_tribe in
# detection capability and in taxonomic-agreement. A model that finds
# near_genus species fine but struggles on far_tribe is pattern-matching
# on genus-level cues; a model that's flat across bands has learned
# generic beetle features. These two gaps end up in the cross-split
# summary as `ood_band_gap_AP_50` and `ood_band_gap_correct_genus`.
#
# WHY IT MATTERS FOR THIS BENCHMARK
# ---------------------------------
# Two systems with the same overall semantic_ood score can have very
# different deployment characteristics. One might do well on near_genus
# but fail on far_tribe — that's near-useless for invasive-species
# detection, where truly novel species (far_tribe) are the threat. This
# band breakdown surfaces that asymmetry directly.
#
# HOW TO INTERPRET
# ----------------
# Look for a steep decline from near_genus → far_tribe in
# class_agnostic_AP_50. A flat profile = good OOD generalization. A
# steep gradient = the model relies on having seen related genera and
# won't generalize to truly novel taxa.
#
# Pair with `mean_confidence_matched` per band: ideally the model is BOTH
# less likely to find a far_tribe beetle AND less confident when it does
# — the latter is the calibration signal that downstream filtering can
# leverage. A model that's equally confident on near_genus and far_tribe
# is overconfident on the latter.
# ============================================================================

def evaluate_ood_by_band(benchmark_dir: Path, split: str, predictions: list,
                         taxonomy: pd.DataFrame, global_cat_lookup: dict,
                         iou_threshold: float = DEFAULT_IOU,
                         quiet: bool = True) -> dict:
    """For each taxonomic distance band, compute detection capability +
    taxonomic agreement + calibration. Meaningful only on splits whose
    species have `distance_band_vs_trainable` populated (i.e. semantic_ood).
    Returns {"by_band": {}} for any other split."""
    if "distance_band_vs_trainable" not in taxonomy.columns:
        return {"by_band": {}}

    coco_path = benchmark_dir / "detection" / "annotations_coco" / f"{split}.json"
    with open(coco_path) as f:
        gt_dict = json.load(f)

    # Map this split's category IDs -> band via species name
    name_to_band = (taxonomy.set_index("scientificName")["distance_band_vs_trainable"]
                    .to_dict())
    cat_id_to_band = {}
    for cat in gt_dict["categories"]:
        band = name_to_band.get(cat["name"])
        if isinstance(band, str) and band.strip():
            cat_id_to_band[cat["id"]] = band

    if not cat_id_to_band:
        # No band info for this split's species — silently return empty
        return {"by_band": {}}

    # Group GT annotations by band; track images and species per band
    anns_by_band = defaultdict(list)
    image_ids_by_band = defaultdict(set)
    species_by_band = defaultdict(set)
    for ann in gt_dict["annotations"]:
        band = cat_id_to_band.get(ann["category_id"])
        if band:
            anns_by_band[band].append(ann)
            image_ids_by_band[band].add(ann["image_id"])
            species_by_band[band].add(ann["category_id"])

    # Taxonomy table for genus/tribe/subfamily lookups on TP matches
    tax_by_name = taxonomy.set_index("scientificName")[
        ["subfamily", "tribe", "genus"]].to_dict("index")

    def _get_tax(name):
        return tax_by_name.get(name, {"subfamily": None, "tribe": None, "genus": None})

    by_band = {}
    for band, anns in anns_by_band.items():
        image_ids = image_ids_by_band[band]

        # ---- Detection capability: class-agnostic AP on this band's images ----
        # Build a class-agnostic sub-COCO restricted to this band's images
        # and annotations. Score the model's predictions on those images.
        sub_gt_dict_ca = {
            "info":       gt_dict.get("info", {}),
            "licenses":   gt_dict.get("licenses", []),
            "categories": [{"id": 1, "name": "beetle", "supercategory": "beetle"}],
            "images":     [img for img in gt_dict["images"] if img["id"] in image_ids],
            "annotations":[{**a, "category_id": 1} for a in anns],
        }
        coco_gt_ca = COCO()
        coco_gt_ca.dataset = sub_gt_dict_ca
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt_ca.createIndex()

        # [FIX 5B] DO NOT PRE-NMS HERE EITHER -- see evaluate_class_agnostic().
        # This feeds class_agnostic_AR_100 per band, hence ood_band_gap_AR_100.
        preds_ca = [{**p, "category_id": 1} for p in predictions
                    if p["image_id"] in image_ids]

        if preds_ca:
            det_metrics = evaluate_coco_standard(coco_gt_ca, preds_ca, quiet=quiet)
        else:
            det_metrics = _zero_coco_metrics()

        # ---- Taxonomic agreement + confidence on TP matches in this band ----
        # Use the original (un-relabeled) sub-GT and original predictions so
        # we can read back what species the model actually predicted.
        sub_gt_dict_orig = {
            "info":       gt_dict.get("info", {}),
            "licenses":   gt_dict.get("licenses", []),
            "categories": gt_dict["categories"],
            "images":     sub_gt_dict_ca["images"],
            "annotations":anns,
        }
        coco_gt_orig = COCO()
        coco_gt_orig.dataset = sub_gt_dict_orig
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt_orig.createIndex()

        preds_band = [p for p in predictions if p["image_id"] in image_ids]
        matches = match_predictions_to_gt(coco_gt_orig, preds_band, iou_threshold)
        tp_matches = [m for m in matches if m["gt"] is not None]

        n_tp = len(tp_matches)
        if n_tp > 0:
            correct_g = correct_t = correct_sf = 0
            tp_confs = []
            for m in tp_matches:
                gt_cid   = int(m["gt"]["category_id"])
                pred_cid = int(m["pred"]["category_id"])
                gt_name   = global_cat_lookup.get(gt_cid)
                pred_name = global_cat_lookup.get(pred_cid)
                if gt_name is None:
                    continue
                gt_tax   = _get_tax(gt_name)
                pred_tax = _get_tax(pred_name) if pred_name else \
                           {"subfamily": None, "tribe": None, "genus": None}
                correct_g  += int(pred_tax["genus"]     is not None and
                                  pred_tax["genus"]     == gt_tax["genus"])
                correct_t  += int(pred_tax["tribe"]     is not None and
                                  pred_tax["tribe"]     == gt_tax["tribe"])
                correct_sf += int(pred_tax["subfamily"] is not None and
                                  pred_tax["subfamily"] == gt_tax["subfamily"])
                tp_confs.append(float(m["pred"]["score"]))
            correct_genus    = correct_g  / n_tp
            correct_tribe    = correct_t  / n_tp
            correct_subfam   = correct_sf / n_tp
            mean_conf_matched = sum(tp_confs) / len(tp_confs)
        else:
            correct_genus = correct_tribe = correct_subfam = 0.0
            mean_conf_matched = 0.0

        # Mean confidence over ALL predictions on this band's images (not just
        # those that hit a TP). Useful for catching overconfidence on far_tribe.
        all_confs = [float(p["score"]) for p in preds_band]
        mean_conf_all = sum(all_confs) / len(all_confs) if all_confs else 0.0

        by_band[band] = {
            "n_species":                 len(species_by_band[band]),
            "n_annotations":             len(anns),
            "n_images":                  len(image_ids),
            "class_agnostic_AP_50":      det_metrics.get("AP_50",    0.0),
            "class_agnostic_AP_50_95":   det_metrics.get("AP_50_95", 0.0),
            "class_agnostic_AR_100":     det_metrics.get("AR_100",   0.0),
            "n_matched_tp":              n_tp,
            "correct_genus":             correct_genus,
            "correct_tribe":             correct_tribe,
            "correct_subfamily":         correct_subfam,
            "mean_confidence_matched":   mean_conf_matched,
            "mean_confidence_all_preds": mean_conf_all,
        }

    # Order bands near -> far so the report reads naturally
    band_order = ["near_genus", "mid_tribe", "far_tribe"]
    ordered = {b: by_band[b] for b in band_order if b in by_band}
    # Append any unrecognized bands at the end (defensive)
    for b, v in by_band.items():
        if b not in ordered:
            ordered[b] = v

    return {"by_band": ordered}


# ============================================================================
# METRIC 6 — CLASS IMBALANCE TOLERANCE
# ============================================================================
#
# WHAT IT MEASURES
# ----------------
# How evenly the model performs across species with very different training
# data quantities. The benchmark's training set is naturally imbalanced —
# some species have thousands of training images, others barely meet the
# 10-specimen trainable threshold. A model that performs well on
# data-rich species but collapses on data-poor ones is *not* tolerant of
# class imbalance.
#
# We bin trainable species into three tiers by their training image count:
#
#   head    top 25% by training data (most images)
#   medium  middle 50%
#   tail    bottom 25% (fewest images)
#
# Then compute mean AP@.5 and AP@[.5:.95] within each tier, plus the
# Pearson correlation between a species' test AP and its training count.
#
# WHY IT MATTERS FOR THIS BENCHMARK
# ---------------------------------
# Real-world species observation data is heavily long-tailed. Bark and
# ambrosia beetles are no exception — common pest species accumulate
# many images, while rare native species lag far behind. A practitioner
# choosing between two models wants to know: does this model treat rare
# species fairly, or does it pattern-match on the abundant ones?
#
# Two key derived metrics:
#
#   head_minus_tail_AP_50  the AP@.5 gap between head and tail tiers.
#                          Large positive value (>0.25) = model is rich-class biased.
#                          Near zero or slightly positive = robust to imbalance.
#                          Negative = unusual; rare classes outperform common
#                          ones (often a sign of overfitting on common classes).
#
#   AP_50_train_count_correlation  Pearson r between per-species AP and
#                                  training image count, across all eval'd species.
#                                  r > 0.6  = AP strongly tracks data abundance
#                                  r ≈ 0.0  = no relationship — imbalance-robust
#                                  r < 0    = inverse; rare classes do BETTER (rare)
#
# HOW TO INTERPRET
# ----------------
# - On iid_test, the head-tail gap tells you the raw imbalance bias of the
#   training procedure.
# - On inat_test, the gap should be similar; if it widens, photography shift
#   hurts rare species more than common ones.
# - On semantic_ood, this metric is computed via class-agnostic matching
#   (since species-correct AP is necessarily zero) and tells you whether
#   the model's detection capability degrades on species whose nearest
#   trained relatives were data-poor.
# ============================================================================

def evaluate_class_imbalance(per_species_results: dict, train_counts: dict) -> dict:
    """Compute head/medium/tail AP and AP-vs-training-count correlation."""
    if not per_species_results or not train_counts:
        return {}

    rows = []
    for cid, info in per_species_results.items():
        sp = info["name"]
        n_train = train_counts.get(sp, 0)
        if n_train > 0:
            rows.append({
                "species": sp,
                "n_train_images": int(n_train),
                "AP_50":    float(info.get("AP_50",    0.0)),
                "AP_50_95": float(info.get("AP_50_95", 0.0)),
            })

    if len(rows) < 4:  # need at least 4 species to compute quartiles meaningfully
        return {"n_species_evaluated": len(rows)}

    df = pd.DataFrame(rows).sort_values("n_train_images")

    # Tiers by training image count quartile.
    # qcut may produce duplicate edges if many species have the same count;
    # use rank-based binning to avoid that crash mode.
    df["rank"] = df["n_train_images"].rank(method="first")
    n = len(df)
    q1 = n * 0.25
    q3 = n * 0.75
    def tier(r):
        if r <= q1:    return "tail"
        if r >  q3:    return "head"
        return "medium"
    df["tier"] = df["rank"].apply(tier)

    out = {"n_species_evaluated": n}
    for t in ["head", "medium", "tail"]:
        sub = df[df["tier"] == t]
        if len(sub) == 0:
            continue
        out[f"{t}_n_species"]     = int(len(sub))
        out[f"{t}_AP_50"]         = float(sub["AP_50"].mean())
        out[f"{t}_AP_50_95"]      = float(sub["AP_50_95"].mean())
        out[f"{t}_min_train_imgs"] = int(sub["n_train_images"].min())
        out[f"{t}_max_train_imgs"] = int(sub["n_train_images"].max())

    if "head_AP_50" in out and "tail_AP_50" in out:
        out["head_minus_tail_AP_50"]    = out["head_AP_50"]    - out["tail_AP_50"]
        out["head_minus_tail_AP_50_95"] = out["head_AP_50_95"] - out["tail_AP_50_95"]

    # Pearson correlation: how strongly does AP track training data quantity?
    if n >= 3:
        corr = df[["n_train_images", "AP_50", "AP_50_95"]].corr()
        r_50    = float(corr.loc["n_train_images", "AP_50"])
        r_50_95 = float(corr.loc["n_train_images", "AP_50_95"])
        # Guard against NaN from zero variance
        out["AP_50_train_count_correlation"]    = r_50    if not np.isnan(r_50)    else 0.0
        out["AP_50_95_train_count_correlation"] = r_50_95 if not np.isnan(r_50_95) else 0.0

    return out


# ============================================================================
# METRIC 7 — DETECTION / CLASSIFICATION DECOMPOSITION
# ============================================================================
#
# WHAT IT MEASURES
# ----------------
# Standard detection mAP combines two abilities into one score: "did you
# find the object?" (detection) and "did you label it correctly?"
# (classification). When mAP is low, knowing whether the failure is in
# detection or classification matters — they have completely different
# fixes.
#
# This section produces three labeled headline numbers per split, plus
# a derived "classification loss ratio":
#
#   detection_recall
#       Fraction of ground-truth beetles the model localized at all,
#       ignoring species (greedy class-agnostic matching at IoU ≥ 0.5).
#       This is the detection ability. It is NMS-free and duplicate-immune.
#
#   acc_species_given_det
#       Of the boxes that ARE true detections, what fraction carry the
#       right species? This is the classification ability, conditioned on
#       having found the object.
#
#   species_recall = detection_recall × acc_species_given_det
#       Found AND correctly labelled, as a fraction of all GT.
#
#   classification_loss = detection_recall − species_recall
#       The share of ground truth the model FOUND but MISLABELLED. This is
#       the number that says how much is being lost to the classifier.
#
# NOTE — the AP-ratio form (1 − joint_AP / class_agnostic_AP) that this
# section used to carry was REMOVED by [FIX 1] and must not come back.
# class-agnostic AP is not a detection upper bound: relabelling every box to
# one class turns the ~21 overlapping hypotheses on each beetle into false
# positives, which pushes class-agnostic AP BELOW the per-class mAP and makes
# the ratio clamp to zero. detection_only_AP_50_ref is retained for continuity
# but is explicitly labelled reference-only, NOT an upper bound.
#
# WHY THIS DECOMPOSITION MATTERS
# ------------------------------
# Compare two hypothetical models with the same mAP@.5 = 0.40:
#
#   Model A:  detection_only = 0.95, joint = 0.40, loss_ratio = 0.58
#       This model finds beetles well but confuses species. The fix is
#       a better classification head, better feature separation between
#       similar species, or per-class focal loss. Detection backbone is
#       fine.
#
#   Model B:  detection_only = 0.45, joint = 0.40, loss_ratio = 0.11
#       This model is bad at finding beetles in the first place. Its
#       classifier is fine — when it finds a beetle, it usually labels
#       it right. The fix is a better detector / different anchor
#       configuration / more training data, NOT a better classifier.
#
# Same headline mAP, completely different remediation paths. This
# section makes that visible.
#
# This is only computed on IID-species splits (iid_test, inat_test). On
# semantic_ood the model cannot get the species correct by construction,
# so classification_given_detection is necessarily zero; see
# hierarchical.correct_genus / correct_tribe for the meaningful
# OOD-classification signal.
# ============================================================================

def compute_decomposition(coco_gt, predictions, taxonomy, global_cat_lookup,
                          split_name, split_results, iou_threshold=DEFAULT_IOU,
                          matches=None):
    """[FIX 1] Sound detection-vs-classification split, valid on EVERY split.

    The previous version used 1 - joint_AP/class_agnostic_AP. class-agnostic AP
    is NOT a valid detection upper bound: relabeling every box to one class turns
    the top-k overlapping hypotheses on one beetle into false positives that push
    class-agnostic AP BELOW the per-class mAP, so the ratio clamps to 0. This
    recomputes from the matching directly:

        detection_recall        matched_GT / total_GT  (class-agnostic)
        acc_species_given_det   correct-species among matched TPs
        species_recall          found AND correctly labelled / total_GT
        classification_loss     detection_recall - species_recall

    On semantic_ood, species accuracy is ~0 by construction; read the genus/tribe
    variants (graceful degradation). The AP reference values are retained but
    labelled reference-only, not an upper bound."""
    n_gt = _n_real_gt(coco_gt)
    if not predictions or n_gt == 0:
        return {}
    tax_by_name = taxonomy.set_index("scientificName")[["subfamily", "tribe", "genus"]].to_dict("index")
    def tx(n): return tax_by_name.get(n, {"subfamily": None, "tribe": None, "genus": None})
    if matches is None:
        matches = match_predictions_to_gt(coco_gt, predictions, iou_threshold)
    tp = [m for m in matches if m["gt"] is not None]
    matched = len(tp); cs = cg = ct = 0
    for m in tp:
        gtn = global_cat_lookup.get(int(m["gt"]["category_id"]))
        prn = global_cat_lookup.get(int(m["pred"]["category_id"]))
        if gtn is None:
            continue
        gt_t, pr_t = tx(gtn), (tx(prn) if prn else tx(None))
        cs += int(prn is not None and prn == gtn)
        cg += int(pr_t["genus"] is not None and pr_t["genus"] == gt_t["genus"])
        ct += int(pr_t["tribe"] is not None and pr_t["tribe"] == gt_t["tribe"])
    det_recall = matched / n_gt
    out = {"n_gt": n_gt, "n_matched_tp": matched,
           "detection_recall": det_recall,
           "acc_species_given_det": (cs / matched) if matched else 0.0,
           "acc_genus_given_det":   (cg / matched) if matched else 0.0,
           "acc_tribe_given_det":   (ct / matched) if matched else 0.0,
           "species_recall": cs / n_gt, "genus_recall": cg / n_gt, "tribe_recall": ct / n_gt,
           "classification_loss": det_recall - cs / n_gt,
           "is_species_meaningful": split_name in IN_DISTRIBUTION_SPECIES_SPLITS}
    # AP reference values (NOT an upper bound; kept for continuity)
    det_ap = split_results.get("class_agnostic", {}).get("class_agnostic_AP_50")
    joint = split_results.get("coco", {}).get("AP_50")
    if det_ap is not None:
        out["detection_only_AP_50_ref"] = float(det_ap)
    if joint is not None:
        out["joint_AP_50"] = float(joint)
    return out


def build_detection_records(matches, taxonomy, global_cat_lookup, split,
                            band_lookup=None, keep_fp_sample=0.0, rng_seed=0):
    """[FIX 3] Per-detection rows (TP-only by default). The substrate for a real
    confusion matrix, calibration at any threshold, and novelty AUROC. Also
    returns matched-TP confidences (overall + per band on OOD) for FIX 4."""
    import random
    rng = random.Random(rng_seed)
    tax_by_name = taxonomy.set_index("scientificName")[["subfamily", "tribe", "genus"]].to_dict("index")
    def tx(n): return tax_by_name.get(n, {"subfamily": None, "tribe": None, "genus": None})
    band_lookup = band_lookup or {}
    recs = []; tp_confs = []; tp_confs_by_band = defaultdict(list)
    for m in matches:
        is_tp = m["gt"] is not None
        if not is_tp and (keep_fp_sample <= 0 or rng.random() > keep_fp_sample):
            continue
        prn = global_cat_lookup.get(int(m["pred"]["category_id"]))
        gtn = global_cat_lookup.get(int(m["gt"]["category_id"])) if is_tp else None
        pr_t, gt_t = tx(prn), tx(gtn)
        band = band_lookup.get(gtn) if gtn else None
        recs.append({"split": split, "image_id": m["pred"]["image_id"], "score": m["pred"]["score"],
                     "iou": m["iou"], "is_tp": int(is_tp), "pred_species": prn, "gt_species": gtn,
                     "pred_genus": pr_t["genus"], "gt_genus": gt_t["genus"],
                     "correct_species": int(is_tp and prn == gtn),
                     "correct_genus": int(is_tp and pr_t["genus"] is not None and pr_t["genus"] == gt_t["genus"]),
                     "band": band})
        if is_tp:
            tp_confs.append(float(m["pred"]["score"]))
            if band:
                tp_confs_by_band[band].append(float(m["pred"]["score"]))
    return recs, tp_confs, dict(tp_confs_by_band)


def _rank_auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]); order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); sa = allv[order]; i = 0; r = np.empty(len(allv))
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sa[j + 1] == sa[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    ranks[order] = r
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _fpr_at_tpr(pos, neg, tpr=0.95):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean(neg >= np.quantile(pos, 1 - tpr)))


def compute_novelty_metrics(per_split):
    """[FIX 4] COMPUTED known-vs-novel separation, replacing the figure-cited
    AUROC. Novelty score = 1 - confidence; positive class = novel (semantic_ood),
    negatives = known (iid_test). Reports detection-level AUROC/FPR@95 overall
    and per distance band, so 'near-relatives are hardest to flag' is measured."""
    iid = per_split.get("iid_test", {}); sood = per_split.get("semantic_ood", {})
    known = [1 - c for c in iid.get("_tp_confs", [])]
    novel = [1 - c for c in sood.get("_tp_confs", [])]
    out = {}
    if known and novel:
        out["novelty_det_AUROC"] = _rank_auroc(novel, known)
        out["novelty_det_FPR@95TPR"] = _fpr_at_tpr(novel, known)
        out["novelty_known_mean_conf"] = float(np.mean([1 - x for x in known]))
        out["novelty_novel_mean_conf"] = float(np.mean([1 - x for x in novel]))
        out["n_known_tp"] = len(known); out["n_novel_tp"] = len(novel)
    for band, confs in (sood.get("_tp_confs_by_band", {}) or {}).items():
        nb = [1 - c for c in confs]
        if nb and known:
            out[f"novelty_det_AUROC_{band}"] = _rank_auroc(nb, known)
            out[f"novelty_novel_mean_conf_{band}"] = float(np.mean(confs))
    return out


# ============================================================================
# METRIC 8 — CROSS-SPLIT SUMMARY METRICS
# ============================================================================
#
# WHAT THEY MEASURE
# -----------------
# These are derived metrics that compare a model's behavior across splits.
# They are the headline numbers for assessing OVERALL benchmark performance
# and should be the first three numbers reported when comparing models.
#
#   photography_robustness_gap = iid_test.AP_50 − inat_test.AP_50
#       The drop in detection quality when moving from institutional
#       photography to field iNaturalist photography for the SAME species.
#       Smaller = more photography-robust. Zero or negative is possible
#       (rare; would indicate iNat photos are somehow easier, e.g.
#       because field beetles are larger in the frame).
#
#   species_generalization_gap = iid_test.class_agnostic.AR_100 −
#                                semantic_ood.class_agnostic.AR_100
#       The drop in "did the model find any beetle?" performance when
#       moving from known species to unknown species. Class-agnostic
#       because predicting the right SPECIES is impossible on OOD by
#       construction. Smaller = better species-level generalization.
#
#   confidence_drop_OOD = mean_confidence(iid_test) −
#                         mean_confidence(semantic_ood)
#       How much less confident the model becomes on unseen species.
#       POSITIVE is good — it indicates the model knows what it
#       doesn't know. Near zero or negative is problematic — it means
#       the model is overconfident on unfamiliar inputs, which is
#       dangerous for deployment.
#
# These three together describe a model's behavior on three orthogonal
# axes: in-distribution detection quality, photography robustness,
# species generalization, and calibration awareness.
# ============================================================================

def compute_summary(per_split: dict) -> dict:
    summary = {}

    iid = per_split.get("iid_test", {})
    inat = per_split.get("inat_test", {})
    sood = per_split.get("semantic_ood", {})

    # Photography robustness — GLOBAL gap across full splits.
    # Note: iid_test contains 65 species, inat_test typically contains a
    # smaller subset (only species with enough iNat data). This gap is
    # confounded by the species mix; see *_shared_species below for the
    # controlled version.
    if iid and inat and iid.get("coco") and inat.get("coco"):
        summary["photography_robustness_gap_AP_50"] = (
            iid["coco"]["AP_50"] - inat["coco"]["AP_50"])
        summary["photography_robustness_gap_AP_50_95"] = (
            iid["coco"]["AP_50_95"] - inat["coco"]["AP_50_95"])

    # Photography robustness — CONTROLLED gap on species that appear in BOTH
    # iid_test and inat_test. This isolates the photography-environment effect
    # from species-mix differences. We use per-species AP from each split,
    # take the intersection of species, and average the per-species deltas.
    if iid.get("per_species") and inat.get("per_species"):
        iid_by_name  = {info["name"]: info for info in iid["per_species"].values()}
        inat_by_name = {info["name"]: info for info in inat["per_species"].values()}
        shared = sorted(set(iid_by_name) & set(inat_by_name))
        if shared:
            d50 = [iid_by_name[s]["AP_50"]    - inat_by_name[s]["AP_50"]    for s in shared]
            d5095 = [iid_by_name[s]["AP_50_95"] - inat_by_name[s]["AP_50_95"] for s in shared]
            summary["photography_robustness_gap_AP_50_shared_species"] = float(np.mean(d50))
            summary["photography_robustness_gap_AP_50_95_shared_species"] = float(np.mean(d5095))
            summary["n_shared_species_iid_inat"] = len(shared)

    # Species generalization (class-agnostic)
    if iid and sood and iid.get("class_agnostic") and sood.get("class_agnostic"):
        summary["species_generalization_gap_AR_100"] = (
            iid["class_agnostic"]["class_agnostic_AR_100"] -
            sood["class_agnostic"]["class_agnostic_AR_100"])
        summary["species_generalization_gap_AP_50"] = (
            iid["class_agnostic"]["class_agnostic_AP_50"] -
            sood["class_agnostic"]["class_agnostic_AP_50"])

    # Calibration drop
    if iid and sood and iid.get("calibration") and sood.get("calibration"):
        summary["confidence_drop_OOD"] = (
            iid["calibration"]["mean_confidence"] -
            sood["calibration"]["mean_confidence"])

    # OOD distance-band gradient: near_genus vs far_tribe headline gaps.
    # Larger positive values = the model degrades sharply as species get more
    # taxonomically distant from training (i.e. it relies on having seen
    # close relatives). Zero or negative = flat performance across bands =
    # good OOD generalization.
    sood_band = sood.get("ood_by_band", {}).get("by_band", {}) if sood else {}
    if "near_genus" in sood_band and "far_tribe" in sood_band:
        n = sood_band["near_genus"]
        f = sood_band["far_tribe"]
        summary["ood_band_gap_AP_50"]          = n["class_agnostic_AP_50"]    - f["class_agnostic_AP_50"]
        summary["ood_band_gap_AR_100"]         = n["class_agnostic_AR_100"]   - f["class_agnostic_AR_100"]
        summary["ood_band_gap_correct_genus"]  = n["correct_genus"]            - f["correct_genus"]
        summary["ood_band_gap_correct_tribe"]  = n["correct_tribe"]            - f["correct_tribe"]
        summary["ood_band_gap_confidence"]     = n["mean_confidence_matched"]  - f["mean_confidence_matched"]

    # [FIX 4] computed known-vs-novel separation (replaces the figure-cited AUROC)
    summary.update(compute_novelty_metrics(per_split))
    return summary


# ============================================================================
# REPORTING
# ============================================================================

def write_summary_json(results: dict, output_dir: Path, model_name: str) -> Path:
    """Full machine-readable dump of every metric."""
    path = output_dir / f"{model_name}_summary.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def write_summary_csv(results: dict, output_dir: Path, model_name: str) -> Path:
    """Flat key/value table for easy comparison across models."""
    rows = []
    for split, split_results in results.get("per_split", {}).items():
        for category, metrics in split_results.items():
            if not isinstance(metrics, dict):
                continue
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    rows.append({"model": model_name, "split": split,
                                 "category": category, "metric": name,
                                 "value": value})
    for k, v in results.get("summary", {}).items():
        if isinstance(v, (int, float)):
            rows.append({"model": model_name, "split": "ALL",
                         "category": "summary", "metric": k, "value": v})
    df = pd.DataFrame(rows)
    path = output_dir / f"{model_name}_summary.csv"
    df.to_csv(path, index=False)
    return path


def write_per_species_csv(results: dict, output_dir: Path, model_name: str) -> Path:
    """Per-species AP for diagnosing weak categories. One row per (split, species).
    [FIX 2] now also carries per-species training image/object counts."""
    train_images = results.get("_train_images", {}) or {}
    train_objects = results.get("_train_objects", {}) or {}
    objects_per_image = results.get("_objects_per_image", {}) or {}
    rows = []
    for split, split_results in results.get("per_split", {}).items():
        for cid, info in split_results.get("per_species", {}).items():
            sp = info["name"]
            rows.append({"model": model_name, "split": split,
                         "category_id": cid, "species": sp,
                         "n_gt_annotations": info["n_gt"],
                         "AP_50": info["AP_50"], "AP_50_95": info["AP_50_95"],
                         "n_train_images": train_images.get(sp),
                         "n_train_objects": train_objects.get(sp),
                         "objects_per_image": objects_per_image.get(sp)})
    df = pd.DataFrame(rows)
    path = output_dir / f"{model_name}_per_species.csv"
    df.to_csv(path, index=False)
    return path


def write_detections_csv(results: dict, output_dir: Path, model_name: str) -> Path:
    """[FIX 3] Write the per-detection TP table assembled in run_evaluation."""
    recs = results.get("_detection_records", [])
    path = output_dir / f"{model_name}_detections.csv"
    pd.DataFrame(recs).to_csv(path, index=False)
    return path


def render_report(results: dict, model_name: str) -> str:
    """Human-readable text report."""
    lines = [
        "=" * 78,
        f"Bark & Ambrosia Beetle Benchmark — Evaluation Report",
        f"Model: {model_name}",
        "=" * 78,
        "",
    ]

    # Headline summary first — most important numbers
    s = results.get("summary", {})
    lines.append("HEADLINE METRICS")
    lines.append("-" * 40)
    for split in ["iid_test", "inat_test", "semantic_ood"]:
        sr = results.get("per_split", {}).get(split, {})
        if not sr:
            continue
        ap5095 = sr.get("coco", {}).get("AP_50_95", float("nan"))
        ap50   = sr.get("coco", {}).get("AP_50",    float("nan"))
        cagn   = sr.get("class_agnostic", {}).get("class_agnostic_AR_100", float("nan"))
        lines.append(f"  {split:14s}  AP@[.5:.95]={ap5095:.3f}  AP@.5={ap50:.3f}  "
                     f"class-agnostic AR@100={cagn:.3f}")
    lines.append("")
    lines.append("CROSS-SPLIT GAPS")
    lines.append("-" * 40)
    for k, v in s.items():
        if isinstance(v, (int, float)):
            lines.append(f"  {k:42s} {v:+.4f}")
    lines.append("")

    # Per-split detail
    for split in ["iid_test", "inat_test", "semantic_ood", "train"]:
        sr = results.get("per_split", {}).get(split)
        if not sr:
            continue
        lines.append("=" * 78)
        lines.append(f"  SPLIT: {split}")
        lines.append("=" * 78)
        if sr.get("warnings"):
            for w in sr["warnings"]:
                lines.append(f"  [warning] {w}")
            lines.append("")

        pd_ = sr.get("prediction_density", {})
        if pd_:
            lines.append(f"  Prediction density: {pd_['n_predictions']:,} boxes / "
                         f"{pd_['n_gt']:,} GT = {pd_['boxes_per_gt']:.1f} boxes per beetle")
            lines.append(f"    -> class-agnostic AP is depressed by these duplicates BY DESIGN")
            lines.append(f"       (conf=0.001, agnostic_nms=False). Quote AR@100, not AP.")
            lines.append("")

        lines.append("  Standard COCO mAP suite:")
        for k, v in sr.get("coco", {}).items():
            lines.append(f"    {k:14s} {v:.4f}")

        lines.append("\n  Class-agnostic detection (ignores predicted class):")
        for k, v in sr.get("class_agnostic", {}).items():
            lines.append(f"    {k:30s} {v:.4f}")

        cal = sr.get("calibration", {})
        if cal:
            lines.append("\n  Calibration:")
            lines.append(f"    ECE             {cal['ECE']:.4f}")
            lines.append(f"    mean confidence {cal['mean_confidence']:.4f}")
            lines.append(f"    n_predictions   {cal['n_predictions']}")

        hier = sr.get("hierarchical", {})
        if hier and hier.get("n_matched", 0) > 0:
            lines.append(f"\n  Taxonomic graceful degradation "
                         f"(among {hier['n_matched']} matched predictions):")
            lines.append(f"    correct_species   {hier['correct_species']:.4f}")
            lines.append(f"    correct_genus     {hier['correct_genus']:.4f}")
            lines.append(f"    correct_tribe     {hier['correct_tribe']:.4f}")
            lines.append(f"    correct_subfamily {hier['correct_subfamily']:.4f}")
            if hier.get("per_band"):
                lines.append("    by OOD distance band:")
                for band, b in hier["per_band"].items():
                    band_str = str(band)
                    lines.append(f"      {band_str:14s} n={b['n']:>4}  "
                                 f"species={b['correct_species']:.3f}  "
                                 f"genus={b['correct_genus']:.3f}  "
                                 f"tribe={b['correct_tribe']:.3f}  "
                                 f"subfamily={b['correct_subfamily']:.3f}")

        # OOD performance by distance band (semantic_ood only)
        bb = sr.get("ood_by_band", {}).get("by_band", {})
        if bb:
            lines.append("\n  OOD performance by taxonomic distance band:")
            lines.append("    " + "-" * 92)
            lines.append(f"    {'band':12s}  {'n_sp':>4} {'n_ann':>5} "
                         f"{'AP@.5':>7} {'AR@100':>7}  "
                         f"{'gen':>5} {'tribe':>5} {'subf':>5}  "
                         f"{'conf(TP)':>9} {'conf(all)':>10}")
            lines.append("    " + "-" * 92)
            for band, b in bb.items():
                lines.append(
                    f"    {band:12s}  {b['n_species']:>4} {b['n_annotations']:>5} "
                    f"{b['class_agnostic_AP_50']:>7.3f} {b['class_agnostic_AR_100']:>7.3f}  "
                    f"{b['correct_genus']:>5.3f} {b['correct_tribe']:>5.3f} {b['correct_subfamily']:>5.3f}  "
                    f"{b['mean_confidence_matched']:>9.3f} {b['mean_confidence_all_preds']:>10.3f}")
            lines.append("    " + "-" * 92)
            # Inline interpretive hint
            if "near_genus" in bb and "far_tribe" in bb:
                ap_drop = bb["near_genus"]["class_agnostic_AP_50"] - bb["far_tribe"]["class_agnostic_AP_50"]
                g_drop  = bb["near_genus"]["correct_genus"]         - bb["far_tribe"]["correct_genus"]
                lines.append(f"    near_genus -> far_tribe: "
                             f"AP@.5 drops by {ap_drop:+.3f}, "
                             f"correct_genus by {g_drop:+.3f}")

        # Class imbalance tolerance section
        ci = sr.get("class_imbalance", {})
        if ci and "head_AP_50" in ci:
            lines.append(f"\n  Class imbalance tolerance ({ci['n_species_evaluated']} species, "
                         f"binned by training image count):")
            for tier in ["head", "medium", "tail"]:
                k_ap = f"{tier}_AP_50"
                if k_ap in ci:
                    lines.append(f"    {tier:6s}  AP@.5={ci[k_ap]:.4f}  AP@[.5:.95]={ci.get(f'{tier}_AP_50_95',0):.4f}  "
                                 f"n={ci.get(f'{tier}_n_species',0):>2}  "
                                 f"train_imgs=[{ci.get(f'{tier}_min_train_imgs',0)}..{ci.get(f'{tier}_max_train_imgs',0)}]")
            if "head_minus_tail_AP_50" in ci:
                lines.append(f"    head − tail   AP@.5 gap = {ci['head_minus_tail_AP_50']:+.4f}  "
                             f"(large positive = rich-class biased; ~0 = robust)")
            if "AP_50_train_count_correlation" in ci:
                lines.append(f"    Pearson r(AP@.5, train_count) = {ci['AP_50_train_count_correlation']:+.3f}  "
                             f"(>0.6 strongly tracks data quantity; ~0 = imbalance-robust)")

        # Detection / classification decomposition section
        dc = sr.get("decomposition", {})
        if dc:
            lines.append("\n  Detection vs classification decomposition:")
            if "joint_AP_50" in dc:
                lines.append(f"    joint_AP_50                    {dc['joint_AP_50']:.4f}  "
                             f"(find AND correctly classify)")
            if "detection_only_AP_50_ref" in dc:
                lines.append(f"    detection_only_AP_50_ref       {dc['detection_only_AP_50_ref']:.4f}  "
                             f"(REFERENCE ONLY -- not an upper bound; see FIX 1)")
            if "detection_recall" in dc:
                lines.append(f"    detection_recall               {dc['detection_recall']:.4f}  "
                             f"(class-agnostic; fraction of GT localized)")
            if "acc_species_given_det" in dc:
                lines.append(f"    acc_species|detection          {dc['acc_species_given_det']:.4f}  "
                             f"(of TP detections, % correct species)")
            if "species_recall" in dc:
                lines.append(f"    species_recall                 {dc['species_recall']:.4f}  "
                             f"(found AND correctly labelled / all GT)")
            if "classification_loss" in dc:
                lines.append(f"    classification_loss            {dc['classification_loss']:.4f}  "
                             f"(found-but-misclassified share of GT)")
        lines.append("")

    lines.append("=" * 78)
    lines.append("Files written:")
    for fp in results.get("output_files", []):
        lines.append(f"  {fp}")
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def run_evaluation(benchmark_dir: Path, predictions_dir: Path, splits: list,
                   iou: float = DEFAULT_IOU, quiet: bool = False) -> dict:
    """Programmatic entry point. Returns the full results dict; doesn't write files."""
    taxonomy = load_taxonomy(benchmark_dir)
    global_cat_lookup = build_global_category_lookup(benchmark_dir)
    train_image_counts = load_train_image_counts(benchmark_dir)
    train_object_counts, objects_per_image = load_train_object_counts(benchmark_dir)
    band_lookup = (taxonomy.set_index("scientificName")["distance_band_vs_trainable"].to_dict()
                   if "distance_band_vs_trainable" in taxonomy.columns else {})
    print(f"Loaded {len(global_cat_lookup)} categories across all benchmark splits.")
    if train_image_counts:
        print(f"Loaded training image counts for {len(train_image_counts)} species "
              f"(used for class-imbalance evaluation).")
    per_split = {}
    detection_records = []

    for split in splits:
        print(f"\nEvaluating split: {split}")
        coco_gt = load_benchmark_coco(benchmark_dir, split)
        raw_preds = load_predictions(predictions_dir, split)
        preds, warnings = validate_predictions(coco_gt, raw_preds, split)
        for w in warnings:
            print(f"  [warn] {w}")
        print(f"  {len(preds)} valid predictions, "
              f"{len(coco_gt.getImgIds())} GT images, "
              f"{len(coco_gt.getAnnIds())} GT annotations")

        split_results = {"warnings": warnings, "n_predictions": len(preds)}

        # 0. [FIX 5B] Match ONCE. Four downstream metrics need the identical
        #    class-agnostic greedy matching (calibration, hierarchical,
        #    decomposition, detection records) and each used to recompute it.
        n_gt_split = _n_real_gt(coco_gt)
        n_img_split = len(coco_gt.getImgIds())
        matches = match_predictions_to_gt(coco_gt, preds, iou) if preds else []

        # 0b. [FIX 5B] Prediction density. This is the disclosure that explains
        #     why class_agnostic_AP is low WITHOUT it being a bug: inference ran
        #     conf=0.001 with ultralytics' default agnostic_nms=False, so each GT
        #     beetle carries many overlapping boxes under different species. Once
        #     relabelled to a single class those become genuine false positives.
        #     Quote class_agnostic_AR_100 (duplicate-immune), not AP.
        split_results["prediction_density"] = {
            "n_predictions":   len(preds),
            "n_gt":            n_gt_split,
            "n_images":        n_img_split,
            "boxes_per_gt":    (len(preds) / n_gt_split) if n_gt_split else 0.0,
            "boxes_per_image": (len(preds) / n_img_split) if n_img_split else 0.0,
        }
        print(f"  - prediction density: {len(preds)/max(n_gt_split,1):.1f} boxes per GT beetle")

        # 1. Standard COCO mAP — meaningful on IID splits, near-zero on OOD
        print("  - standard COCO mAP suite")
        split_results["coco"] = evaluate_coco_standard(coco_gt, preds, quiet=quiet)

        # 2. Per-species AP — diagnostic, written to its own CSV
        print("  - per-species AP")
        split_results["per_species"] = evaluate_per_species(coco_gt, preds)

        # 3. Class-agnostic detection — primary metric on semantic_ood
        print("  - class-agnostic detection")
        split_results["class_agnostic"] = evaluate_class_agnostic(
            benchmark_dir, split, preds, quiet=quiet)

        # 4. Calibration
        print("  - calibration / ECE")
        split_results["calibration"] = compute_calibration(
            coco_gt, preds, iou_threshold=iou, matches=matches)

        # 5. Hierarchical / taxonomic agreement — most informative on semantic_ood
        print("  - hierarchical taxonomic agreement")
        split_results["hierarchical"] = evaluate_hierarchical(
            coco_gt, preds, taxonomy, global_cat_lookup, iou_threshold=iou,
            matches=matches)

        # 5b. OOD performance by taxonomic distance band (semantic_ood only)
        if split == "semantic_ood":
            print("  - OOD performance by taxonomic distance band")
            split_results["ood_by_band"] = evaluate_ood_by_band(
                benchmark_dir, split, preds, taxonomy, global_cat_lookup,
                iou_threshold=iou, quiet=quiet)

        # 6. Class imbalance tolerance — uses per-species AP + training counts
        print("  - class imbalance tolerance")
        split_results["class_imbalance"] = evaluate_class_imbalance(
            split_results["per_species"], train_image_counts)

        # 7. [FIX 1] Sound detection/classification decomposition (every split)
        print("  - detection / classification decomposition")
        split_results["decomposition"] = compute_decomposition(
            coco_gt, preds, taxonomy, global_cat_lookup, split, split_results,
            iou_threshold=iou, matches=matches)

        # 7b. [FIX 3+4] per-detection records + matched-TP confidences for novelty
        if preds:
            recs, tp_confs, tp_by_band = build_detection_records(
                matches, taxonomy, global_cat_lookup, split, band_lookup)
            detection_records.extend(recs)
            split_results["_tp_confs"] = tp_confs
            if split == "semantic_ood":
                split_results["_tp_confs_by_band"] = tp_by_band

        per_split[split] = split_results

    summary = compute_summary(per_split)
    return {"per_split": per_split, "summary": summary,
            "config": {"iou_threshold": iou, "splits": splits},
            "_train_images": train_image_counts, "_train_objects": train_object_counts,
            "_objects_per_image": objects_per_image, "_detection_records": detection_records}


def _strip_private(results: dict) -> dict:
    """Drop bulky/private underscore keys before JSON serialization."""
    for k in list(results.keys()):
        if k.startswith("_"):
            results.pop(k, None)
    for sr in results.get("per_split", {}).values():
        for k in list(sr.keys()):
            if k.startswith("_"):
                sr.pop(k, None)
    return results


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = run_evaluation(args.benchmark_dir, args.predictions_dir,
                             splits=args.splits, iou=args.iou, quiet=args.quiet)

    # Persist results to disk. CSVs that need the private keys (train counts,
    # per-detection records) are written FIRST; then those keys are stripped
    # before the JSON dump so summary.json stays compact.
    output_files = []
    output_files.append(str(write_per_species_csv(results, args.output_dir, args.model_name)))
    output_files.append(str(write_detections_csv(results, args.output_dir, args.model_name)))
    _strip_private(results)
    output_files.append(str(write_summary_json(results, args.output_dir, args.model_name)))
    output_files.append(str(write_summary_csv(results, args.output_dir, args.model_name)))
    results["output_files"] = output_files

    report = render_report(results, args.model_name)
    report_path = args.output_dir / f"{args.model_name}_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    output_files.append(str(report_path))
    print()
    print(report)


if __name__ == "__main__":
    main()