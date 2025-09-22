# src/ibbi/evaluate/object_detection.py

from collections import defaultdict
from typing import Any, Union

import numpy as np
import pandas as pd


def _calculate_iou(boxA, boxB):
    """Calculates Intersection over Union for two bounding boxes.

    This function takes two bounding boxes in the format [x1, y1, x2, y2] and computes
    the Intersection over Union (IoU) score, which is a measure of the extent of their overlap.

    Args:
        boxA (list or np.ndarray): The first bounding box, specified as [x1, y1, x2, y2].
        boxB (list or np.ndarray): The second bounding box, specified as [x1, y1, x2, y2].

    Returns:
        float: The IoU score, a value between 0 and 1.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    denominator = float(boxAArea + boxBArea - interArea)
    iou = interArea / denominator if denominator > 0 else 0
    return iou


def object_detection_performance(
    gt_boxes: np.ndarray,
    gt_labels: list[int],
    gt_image_ids: list[Any],
    pred_boxes: np.ndarray,
    pred_labels: list[int],
    pred_scores: list[float],
    pred_image_ids: list[Any],
    iou_thresholds: Union[float, list[float], np.ndarray] | None = None,
    confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculates a comprehensive suite of object detection metrics.

    This function evaluates the performance of an object detection model by computing the
    mean Average Precision (mAP) over a range of Intersection over Union (IoU) thresholds.
    It provides a detailed breakdown of performance, including per-class AP scores.

    Args:
        gt_boxes (np.ndarray): A numpy array of ground truth bounding boxes, with each box in [x1, y1, x2, y2] format.
        gt_labels (list[int]): A list of integer labels for each ground truth box.
        gt_image_ids (list[Any]): A list of image identifiers for each ground truth box.
        pred_boxes (np.ndarray): A numpy array of predicted bounding boxes.
        pred_labels (list[int]): A list of predicted integer labels for each box.
        pred_scores (list[float]): A list of confidence scores for each predicted box.
        pred_image_ids (list[Any]): A list of image identifiers for each predicted box.
        iou_thresholds (Union[float, list[float], np.ndarray], optional): The IoU threshold(s) for matching predictions to
                                                                         ground truth. Can be a single float, a list of floats,
                                                                         or a numpy array. Defaults to np.arange(0.5, 1.0, 0.05).
        confidence_threshold (float, optional): The confidence score threshold below which predictions are ignored.
                                                Defaults to 0.5.

    Returns:
        dict[str, Any]: A dictionary containing:
                        - "mAP": The mean Average Precision averaged over all IoU thresholds.
                        - "per_class_AP_at_last_iou": A dictionary mapping class IDs to their AP score at the last IoU threshold.
                        - "per_threshold_scores": A dictionary mapping each IoU threshold to its corresponding mAP score.
                        - "sample_results": A pandas DataFrame with detailed information on each ground truth and predicted box.
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    if isinstance(iou_thresholds, (int, float)):
        iou_thresholds = [iou_thresholds]

    all_classes = sorted(set(gt_labels) | set(pred_labels))

    # --- Data Restructuring ---
    gt_by_image = defaultdict(lambda: {"boxes": [], "labels": [], "used": []})
    for box, label, image_id in zip(gt_boxes, gt_labels, gt_image_ids):
        gt_by_image[image_id]["boxes"].append(box)
        gt_by_image[image_id]["labels"].append(label)
        gt_by_image[image_id]["used"].append(False)

    sample_results = []
    for image_id, data in gt_by_image.items():
        for i in range(len(data["boxes"])):
            sample_results.append(
                {
                    "image_id": image_id,
                    "type": "ground_truth",
                    "box": data["boxes"][i],
                    "label": data["labels"][i],
                    "score": None,
                }
            )

    preds_by_class = defaultdict(list)
    gt_counts_by_class = defaultdict(int)

    for gt_data in gt_by_image.values():
        for label in gt_data["labels"]:
            gt_counts_by_class[label] += 1

    for box, label, score, image_id in zip(pred_boxes, pred_labels, pred_scores, pred_image_ids):
        if score >= confidence_threshold:
            preds_by_class[label].append({"box": box, "score": score, "image_id": image_id})
            sample_results.append(
                {
                    "image_id": image_id,
                    "type": "prediction",
                    "box": box,
                    "label": label,
                    "score": score,
                }
            )

    per_threshold_scores = {}
    aps_last_iou = {}

    # --- Main Calculation Loop ---
    for iou_threshold in iou_thresholds:
        aps = {}
        for class_id in all_classes:
            class_preds = sorted(preds_by_class[class_id], key=lambda x: x["score"], reverse=True)
            num_gt_boxes = gt_counts_by_class[class_id]

            if num_gt_boxes == 0:
                aps[class_id] = 1.0 if not class_preds else 0.0
                continue
            if not class_preds:
                aps[class_id] = 0.0
                continue

            tp = np.zeros(len(class_preds))
            fp = np.zeros(len(class_preds))

            for i, pred in enumerate(class_preds):
                gt_data = gt_by_image[pred["image_id"]]
                best_iou = -1.0
                best_gt_idx = -1

                for j, gt_box in enumerate(gt_data["boxes"]):
                    if gt_data["labels"][j] == class_id:
                        iou = _calculate_iou(pred["box"], gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j

                if best_iou >= iou_threshold:
                    if not gt_data["used"][best_gt_idx]:
                        tp[i] = 1
                        gt_data["used"][best_gt_idx] = True
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1

            # Reset 'used' flags for the next IoU threshold
            for data in gt_by_image.values():
                data["used"] = [False] * len(data["boxes"])

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls = tp_cumsum / (num_gt_boxes + np.finfo(float).eps)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + np.finfo(float).eps)

            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            for j in range(len(precisions) - 2, -1, -1):
                precisions[j] = max(precisions[j], precisions[j + 1])

            recall_indices = np.where(recalls[1:] != recalls[:-1])[0]
            ap = np.sum((recalls[recall_indices + 1] - recalls[recall_indices]) * precisions[recall_indices + 1])
            aps[class_id] = ap

        per_threshold_scores[f"mAP@{iou_threshold:.2f}"] = np.mean(list(aps.values())) if aps else 0.0
        aps_last_iou = aps

    final_map_averaged = np.mean(list(per_threshold_scores.values())) if per_threshold_scores else 0.0

    return {
        "mAP": final_map_averaged,
        "per_class_AP_at_last_iou": aps_last_iou,
        "per_threshold_scores": per_threshold_scores,
        "sample_results": pd.DataFrame(sample_results),
    }
