# src/ibbi/evaluate/object_detection.py

from collections import defaultdict
from typing import Any, Union

import numpy as np


def _calculate_iou(boxA, boxB):
    """Calculates Intersection over Union for two bounding boxes [x1, y1, x2, y2]."""
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
    iou_thresholds: Union[float, list[float]] = 0.5,
) -> dict[str, Any]:
    """
    Calculates mean Average Precision (mAP) over one or more IoU thresholds.
    """
    if isinstance(iou_thresholds, (int, float)):
        iou_thresholds = [iou_thresholds]

    # --- Data Restructuring ---
    gt_by_image = defaultdict(lambda: {"boxes": [], "labels": []})
    for box, label, image_id in zip(gt_boxes, gt_labels, gt_image_ids):
        gt_by_image[image_id]["boxes"].append(box)
        gt_by_image[image_id]["labels"].append(label)

    preds_by_class = defaultdict(list)
    gt_counts_by_class = defaultdict(int)

    for gt_data in gt_by_image.values():
        for label in gt_data["labels"]:
            gt_counts_by_class[label] += 1

    for box, label, score, image_id in zip(pred_boxes, pred_labels, pred_scores, pred_image_ids):
        preds_by_class[label].append({"box": box, "score": score, "image_id": image_id})

    all_classes = sorted(set(gt_labels) | set(pred_labels))
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
            gt_matched = {img_id: np.zeros(len(data["boxes"])) for img_id, data in gt_by_image.items()}

            for i, pred in enumerate(class_preds):
                gt_info_for_class = [
                    (j, box)
                    for j, box in enumerate(gt_by_image[pred["image_id"]]["boxes"])
                    if gt_by_image[pred["image_id"]]["labels"][j] == class_id
                ]

                best_iou = -1.0
                best_gt_original_idx = -1

                for original_idx, gt_box in gt_info_for_class:
                    iou = _calculate_iou(pred["box"], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_original_idx = original_idx

                if best_iou >= iou_threshold and best_gt_original_idx != -1:
                    if not gt_matched[pred["image_id"]][best_gt_original_idx]:
                        tp[i] = 1
                        gt_matched[pred["image_id"]][best_gt_original_idx] = 1
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls = tp_cumsum / (num_gt_boxes + np.finfo(float).eps)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + np.finfo(float).eps)

            # --- Start of corrected section ---
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            for j in range(len(precisions) - 2, -1, -1):
                precisions[j] = max(precisions[j], precisions[j + 1])

            recall_indices = np.where(recalls[1:] != recalls[:-1])[0]
            ap = np.sum((recalls[recall_indices + 1] - recalls[recall_indices]) * precisions[recall_indices + 1])
            # --- End of corrected section ---
            aps[class_id] = ap

        per_threshold_scores[f"mAP@{iou_threshold:.2f}"] = np.mean(list(aps.values())) if aps else 0.0
        aps_last_iou = aps

    final_map_averaged = np.mean(list(per_threshold_scores.values())) if per_threshold_scores else 0.0

    return {
        "mAP_averaged": final_map_averaged,
        "per_class_AP_at_last_iou": aps_last_iou,
        "per_threshold_scores": per_threshold_scores,
    }
