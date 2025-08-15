from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_performance(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calculates a suite of classification metrics.

    Args:
        true_labels (np.ndarray): An array of ground-truth labels.
        predicted_labels (np.ndarray): An array of labels predicted by the model.
        target_names (List[str], optional): Display names for the labels.

    Returns:
        Dict[str, Any]: A dictionary containing key performance metrics.
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    # The type stubs for sklearn are likely incorrect, expecting a str.
    # "warn" is a valid option that achieves the same goal as 0.
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="macro", zero_division="warn"
    )
    cm = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=target_names,
        output_dict=True,
        zero_division="warn",
    )

    performance_metrics = {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    return performance_metrics


def object_detection_performance(
    gt_boxes: np.ndarray,
    gt_labels: List[int],
    gt_image_ids: List[Any],
    pred_boxes: np.ndarray,
    pred_labels: List[int],
    pred_scores: List[float],
    pred_image_ids: List[Any],
    iou_thresholds: Union[float, List[float]] = 0.5,
) -> Dict[str, Any]:
    """
    Calculates mean Average Precision (mAP) over one or more IoU thresholds.

    This version accepts flattened lists for detections, where the list indices
    must correspond (e.g., gt_boxes[i], gt_labels[i], and gt_image_ids[i] all
    describe the same ground-truth box).

    Args:
        gt_boxes (np.ndarray): NumPy array of ground-truth bounding boxes with shape (N, 4).
                                Each row is [x1, y1, x2, y2].
        gt_labels (List[Any]): List of ground-truth labels [Any,...].
        gt_image_ids (List[Any]): List of image IDs for each ground-truth box.
        pred_boxes (np.ndarray): NumPy array of predicted bounding boxes with shape (M, 4).
                                 Each row is [x1, y1, x2, y2].
        pred_labels (List[Any]): List of predicted labels [Any,...].
        pred_scores (List[float]): List of prediction confidence scores [float,...].
        pred_image_ids (List[Any]): List of image IDs for each predicted box.
        iou_thresholds (Union[float, List[float]]): A single IoU threshold or a list of thresholds.

    Returns:
        Dict[str, Any]: A dictionary containing the final averaged mAP, per-class APs for the
                        last IoU threshold, and mAP scores for each individual threshold.
    """

    def _calculate_iou(boxA, boxB):
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Compute the intersection over union
        denominator = float(boxAArea + boxBArea - interArea)
        iou = interArea / denominator if denominator > 0 else 0
        return iou

    if isinstance(iou_thresholds, (int, float)):
        iou_thresholds = [iou_thresholds]

    # --- 1. Data Restructuring ---
    # Convert flat lists into a per-image dictionary format for easier processing.
    gt_by_image = defaultdict(lambda: {"boxes": [], "labels": []})
    for box, label, image_id in zip(gt_boxes, gt_labels, gt_image_ids):
        gt_by_image[image_id]["boxes"].append(box)
        gt_by_image[image_id]["labels"].append(label)

    # Pre-organize all predictions and GT counts by class for efficiency
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

    # --- 2. Main Calculation Loop ---
    for iou_threshold in iou_thresholds:
        aps = {}
        for class_id in all_classes:
            class_preds = preds_by_class[class_id]
            num_gt_boxes = gt_counts_by_class[class_id]

            # Handle edge cases
            if num_gt_boxes == 0:
                aps[class_id] = 1.0 if not class_preds else 0.0
                continue
            if not class_preds:
                aps[class_id] = 0.0
                continue

            # Sort predictions by score
            class_preds.sort(key=lambda x: x["score"], reverse=True)

            # Match predictions to ground truths
            tp = np.zeros(len(class_preds))
            fp = np.zeros(len(class_preds))
            gt_matched = {img_id: np.zeros(len(data["boxes"])) for img_id, data in gt_by_image.items()}

            for i, pred in enumerate(class_preds):
                image_id = pred["image_id"]
                gt = gt_by_image.get(image_id)

                if not gt:
                    fp[i] = 1
                    continue

                gt_info_for_class = [(j, box) for j, box in enumerate(gt["boxes"]) if gt["labels"][j] == class_id]

                if not gt_info_for_class:
                    fp[i] = 1
                    continue

                best_iou = -1.0
                best_gt_original_idx = -1
                for original_idx, gt_box in gt_info_for_class:
                    iou = _calculate_iou(pred["box"], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_original_idx = original_idx

                if best_iou >= iou_threshold:
                    if not gt_matched[image_id][best_gt_original_idx]:
                        tp[i] = 1
                        gt_matched[image_id][best_gt_original_idx] = 1
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1

            # Calculate precision-recall curve
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls = tp_cumsum / num_gt_boxes
            precisions = tp_cumsum / (
                tp_cumsum + fp_cumsum + np.finfo(float).eps
            )  # Add epsilon to avoid division by zero

            # Use Area Under Curve (AUC) method for AP calculation (more modern than 11-point)
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            # Replace 11-point interpolation with modern AUC calculation
            for j in range(len(precisions) - 2, -1, -1):
                precisions[j] = max(precisions[j], precisions[j + 1])

            recall_indices = np.where(recalls[1:] != recalls[:-1])[0]
            ap = np.sum((recalls[recall_indices + 1] - recalls[recall_indices]) * precisions[recall_indices + 1])
            aps[class_id] = ap

        per_threshold_scores[f"mAP@{iou_threshold:.2f}"] = np.mean(list(aps.values())) if aps else 0.0
        aps_last_iou = aps

    final_map_averaged = np.mean(list(per_threshold_scores.values())) if per_threshold_scores else 0.0

    # --- 3. Return Results ---
    return {
        "mAP_averaged": final_map_averaged,
        "per_class_AP_at_last_iou": aps_last_iou,
        "per_threshold_scores": per_threshold_scores,
    }
