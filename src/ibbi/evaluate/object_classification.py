"""
Provides a unified function for evaluating object detection and classification performance.

This module combines traditional object detection metrics like mean Average Precision (mAP)
with a comprehensive suite of classification metrics. It is designed to provide a holistic
view of a model's ability to both correctly localize and classify objects within images.
The evaluation is performed on a per-object basis, matching predicted detections to
ground truth objects before comparing their class labels.
"""

from collections import defaultdict
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


def _calculate_iou(boxA, boxB):
    """Calculates Intersection over Union for two bounding boxes.

    This function takes two bounding boxes in the format [x1, y1, x2, y2] and computes
    the Intersection over Union (IoU) score, which is a measure of the extent of their overlap.
    A higher IoU score indicates a greater degree of overlap between the two boxes.

    Args:
        boxA (list or np.ndarray): The first bounding box, specified as [x1, y1, x2, y2].
        boxB (list or np.ndarray): The second bounding box, specified as [x1, y1, x2, y2].

    Returns:
        float: The IoU score, a value between 0.0 and 1.0. An IoU of 1.0 indicates a perfect
               overlap, while an IoU of 0.0 indicates no overlap.
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


def object_classification_performance(
    gt_boxes: np.ndarray,
    gt_labels: list[int],
    gt_image_ids: list[Any],
    pred_results_with_probs: list[dict],
    gt_label_names: list[str],
    iou_thresholds: Union[float, list[float], np.ndarray] | None = None,
    confidence_threshold: float = 0.1,
    average: str = "macro",
    zero_division: Union[str, int, float] = np.nan,
    model_classes: Optional[list[str]] = None,
    idx_to_name: Optional[dict[int, str]] = None,
) -> dict[str, Any]:
    """Calculates a comprehensive suite of object detection and classification metrics.

    This function performs a holistic evaluation, combining mAP and standard classification
    metrics. It also generates a detailed object-level table mapping ground truth objects
    to their best-matching predictions and full per-class confidence scores.

    Args:
        gt_boxes (np.ndarray): A NumPy array of ground truth bounding boxes, in [x1, y1, x2, y2] format.
        gt_labels (list[int]): A list of integer labels corresponding to each ground truth box.
        gt_image_ids (list[Any]): A list of image identifiers for each ground truth box.
        pred_results_with_probs (list[dict]): A list of prediction results (one per image), where each
                                              result dictionary must contain 'full_results' with per-class probabilities.
        gt_label_names (list[str]): A list of the original ground truth label names from the dataset.
        iou_thresholds (Union[float, list[float], np.ndarray], optional): The IoU threshold(s)
            for mAP and classification metric calculation. Defaults to np.arange(0.5, 1.0, 0.05).
        confidence_threshold (float, optional): The confidence score threshold below which
            predictions are ignored for mAP/classification and matching. Defaults to 0.5.
        average (str, optional): The averaging method for multiclass classification metrics. Defaults to "macro".
        zero_division (Union[str, int, float], optional): Sets the value to return when there is a zero
            division in classification metric calculations. Defaults to "np.nan".
        model_classes (list[str], optional): A list of all class names the model was trained/set for.
        idx_to_name (dict, optional): A mapping from integer class IDs to class names.

    Returns:
        dict[str, Any]: A dictionary containing a comprehensive set of performance metrics:
                        - "mAP": The mean Average Precision averaged over all IoU thresholds.
                        - "per_class_AP_at_last_iou": A dict mapping class names to their AP score at the last IoU threshold.
                        - "per_threshold_scores": A dict mapping each IoU threshold to its mAP score.
                        - "per_iou_classification_metrics": A dictionary of classification metrics per IoU threshold.
                        - "object_level_performance": A pandas DataFrame with detailed performance per ground truth object.
    """
    if model_classes is None or idx_to_name is None:
        raise ValueError("model_classes and idx_to_name must be provided.")

    name_to_idx = {v: k for k, v in idx_to_name.items()}

    # --- 1. Flatten Predictions for mAP Calculation and Store Detailed Preds for Matching ---
    # These flat lists are used for mAP and classification metrics
    pred_boxes, pred_labels, pred_scores, pred_image_ids = [], [], [], []
    # This nested structure stores rich prediction data for the final table
    image_to_preds = defaultdict(list)

    for image_id, res in enumerate(pred_results_with_probs):
        if not res or not res.get("boxes"):
            continue

        # Ensure we can iterate over full_results safely, even if a basic predict result is returned
        full_results = res.get("full_results", [None] * len(res["boxes"]))

        for box, label, score, full_result in zip(res["boxes"], res["labels"], res["scores"], full_results):
            # Only consider predictions above the confidence threshold
            if score >= confidence_threshold:
                # For mAP/Classification (needs flat lists of integer IDs)
                if label in name_to_idx:
                    pred_boxes.append(np.array(box).flatten())
                    pred_labels.append(name_to_idx[label])
                    pred_scores.append(score)
                    pred_image_ids.append(image_id)

                # For the object_level_performance table
                image_to_preds[image_id].append({"box": box, "score": score, "label_name": label, "full_results": full_result})

    # --- 2. Setup for mAP/Classification Metrics ---
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    if isinstance(iou_thresholds, (int, float)):
        iou_thresholds = [iou_thresholds]

    all_gt_labels = list(set(gt_labels))
    all_pred_labels = list(set(pred_labels))
    class_lst_us = list(set(all_gt_labels) | set(all_pred_labels))
    class_lst = sorted(class_lst_us)
    all_classes_us = list(set(c for c in class_lst if c != -1))  # Exclude -1 from all_classes
    all_classes = sorted(all_classes_us)

    # Group GT by image for faster lookups and 'used' tracking
    gt_by_image = defaultdict(lambda: {"boxes": [], "labels": [], "used": []})
    for box, label, image_id in zip(gt_boxes, gt_labels, gt_image_ids):
        gt_by_image[image_id]["boxes"].append(box)
        gt_by_image[image_id]["labels"].append(label)
        gt_by_image[image_id]["used"].append(False)

    preds_by_class = defaultdict(list)
    gt_counts_by_class = defaultdict(int)

    for gt_data in gt_by_image.values():
        for label in gt_data["labels"]:
            gt_counts_by_class[label] += 1

    for box, label, score, image_id in zip(pred_boxes, pred_labels, pred_scores, pred_image_ids):
        preds_by_class[label].append({"box": box, "score": score, "image_id": image_id})

    per_threshold_scores = {}
    per_iou_classification_metrics = {}
    aps_last_iou = {}

    # --- 3. mAP/Classification Metrics Calculation Loop ---
    all_preds_by_image_flat = defaultdict(list)
    for box, label, score, image_id in zip(pred_boxes, pred_labels, pred_scores, pred_image_ids):
        all_preds_by_image_flat[image_id].append(
            {
                "box": box,
                "label": label,  # Integer ID
                "score": score,
            }
        )

    for iou_threshold in iou_thresholds:
        aps = {}
        true_class_labels_for_iou = []
        pred_class_labels_for_iou = []

        # Reset 'used' flags for mAP calculation
        for data in gt_by_image.values():
            data["used"] = [False] * len(data["boxes"])

        # a. mAP Calculation
        for class_id in all_classes:
            class_preds = sorted(preds_by_class[class_id], key=lambda x: x["score"], reverse=True)
            num_gt_boxes = gt_counts_by_class.get(class_id, 0)

            if num_gt_boxes == 0:
                aps[class_id] = 1.0 if not class_preds else 0.0
                continue
            if not class_preds:
                aps[class_id] = 0.0
                continue

            tp, fp = np.zeros(len(class_preds)), np.zeros(len(class_preds))
            for i, pred in enumerate(class_preds):
                gt_data = gt_by_image.get(pred["image_id"], {"boxes": [], "labels": [], "used": []})
                best_iou = -1.0
                best_gt_idx = -1

                for j, gt_box in enumerate(gt_data["boxes"]):
                    if gt_data["labels"][j] == class_id:
                        iou = _calculate_iou(pred["box"], gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j

                if best_iou >= iou_threshold and best_gt_idx != -1:
                    if not gt_data["used"][best_gt_idx]:
                        tp[i] = 1
                        gt_data["used"][best_gt_idx] = True
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1

            tp_cumsum, fp_cumsum = np.cumsum(tp), np.cumsum(fp)
            recalls = tp_cumsum / (num_gt_boxes + np.finfo(float).eps)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + np.finfo(float).eps)
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))
            for j in range(len(precisions) - 2, -1, -1):
                precisions[j] = max(precisions[j], precisions[j + 1])
            recall_indices = np.where(recalls[1:] != recalls[:-1])[0]
            aps[class_id] = np.sum((recalls[recall_indices + 1] - recalls[recall_indices]) * precisions[recall_indices + 1])

        per_threshold_scores[f"mAP@{iou_threshold:.2f}"] = np.mean(list(aps.values())) if aps else 0.0
        aps_last_iou = aps

        # b. Classification Metrics Calculation
        for gt_image_id, gt_data in gt_by_image.items():
            for gt_box, gt_label in zip(gt_data["boxes"], gt_data["labels"]):
                true_class_labels_for_iou.append(gt_label)
                best_iou = -1.0
                best_pred_label = -1  # -1 is the "no-match" label

                for pred in all_preds_by_image_flat[gt_image_id]:
                    iou = _calculate_iou(gt_box, pred["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_label = pred["label"]  # Integer ID

                if best_iou >= iou_threshold:
                    pred_class_labels_for_iou.append(best_pred_label)
                else:
                    pred_class_labels_for_iou.append(-1)

        accuracy = accuracy_score(true_class_labels_for_iou, pred_class_labels_for_iou)
        balanced_accuracy = balanced_accuracy_score(true_class_labels_for_iou, pred_class_labels_for_iou)

        has_multiple_effective_classes = len(np.unique(true_class_labels_for_iou)) > 1 and len(np.unique(pred_class_labels_for_iou)) > 1
        kappa = cohen_kappa_score(true_class_labels_for_iou, pred_class_labels_for_iou) if has_multiple_effective_classes else np.nan
        mcc = matthews_corrcoef(true_class_labels_for_iou, pred_class_labels_for_iou) if has_multiple_effective_classes else np.nan

        extended_classes_us = list(set(all_classes) | {-1})
        extended_classes = sorted(extended_classes_us)
        extended_target_names = [idx_to_name.get(c, "No_Match") for c in extended_classes]

        class_subset_for_metrics_us = list(set(true_class_labels_for_iou) - {-1})
        class_subset_for_metrics = sorted(class_subset_for_metrics_us)
        target_subset_for_metrics = [idx_to_name.get(c, c) for c in class_subset_for_metrics]

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_class_labels_for_iou,
            pred_class_labels_for_iou,
            average=average,
            zero_division=zero_division,  # type: ignore
            labels=class_subset_for_metrics,
        )

        cm = confusion_matrix(true_class_labels_for_iou, pred_class_labels_for_iou, labels=extended_classes)
        cm_df = pd.DataFrame(cm, index=pd.Index(extended_target_names), columns=pd.Index(extended_target_names))
        cm_df.index.name = "True Label"
        cm_df.columns.name = "Predicted Label"

        report = classification_report(
            true_class_labels_for_iou,
            pred_class_labels_for_iou,
            labels=class_subset_for_metrics,
            target_names=target_subset_for_metrics,
            output_dict=True,
            zero_division=zero_division,  # type: ignore
        )

        per_iou_classification_metrics[f"iou_{iou_threshold:.2f}"] = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            f"{average}_precision": precision,
            f"{average}_recall": recall,
            f"{average}_f1_score": f1,
            "cohen_kappa": kappa,
            "matthews_corrcoef": mcc,
            "confusion_matrix_df": cm_df,
            "classification_report": report,
        }

    final_map_averaged = np.mean(list(per_threshold_scores.values())) if per_threshold_scores else 0.0

    # --- 4. Object Level Performance Table Generation ---
    object_performance_data = []
    gt_object_id_counter = defaultdict(int)

    for i in range(len(gt_boxes)):
        image_id = gt_image_ids[i]
        gt_box = gt_boxes[i]
        gt_label_name = gt_label_names[i]
        object_local_id = gt_object_id_counter[image_id]
        gt_object_id_counter[image_id] += 1

        current_preds = image_to_preds[image_id]
        best_iou = 0.0
        best_pred_box = None
        best_pred_full_results = None

        for pred in current_preds:
            iou = _calculate_iou(gt_box, pred["box"])
            if iou > best_iou:
                best_iou = iou
                best_pred_box = pred["box"]
                best_pred_full_results = pred["full_results"]

        row = {
            "image_id": image_id,
            "object_id": object_local_id,
            "gt_bbox": gt_box,
            "gt_label": gt_label_name,
        }

        if best_iou >= confidence_threshold:
            row["pred_bbox"] = best_pred_box
            row["iou"] = best_iou
        else:
            row["pred_bbox"] = [np.nan] * 4
            row["iou"] = 0.0

        class_confidences = dict.fromkeys(model_classes, 0.0)
        if best_pred_full_results and best_pred_full_results.get("class_probabilities"):
            for class_name_in_model_classes, prob in zip(model_classes, best_pred_full_results["class_probabilities"]):
                class_confidences[class_name_in_model_classes] = prob

        row.update({f"conf_{name if name else 'background'}": conf for name, conf in class_confidences.items()})
        object_performance_data.append(row)

    object_level_performance_df = pd.DataFrame(object_performance_data)

    return {
        "mAP": final_map_averaged,
        "per_class_AP_at_last_iou": {idx_to_name.get(k, k): v for k, v in aps_last_iou.items()},
        "per_threshold_scores": per_threshold_scores,
        "per_iou_classification_metrics": per_iou_classification_metrics,
        "object_level_performance": object_level_performance_df,
    }
