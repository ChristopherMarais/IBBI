# src/ibbi/evaluate/object_classification.py
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
    pred_boxes: np.ndarray,
    pred_labels: list[int],
    pred_scores: list[float],
    pred_image_ids: list[Any],
    iou_thresholds: Union[float, list[float], np.ndarray] | None = None,
    confidence_threshold: float = 0.5,
    average: str = "macro",
    zero_division: Union[str, int, float] = np.nan,
) -> dict[str, Any]:
    """Calculates a comprehensive suite of object detection and classification metrics.

    This function provides a holistic evaluation of an object detection model's performance.
    It computes the mean Average Precision (mAP) over a range of Intersection over Union (IoU)
    thresholds to assess localization accuracy. Simultaneously, it calculates a full suite of
    classification metrics (accuracy, precision, recall, F1-score, etc.) for each IoU threshold by
    matching predicted objects to ground truth objects.

    This unified approach allows for a detailed breakdown of a model's ability to both find
    and correctly identify objects at various levels of localization stringency.

    Args:
        gt_boxes (np.ndarray): A NumPy array of ground truth bounding boxes, with each box
                               in [x1, y1, x2, y2] format. Shape: (N, 4).
        gt_labels (list[int]): A list of integer labels corresponding to each ground truth box. Length: N.
        gt_image_ids (list[Any]): A list of image identifiers for each ground truth box, used to
                                  group boxes by image. Length: N.
        pred_boxes (np.ndarray): A NumPy array of predicted bounding boxes in [x1, y1, x2, y2] format. Shape: (M, 4).
        pred_labels (list[int]): A list of predicted integer labels for each predicted box. Length: M.
        pred_scores (list[float]): A list of confidence scores for each predicted box. Length: M.
        pred_image_ids (list[Any]): A list of image identifiers for each predicted box. Length: M.
        iou_thresholds (Union[float, list[float], np.ndarray], optional): The IoU threshold(s)
            for mAP and classification metric calculation. Can be a single float or a list/array of floats.
            Defaults to np.arange(0.5, 1.0, 0.05).
        confidence_threshold (float, optional): The confidence score threshold below which
            predictions are ignored. Defaults to 0.5.
        average (str, optional): The averaging method for multiclass classification metrics
            (e.g., 'micro', 'macro', 'weighted'). Defaults to "macro".
        zero_division (Union[str, int], optional): Sets the value to return when there is a zero
            division in classification metric calculations (e.g., "warn" for a warning output, 0, or np.nan). Defaults to "np.nan".

    Returns:
        dict[str, Any]: A dictionary containing a comprehensive set of performance metrics:
                        - "mAP": The mean Average Precision averaged over all IoU thresholds.
                        - "per_class_AP_at_last_iou": A dict mapping class IDs to their AP score at the last IoU threshold.
                        - "per_threshold_scores": A dict mapping each IoU threshold to its mAP score.
                        - "per_iou_classification_metrics": A dictionary where keys are IoU thresholds (e.g., "iou_0.50")
                          and values are dictionaries containing a full suite of classification metrics calculated
                          at that specific IoU threshold. This includes accuracy, balanced accuracy, precision,
                          recall, F1-score, Cohen's Kappa, Matthews Correlation Coefficient, a confusion matrix DataFrame,
                          and a classification report.
                        - "sample_results": A pandas DataFrame with detailed information on each
                                            ground truth and predicted box for error analysis.
                        - "object_table": A pandas DataFrame with detailed information on each matched object.
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    if isinstance(iou_thresholds, (int, float)):
        iou_thresholds = [iou_thresholds]
    class_lst = list(set(gt_labels) | set(pred_labels))
    all_classes = sorted(class_lst)
    target_names = [str(c) for c in all_classes]

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
    per_iou_classification_metrics = {}
    aps_last_iou = {}
    object_table = []

    # --- mAP Calculation Loop ---
    for iou_threshold in iou_thresholds:
        aps = {}
        true_class_labels_for_iou = []
        pred_class_labels_for_iou = []

        # Reset 'used' flags for each IoU threshold
        for data in gt_by_image.values():
            data["used"] = [False] * len(data["boxes"])

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
                best_gt_box = None

                for j, gt_box in enumerate(gt_data["boxes"]):
                    if gt_data["labels"][j] == class_id:
                        iou = _calculate_iou(pred["box"], gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j
                            best_gt_box = gt_box

                if best_iou >= iou_threshold:
                    if not gt_data["used"][best_gt_idx]:
                        tp[i] = 1
                        gt_data["used"][best_gt_idx] = True
                        object_table.append(
                            {
                                "image_id": pred["image_id"],
                                "gt_bbox": best_gt_box,
                                "pred_bbox": pred["box"],
                                "iou": best_iou,
                                "class_id": class_id,
                                "score": pred["score"],
                            }
                        )
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1

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

        # Classification metrics for this IoU
        for gt_image_id, gt_data in gt_by_image.items():
            for gt_box, gt_label in zip(gt_data["boxes"], gt_data["labels"]):
                true_class_labels_for_iou.append(gt_label)

                best_iou = -1
                best_pred_label = -1

                for pred_label, preds in preds_by_class.items():
                    for pred in preds:
                        if pred["image_id"] == gt_image_id:
                            iou = _calculate_iou(gt_box, pred["box"])
                            if iou > best_iou:
                                best_iou = iou
                                best_pred_label = pred_label

                if best_iou >= iou_threshold:
                    pred_class_labels_for_iou.append(best_pred_label)
                else:
                    pred_class_labels_for_iou.append(-1)

        accuracy = accuracy_score(true_class_labels_for_iou, pred_class_labels_for_iou)
        balanced_accuracy = balanced_accuracy_score(true_class_labels_for_iou, pred_class_labels_for_iou)
        kappa = cohen_kappa_score(true_class_labels_for_iou, pred_class_labels_for_iou)
        mcc = matthews_corrcoef(true_class_labels_for_iou, pred_class_labels_for_iou)

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_class_labels_for_iou,
            pred_class_labels_for_iou,
            average=average,
            zero_division=zero_division,  # type: ignore
            labels=all_classes,
        )

        cm = confusion_matrix(true_class_labels_for_iou, pred_class_labels_for_iou, labels=all_classes)
        cm_df = pd.DataFrame(cm, index=pd.Index(target_names), columns=pd.Index(target_names))
        cm_df.index.name = "True Label"
        cm_df.columns.name = "Predicted Label"

        report = classification_report(
            true_class_labels_for_iou,
            pred_class_labels_for_iou,
            labels=all_classes,
            target_names=target_names,
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
    object_df = pd.DataFrame(object_table)

    return {
        "mAP": final_map_averaged,
        "per_class_AP_at_last_iou": aps_last_iou,
        "per_threshold_scores": per_threshold_scores,
        "per_iou_classification_metrics": per_iou_classification_metrics,
        "sample_results": pd.DataFrame(sample_results),
        "object_table": object_df,
    }


def out_of_distribution_detection(
    id_results: list[dict],
    ood_results: list[dict],
    id_dataset: list[dict],
    ood_dataset: list[dict],
    class_names: list[str],
    thresholds: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Performs a comprehensive out-of-distribution (OOD) analysis.

    This function compares the model's behavior on in-distribution (ID) versus
    out-of-distribution (OOD) data. It provides detailed statistics on confidence
    scores, allowing for the determination of an optimal threshold for separating
    known from unknown classes.

    Args:
        id_results (list[dict]): A list of prediction results from the model on the ID dataset.
        ood_results (list[dict]): A list of prediction results from the model on the OOD dataset.
        id_dataset (list[dict]): The in-distribution dataset.
        ood_dataset (list[dict]): The out-of-distribution dataset.
        class_names (list[str]): A list of the model's class names.
        thresholds (np.ndarray, optional): An array of confidence thresholds to evaluate.
                                           Defaults to np.arange(0.05, 1.0, 0.05).

    Returns:
        dict[str, Any]: A dictionary containing a comprehensive OOD analysis, including:
                        - "confidence_score_analysis": A DataFrame with comparative statistics
                          (mean, std, percentiles) for ID and OOD confidence scores.
                        - "per_threshold_metrics": A DataFrame showing TPR, FNR, FPR, and TNR
                          for both ID and OOD data at various thresholds.
                        - "ood_sample_analysis": A DataFrame with a column for the true label
                          and a column for each predicted species' confidence score.
                        - "id_sample_analysis": A DataFrame with a column for the true label
                          and a column for each predicted species' confidence score for the in-distribution data.
                        - "id_object_table": A pandas DataFrame with detailed information on each matched in-distribution object.
                        - "ood_object_table": A pandas DataFrame with detailed information on each matched out-of-distribution object.
    """
    # set default thresholds if none provided
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    # --- 1. Confidence Score Analysis ---
    id_max_confidences = [max(res["scores"]) if res and res.get("scores") else 0 for res in id_results]
    ood_max_confidences = [max(res["scores"]) if res and res.get("scores") else 0 for res in ood_results]

    confidence_stats = {
        "Metric": ["Mean", "Std Dev", "Median", "95th Percentile", "99th Percentile"],
        "In-Distribution": [
            np.mean(id_max_confidences),
            np.std(id_max_confidences),
            np.median(id_max_confidences),
            np.percentile(id_max_confidences, 95),
            np.percentile(id_max_confidences, 99),
        ],
        "Out-of-Distribution": [
            np.mean(ood_max_confidences),
            np.std(ood_max_confidences),
            np.median(ood_max_confidences),
            np.percentile(ood_max_confidences, 95),
            np.percentile(ood_max_confidences, 99),
        ],
    }
    confidence_df = pd.DataFrame(confidence_stats)

    # --- 2. Per-Threshold Metrics (TPR, FNR, FPR, TNR) ---
    threshold_metrics = []
    for t in thresholds:
        id_tp = np.sum([1 for conf in id_max_confidences if conf >= t])
        id_fn = len(id_max_confidences) - id_tp
        ood_fp = np.sum([1 for conf in ood_max_confidences if conf >= t])
        ood_tn = len(ood_max_confidences) - ood_fp

        tpr_id = id_tp / len(id_max_confidences) if id_max_confidences else 0
        fnr_id = id_fn / len(id_max_confidences) if id_max_confidences else 0
        fpr_ood = ood_fp / len(ood_max_confidences) if ood_max_confidences else 0
        tnr_ood = ood_tn / len(ood_max_confidences) if ood_max_confidences else 0

        threshold_metrics.append(
            {
                "Threshold": t,
                "TPR_ID": tpr_id,
                "FNR_ID": fnr_id,
                "FPR_OOD": fpr_ood,
                "TNR_OOD": tnr_ood,
            }
        )
    threshold_df = pd.DataFrame(threshold_metrics)

    # --- 3. OOD Sample-Level Analysis with Full Class Confidences ---
    ood_sample_details = []
    for i, res in enumerate(ood_results):
        true_label = "Unknown"
        if "objects" in ood_dataset[i] and "category" in ood_dataset[i]["objects"] and ood_dataset[i]["objects"]["category"]:
            true_label = ood_dataset[i]["objects"]["category"][0]
        row = {"true_label": true_label}

        if res and res.get("full_results"):
            # Assuming one detection per image for OOD, take the first one
            full_result = res["full_results"][0]
            for class_name, prob in zip(class_names, full_result["class_probabilities"]):
                row[class_name] = prob
        else:
            # If no detection, fill with zeros
            for class_name in class_names:
                row[class_name] = 0.0

        ood_sample_details.append(row)

    ood_sample_df = pd.DataFrame(ood_sample_details)

    # --- 4. ID Sample-Level Analysis with Full Class Confidences ---
    id_sample_details = []
    for i, res in enumerate(id_results):
        true_label = "Unknown"
        if "objects" in id_dataset[i] and "category" in id_dataset[i]["objects"] and id_dataset[i]["objects"]["category"]:
            true_label = id_dataset[i]["objects"]["category"][0]
        row = {"true_label": true_label}

        if res and res.get("full_results"):
            # Assuming one detection per image for ID, take the first one
            full_result = res["full_results"][0]
            for class_name, prob in zip(class_names, full_result["class_probabilities"]):
                row[class_name] = prob
        else:
            # If no detection, fill with zeros
            for class_name in class_names:
                row[class_name] = 0.0

        id_sample_details.append(row)

    id_sample_df = pd.DataFrame(id_sample_details)

    # --- 5. Object Table Generation ---
    def _generate_object_table(dataset, results, class_names):
        gt_by_image = defaultdict(lambda: {"boxes": [], "labels": [], "used": []})
        for i, item in enumerate(dataset):
            if "objects" in item and "bbox" in item["objects"] and "category" in item["objects"]:
                for j in range(len(item["objects"]["category"])):
                    label_name = item["objects"]["category"][j]
                    if label_name in class_names:
                        bbox = item["objects"]["bbox"][j]
                        x1, y1, w, h = bbox
                        x2 = x1 + w
                        y2 = y1 + h
                        gt_by_image[i]["boxes"].append([x1, y1, x2, y2])
                        gt_by_image[i]["labels"].append(class_names.index(label_name))
                        gt_by_image[i]["used"].append(False)

        preds_by_class = defaultdict(list)
        for i, res in enumerate(results):
            if res and res.get("boxes"):
                for box, label, score in zip(res["boxes"], res["labels"], res["scores"]):
                    if label in class_names:
                        preds_by_class[class_names.index(label)].append({"box": box, "score": score, "image_id": i})

        object_table = []
        for class_id, class_preds in preds_by_class.items():
            for pred in class_preds:
                gt_data = gt_by_image[pred["image_id"]]
                best_iou = -1.0
                best_gt_box = None

                for j, gt_box in enumerate(gt_data["boxes"]):
                    if gt_data["labels"][j] == class_id:
                        iou = _calculate_iou(pred["box"], gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_box = gt_box

                if best_iou >= 0.5:  # Using a fixed 0.5 IoU threshold for simplicity
                    object_table.append({"image_id": pred["image_id"], "gt_bbox": best_gt_box, "pred_bbox": pred["box"], "iou": best_iou})
        return pd.DataFrame(object_table)

    id_object_table = _generate_object_table(id_dataset, id_results, class_names)
    ood_object_table = _generate_object_table(ood_dataset, ood_results, class_names)

    return {
        "confidence_score_analysis": confidence_df,
        "per_threshold_metrics": threshold_df,
        "ood_sample_analysis": ood_sample_df,
        "id_sample_analysis": id_sample_df,
        "id_object_table": id_object_table,
        "ood_object_table": ood_object_table,
    }
