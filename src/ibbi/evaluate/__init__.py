# src/ibbi/evaluate/__init__.py

"""
Provides the high-level Evaluator class for assessing model performance.
"""

from typing import Any, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from ..models import ModelType
from .classification import classification_performance
from .embeddings import EmbeddingEvaluator
from .object_detection import object_detection_performance


class Evaluator:
    """
    A unified evaluator for assessing IBBI models on various tasks.
    """

    def __init__(self, model: ModelType):
        self.model = model

    def classification(self, dataset, predict_kwargs: Optional[dict[str, Any]] = None, **kwargs):
        """
        Runs a full classification performance analysis.
        """
        if predict_kwargs is None:
            predict_kwargs = {}

        print("Running classification evaluation...")
        true_labels = [item["label"] for item in dataset]
        predicted_labels = []

        print("Making predictions for classification report...")
        for item in tqdm(dataset):
            results = self.model.predict(item["image"], verbose=False, **predict_kwargs)
            if not results:
                predicted_labels.append(-1)
                continue

            # Check if boxes exist before trying to access them
            pred_classes = {int(box.cls) for res in results if res.boxes is not None for box in res.boxes}

            if item["label"] in pred_classes:
                predicted_labels.append(item["label"])
            elif pred_classes:
                # Fallback to the highest confidence prediction
                all_boxes = [box for res in results if res.boxes is not None for box in res.boxes]
                if all_boxes:
                    highest_conf_box = max(all_boxes, key=lambda b: b.conf.item())
                    predicted_labels.append(int(highest_conf_box.cls))
                else:
                    predicted_labels.append(-1)
            else:
                predicted_labels.append(-1)

        return classification_performance(np.array(true_labels), np.array(predicted_labels), **kwargs)

    def object_detection(
        self, dataset, iou_thresholds: Union[float, list[float]] = 0.5, predict_kwargs: Optional[dict[str, Any]] = None
    ):
        """
        Runs a mean Average Precision (mAP) object detection analysis.
        """
        if predict_kwargs is None:
            predict_kwargs = {}

        print("Running object detection evaluation...")
        gt_boxes, gt_labels, gt_image_ids = [], [], []
        pred_boxes, pred_labels, pred_scores, pred_image_ids = [], [], [], []

        print("Extracting ground truth and making predictions for mAP...")
        for i, item in enumerate(tqdm(dataset)):
            if "bboxes" in item and "cls" in item and item["cls"] is not None:
                for j in range(len(item["cls"])):
                    gt_boxes.append(item["bboxes"][j])
                    gt_labels.append(item["cls"][j])
                    gt_image_ids.append(i)

            results = self.model.predict(item["image"], verbose=False, **predict_kwargs)
            if not results:
                continue

            for res in results:
                if hasattr(res, "boxes") and res.boxes is not None:
                    for box in res.boxes:
                        box_coords = box.xyxy[0]
                        if isinstance(box_coords, torch.Tensor):
                            box_coords = box_coords.cpu().numpy()
                        pred_boxes.append(box_coords.flatten())
                        pred_labels.append(int(box.cls))
                        pred_scores.append(float(box.conf))
                        pred_image_ids.append(i)

        return object_detection_performance(
            np.array(gt_boxes),
            gt_labels,
            gt_image_ids,
            np.array(pred_boxes),
            pred_labels,
            pred_scores,
            pred_image_ids,
            iou_thresholds=iou_thresholds,
        )

    def embeddings(
        self,
        dataset,
        use_umap: bool = True,
        extract_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Evaluates the quality of the model's feature embeddings.
        """
        if extract_kwargs is None:
            extract_kwargs = {}

        print("Extracting embeddings for evaluation...")
        embeddings_list = [self.model.extract_features(item["image"], **extract_kwargs) for item in tqdm(dataset)]
        embeddings = np.array([emb.flatten() for emb in embeddings_list if emb is not None])

        if embeddings.shape[0] == 0:
            print("Warning: Could not extract any valid embeddings from the dataset.")
            return {}

        evaluator = EmbeddingEvaluator(embeddings, use_umap=use_umap, **kwargs)

        results = {}
        results.update(evaluator.evaluate_cluster_structure())

        if "label" in dataset.column_names:
            true_labels = np.array([item["label"] for item in dataset])
            results.update(evaluator.evaluate_against_truth(true_labels))

            try:
                mantel_corr, p_val, n = evaluator.compare_to_distance_matrix(true_labels)
                results["mantel_correlation"] = {"r": mantel_corr, "p_value": p_val, "n_items": n}
            except (ImportError, FileNotFoundError, ValueError) as e:
                print(f"Could not run Mantel test: {e}")

        return results


__all__ = ["Evaluator"]
