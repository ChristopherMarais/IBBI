# src/ibbi/evaluate/__init__.py

from typing import Any, Optional, Union

import numpy as np
from tqdm import tqdm

from ..models import ModelType
from ..models.feature_extractors import HuggingFaceFeatureExtractor
from ..models.zero_shot import GroundingDINOModel
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

        if isinstance(self.model, (HuggingFaceFeatureExtractor, GroundingDINOModel)):
            print("Warning: Classification evaluation is not supported for this model type.")
            return {}

        if not hasattr(self.model, "get_classes") or not callable(self.model.get_classes):
            print("Warning: Model does not have a 'get_classes' method for class mapping. Skipping classification.")
            return {}

        raw_model_classes = self.model.get_classes()
        if isinstance(raw_model_classes, dict):
            model_classes = list(raw_model_classes.values())
        else:
            model_classes = raw_model_classes
        class_name_to_idx = {v: k for k, v in enumerate(model_classes)}
        true_labels = []
        items_for_prediction = []

        for item in dataset:
            if "objects" in item and "category" in item["objects"] and item["objects"]["category"]:
                label_name = item["objects"]["category"][0]
                if label_name in class_name_to_idx:
                    true_labels.append(class_name_to_idx[label_name])
                    items_for_prediction.append(item)

        if not true_labels:
            print(
                "Warning: No valid labels found in the dataset that match the model's classes. Skipping classification."
            )
            return {}

        predicted_labels = []

        print("Making predictions for classification report...")
        for i, item in enumerate(tqdm(items_for_prediction)):
            results = self.model.predict(item["image"], verbose=False, **predict_kwargs)
            true_label_for_item = true_labels[i]

            if not results or not results.get("labels"):
                predicted_labels.append(-1)
                continue

            pred_labels_for_item = results.get("labels", [])
            pred_classes = {class_name_to_idx[label] for label in pred_labels_for_item if label in class_name_to_idx}

            if true_label_for_item in pred_classes:
                predicted_labels.append(true_label_for_item)
            elif pred_labels_for_item:
                # Find the label with the highest score
                scores = results.get("scores", [])
                if scores:
                    highest_conf_idx = np.argmax(scores)
                    highest_conf_label = pred_labels_for_item[highest_conf_idx]
                    if highest_conf_label in class_name_to_idx:
                        predicted_labels.append(class_name_to_idx[highest_conf_label])
                    else:
                        predicted_labels.append(-1)
                else:
                    predicted_labels.append(-1)
            else:
                predicted_labels.append(-1)

        # This removes the duplicate 'target_names' argument from kwargs if it exists
        kwargs.pop("target_names", None)
        return classification_performance(
            np.array(true_labels), np.array(predicted_labels), target_names=model_classes, **kwargs
        )

    def object_detection(
        self, dataset, iou_thresholds: Union[float, list[float]] = 0.5, predict_kwargs: Optional[dict[str, Any]] = None
    ):
        """
        Runs a mean Average Precision (mAP) object detection analysis.
        """
        if predict_kwargs is None:
            predict_kwargs = {}

        print("Running object detection evaluation...")

        if isinstance(self.model, (HuggingFaceFeatureExtractor, GroundingDINOModel)):
            print("Warning: Object detection evaluation is not supported for this model type.")
            return {}

        if not hasattr(self.model, "get_classes") or not callable(self.model.get_classes):
            print("Warning: Model does not have a 'get_classes' method for class mapping. Skipping object detection.")
            return {}

        raw_model_classes = self.model.get_classes()
        if isinstance(raw_model_classes, dict):
            model_classes: list[str] = list(raw_model_classes.values())
        else:
            model_classes: list[str] = raw_model_classes
        class_name_to_idx = {v: k for k, v in enumerate(model_classes)}
        idx_to_name = dict(enumerate(model_classes))

        gt_boxes, gt_labels, gt_image_ids = [], [], []
        pred_boxes, pred_labels, pred_scores, pred_image_ids = [], [], [], []

        print("Extracting ground truth and making predictions for mAP...")
        for i, item in enumerate(tqdm(dataset)):
            if "objects" in item and "bbox" in item["objects"] and "category" in item["objects"]:
                for j in range(len(item["objects"]["category"])):
                    label_name = item["objects"]["category"][j]
                    if label_name in class_name_to_idx:
                        bbox = item["objects"]["bbox"][j]
                        x1, y1, w, h = bbox
                        x2 = x1 + w
                        y2 = y1 + h
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_labels.append(class_name_to_idx[label_name])
                        gt_image_ids.append(i)

            results = self.model.predict(item["image"], verbose=False, **predict_kwargs)
            if not results:
                continue

            if results and results.get("boxes"):
                for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
                    box_coords = np.array(box)
                    pred_boxes.append(box_coords.flatten())
                    if label in class_name_to_idx:
                        pred_labels.append(class_name_to_idx[label])
                    else:
                        pred_labels.append(-1)
                    pred_scores.append(score)
                    pred_image_ids.append(i)

        performance_results = object_detection_performance(
            np.array(gt_boxes),
            gt_labels,
            gt_image_ids,
            np.array(pred_boxes),
            pred_labels,
            pred_scores,
            pred_image_ids,
            iou_thresholds=iou_thresholds,
        )

        if "per_class_AP_at_last_iou" in performance_results:
            class_aps = performance_results["per_class_AP_at_last_iou"]
            named_class_aps = {
                idx_to_name.get(class_id, f"unknown_class_{class_id}"): ap for class_id, ap in class_aps.items()
            }
            performance_results["per_class_AP_at_last_iou"] = named_class_aps

        if "sample_results" in performance_results:
            performance_results["sample_results"]["label"] = performance_results["sample_results"]["label"].map(
                idx_to_name
            )

        return performance_results

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

        embeddings = np.array([emb.cpu().numpy().flatten() for emb in embeddings_list if emb is not None])

        if embeddings.shape[0] == 0:
            print("Warning: Could not extract any valid embeddings from the dataset.")
            return {}

        evaluator = EmbeddingEvaluator(embeddings, use_umap=use_umap, **kwargs)

        results = {}
        results["internal_cluster_validation"] = evaluator.evaluate_cluster_structure()

        if (
            "objects" in dataset.features
            and hasattr(dataset.features["objects"], "feature")
            and "category" in dataset.features["objects"].feature
        ):
            true_labels = []
            valid_indices = []
            unique_labels_lst = list(set(cat for item in dataset for cat in item["objects"]["category"]))
            unique_labels = sorted(unique_labels_lst)
            name_to_idx = {name: i for i, name in enumerate(unique_labels)}
            idx_to_name = dict(enumerate(unique_labels))

            for i, item in enumerate(dataset):
                if "objects" in item and "category" in item["objects"] and item["objects"]["category"]:
                    label_name = item["objects"]["category"][0]
                    if label_name in name_to_idx:
                        true_labels.append(name_to_idx[label_name])
                        valid_indices.append(i)

            if true_labels:
                true_labels = np.array(true_labels)
                results["external_cluster_validation"] = evaluator.evaluate_against_truth(true_labels)
                results["sample_results"] = evaluator.get_sample_results(true_labels, label_map=idx_to_name)

                try:
                    if len(np.unique(true_labels)) >= 3:
                        valid_embeddings = embeddings[valid_indices]
                        evaluator_for_mantel = EmbeddingEvaluator(valid_embeddings, use_umap=False)

                        mantel_corr, p_val, n, per_class_df = evaluator_for_mantel.compare_to_distance_matrix(
                            true_labels, label_map=idx_to_name
                        )
                        results["mantel_correlation"] = {"r": mantel_corr, "p_value": p_val, "n_items": n}
                        results["per_class_centroids"] = per_class_df
                    else:
                        print("Not enough unique labels in the dataset subset to run the Mantel test.")

                except (ImportError, FileNotFoundError, ValueError) as e:
                    print(f"Could not run Mantel test: {e}")
            else:
                results["sample_results"] = evaluator.get_sample_results()
        else:
            print("Dataset does not have the required 'objects' and 'category' fields for external validation.")
            results["sample_results"] = evaluator.get_sample_results()

        return results


__all__ = ["Evaluator"]
