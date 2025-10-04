# src/ibbi/evaluate/__init__.py

"""
Provides the high-level Evaluator class for comprehensive model assessment.

This module serves as the primary interface for evaluating models within the `ibbi` package.
It introduces the `Evaluator` class, which streamlines the process of assessing models
on various tasks, including object detection, classification, and embedding quality.
By handling the boilerplate code for dataset iteration, prediction, and metric calculation,
the `Evaluator` class allows users to focus on interpreting the results.
"""

from typing import Any, Optional, Union

import numpy as np
from tqdm import tqdm

from ..models import ModelType
from ..models.feature_extractors import HuggingFaceFeatureExtractor, UntrainedFeatureExtractor
from ..models.zero_shot import GroundingDINOModel, YOLOWorldModel
from .embeddings import EmbeddingEvaluator
from .object_classification import object_classification_performance


class Evaluator:
    """A unified evaluator for assessing IBBI models on various tasks.

    This class provides a streamlined interface for evaluating the performance of
    models on tasks such as object classification and embedding quality.
    It handles the boilerplate code for iterating through datasets, making predictions,
    and calculating a comprehensive suite of metrics for a holistic model assessment.

    The `Evaluator` is initialized with a model instance from the `ibbi` package.
    It provides methods to run different types of evaluations, returning detailed
    performance reports.

    Attributes:
        model (ModelType): The instantiated `ibbi` model to be evaluated.
    """

    def __init__(self, model: ModelType):
        """Initializes the Evaluator with a specific model.

        Args:
            model (ModelType): The model to be evaluated. This should be an instance of a class
                               that adheres to the `ModelType` protocol, meaning it has `predict`
                               and `extract_features` methods.
        """
        self.model = model

    def object_classification(
        self, dataset, iou_thresholds: Union[float, list[float]] = 0.5, predict_kwargs: Optional[dict[str, Any]] = None, **kwargs
    ):
        """Runs a comprehensive object detection and classification performance analysis.

        This method assesses the model's ability to both accurately localize and correctly
        classify objects within a dataset. It iterates through the provided dataset, gathering
        ground truth information and generating model predictions. These are then passed to the
        `object_classification_performance` function to compute a detailed suite of metrics.

        The evaluation provides a holistic view of performance, combining traditional object
        detection metrics (like mAP) with a full suite of classification metrics for each IoU
        threshold.

        Args:
            dataset (iterable): An iterable dataset where each item is a dictionary-like object
                                containing at least an 'image' key. For evaluation, items should
                                also contain an 'objects' key, which is a dictionary with 'bbox'
                                and 'category' keys.
            iou_thresholds (Union[float, list[float]], optional): The IoU threshold(s) at which
                to compute mAP and classification metrics. Can be a single float or a list of floats.
                Defaults to 0.5.
            predict_kwargs (Optional[dict[str, Any]], optional): A dictionary of keyword arguments
                to be passed directly to the model's `predict` method during evaluation.
                This is useful for model-specific parameters like `text_prompt` for zero-shot models.
                Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the underlying
                      `object_classification_performance` function (e.g., `average`, `zero_division`).

        Returns:
            dict: A dictionary containing a comprehensive set of object detection and
                  classification metrics, including per-iou threshold classification performance,
                  and a detailed object-level performance table.
        """
        if predict_kwargs is None:
            predict_kwargs = {}

        # Set classes for GroundingDINO before evaluation
        if isinstance(self.model, GroundingDINOModel):
            if "text_prompt" in predict_kwargs:
                self.model.set_classes(predict_kwargs["text_prompt"])

        print("Running object classification evaluation...")

        if isinstance(self.model, (HuggingFaceFeatureExtractor, UntrainedFeatureExtractor)):
            print("Warning: Object classification evaluation is not supported for pure feature extractors.")
            return {}

        if isinstance(self.model, (GroundingDINOModel, YOLOWorldModel)):
            if "text_prompt" not in predict_kwargs and not self.model.get_classes():
                print("Warning: Zero-shot model has no classes set. Please provide a 'text_prompt' in 'predict_kwargs'.")
                return {}

        gt_boxes, gt_labels, gt_image_ids, gt_label_names = [], [], [], []
        pred_results_with_probs = []  # Full prediction result per image
        # Initialize model_classes before the loop.
        model_classes: list[str] = []
        if isinstance(self.model, (GroundingDINOModel)):
            if hasattr(self.model, "get_classes") and callable(self.model.get_classes):
                raw_model_classes = self.model.get_classes()
                if isinstance(raw_model_classes, dict):
                    model_classes = list(raw_model_classes.values())
                else:
                    model_classes = raw_model_classes
        class_name_to_idx: dict[str, int] = {}
        idx_to_name: dict[int, str] = {}

        print("Extracting ground truth and making predictions...")
        predict_kwargs_for_call = {**predict_kwargs, "include_full_probabilities": True}

        for i, item in enumerate(tqdm(dataset)):
            # Make the first prediction to set classes for YOLOWorld
            results = self.model.predict(item["image"], verbose=False, **predict_kwargs_for_call)
            pred_results_with_probs.append(results)

            if not model_classes:
                if not hasattr(self.model, "get_classes") or not callable(self.model.get_classes):
                    print("Warning: Model does not have a 'get_classes' method for class mapping. Skipping evaluation.")
                    return {}

                raw_model_classes = self.model.get_classes()
                if isinstance(raw_model_classes, dict):
                    model_classes: list[str] = list(raw_model_classes.values())
                else:
                    model_classes: list[str] = raw_model_classes

                if not model_classes:
                    print("Warning: Model returned an empty class list. Cannot proceed with classification-dependent metrics.")
                    return {}

                class_name_to_idx = {v: k for k, v in enumerate(model_classes)}
                idx_to_name = dict(enumerate(model_classes))

            # --- Extract Ground Truth ---
            if "objects" in item and "bbox" in item["objects"] and "category" in item["objects"]:
                for j in range(len(item["objects"]["category"])):
                    label_name = item["objects"]["category"][j]
                    gt_label_names.append(label_name)
                    bbox = item["objects"]["bbox"][j]
                    x1, y1, w, h = bbox
                    x2 = x1 + w
                    y2 = y1 + h
                    gt_boxes.append([x1, y1, x2, y2])
                    gt_labels.append(class_name_to_idx.get(label_name, -1))

                    gt_image_ids.append(i)

        # The GT and raw prediction data is prepared. Now run the core evaluation logic.
        performance_results = object_classification_performance(
            np.array(gt_boxes),
            gt_labels,
            gt_image_ids,
            pred_results_with_probs,
            gt_label_names=gt_label_names,
            iou_thresholds=iou_thresholds,
            model_classes=model_classes,
            idx_to_name=idx_to_name,
            **kwargs,
        )

        # Apply naming to the mAP results
        if "per_class_AP_at_last_iou" in performance_results:
            class_aps = performance_results["per_class_AP_at_last_iou"]
            named_class_aps = {idx_to_name.get(class_id, class_id): ap for class_id, ap in class_aps.items()}
            performance_results["per_class_AP_at_last_iou"] = named_class_aps

        return performance_results

    def embeddings(
        self,
        dataset,
        evaluation_level: str = "image",
        use_umap: bool = True,
        extract_kwargs: Optional[dict[str, Any]] = None,
        batch_size: int = 32,
        **kwargs,
    ):
        """Evaluates the quality of the model's feature embeddings.

        This method extracts feature embeddings from the provided dataset. It can operate
        at two levels: 'image' (extracting one embedding per image) or 'object'
        (extracting an embedding for each annotated object in each image). The quality of
        these embeddings is then assessed using clustering algorithms and a suite of
        internal and external validation metrics.

        Args:
            dataset (iterable): An iterable dataset where each item contains an 'image' key.
                                For 'object' level evaluation, items should also contain 'objects'
                                with 'bbox' and 'category' keys.
            evaluation_level (str, optional): The level at which to evaluate embeddings.
                                              Can be "image" or "object". Defaults to "image".
            use_umap (bool, optional): If True, applies UMAP for dimensionality reduction
                                       before clustering. Defaults to True.
            extract_kwargs (Optional[dict[str, Any]], optional): Keyword arguments to be passed
                to the model's `extract_features` method. Defaults to None.
            batch_size (int, optional): The batch size for GPU distance matrix calculation.
                                        Defaults to 32.
            **kwargs: Additional keyword arguments to be passed to the `EmbeddingEvaluator`.
                      See `ibbi.evaluate.embeddings.EmbeddingEvaluator` for more details.

        Returns:
            dict: A dictionary containing the results of the embedding evaluation, including
                  clustering metrics and optionally, correlation with external data.
        """
        if extract_kwargs is None:
            extract_kwargs = {}
        if evaluation_level not in ["image", "object"]:
            raise ValueError("evaluation_level must be either 'image' or 'object'.")

        print(f"Extracting embeddings for evaluation at the '{evaluation_level}' level...")
        embeddings_list = []
        true_labels = []
        valid_indices = []

        # Pre-calculate label mappings for efficiency
        unique_labels_lst = list(set(cat for item in dataset for cat in item.get("objects", {}).get("category", [])))
        unique_labels = sorted(unique_labels_lst)
        name_to_idx = {name: i for i, name in enumerate(unique_labels)}
        idx_to_name = dict(enumerate(unique_labels))

        for i, item in enumerate(tqdm(dataset)):
            if evaluation_level == "image":
                embedding = self.model.extract_features(item["image"], **extract_kwargs)
                if embedding is not None:
                    embeddings_list.append(embedding)
                    if "objects" in item and "category" in item["objects"] and item["objects"]["category"]:
                        label_name = item["objects"]["category"][0]
                        if label_name in name_to_idx:
                            true_labels.append(name_to_idx[label_name])
                            valid_indices.append(len(embeddings_list) - 1)

            elif evaluation_level == "object":
                if "objects" not in item or "bbox" not in item["objects"] or "category" not in item["objects"]:
                    continue

                original_image = item["image"]
                for j, bbox in enumerate(item["objects"]["bbox"]):
                    x, y, w, h = bbox
                    if w > 0 and h > 0:
                        cropped_image = original_image.crop((x, y, x + w, y + h))
                        embedding = self.model.extract_features(cropped_image, **extract_kwargs)
                        if embedding is not None:
                            embeddings_list.append(embedding)
                            label_name = item["objects"]["category"][j]
                            if label_name in name_to_idx:
                                true_labels.append(name_to_idx[label_name])
                                valid_indices.append(len(embeddings_list) - 1)

        if not embeddings_list:
            print("Warning: Could not extract any valid embeddings from the dataset.")
            return {}

        embeddings = np.array([emb.cpu().numpy().flatten() for emb in embeddings_list])
        evaluator = EmbeddingEvaluator(embeddings, use_umap=use_umap, **kwargs)

        results = {}
        results["internal_cluster_validation"] = evaluator.evaluate_cluster_structure()

        if true_labels:
            true_labels = np.array(true_labels)
            results["external_cluster_validation"] = evaluator.evaluate_against_truth(true_labels)
            results["sample_results"] = evaluator.get_sample_results(true_labels, label_map=idx_to_name)

            try:
                if len(np.unique(true_labels)) >= 3:
                    valid_embeddings = embeddings[valid_indices]
                    evaluator_for_mantel = EmbeddingEvaluator(valid_embeddings, use_umap=False)
                    mantel_corr, p_val, n, per_class_df = evaluator_for_mantel.compare_to_distance_matrix(
                        true_labels, label_map=idx_to_name, batch_size=batch_size
                    )
                    results["mantel_correlation"] = {"r": mantel_corr, "p_value": p_val, "n_items": n}
                    results["per_class_centroids"] = per_class_df
                else:
                    print("Not enough unique labels in the dataset subset to run the Mantel test.")
            except (ImportError, FileNotFoundError, ValueError) as e:
                print(f"Could not run Mantel test: {e}")
        else:
            print("Dataset does not have the required 'objects' and 'category' fields for external validation.")
            results["sample_results"] = evaluator.get_sample_results()

        return results


__all__ = ["Evaluator"]
