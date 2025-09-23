# src/ibbi/evaluate/classification.py

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


def classification_performance(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    target_names: Optional[list[str]] = None,
    average: str = "macro",
    sample_weight: Optional[np.ndarray] = None,
    zero_division: Union[str, int] = "warn",  # type: ignore
) -> dict[str, Any]:
    """Calculates a comprehensive suite of classification metrics.

    This function provides a detailed analysis of classification performance by computing a variety
    of metrics. It returns a dictionary containing not only overall performance scores but also
    detailed reports like a confusion matrix and a classification report from scikit-learn.
    It also provides sample-level results, which can be useful for in-depth error analysis.

    Args:
        true_labels (np.ndarray): An array of ground truth labels for each sample.
        predicted_labels (np.ndarray): An array of predicted labels for each sample.
        target_names (Optional[list[str]], optional): A list of display names for the target classes.
                                                    If not provided, labels will be used as names. Defaults to None.
        average (str, optional): The averaging method to use for precision, recall, and F1-score.
                                Common options are 'micro', 'macro', and 'weighted'. Defaults to "macro".
        sample_weight (Optional[np.ndarray], optional): An array of weights to apply to each sample.
                                                        Defaults to None.
        zero_division (Union[str, int], optional): Sets the value to return when there is a zero division.
                                                Can be "warn", 0, or 1. Defaults to "warn".

    Returns:
        dict[str, Any]: A dictionary containing the following performance metrics:
                        - "accuracy": The overall accuracy score.
                        - "balanced_accuracy": The balanced accuracy score, which is useful for imbalanced datasets.
                        - f"{average}_precision": The precision score, averaged according to the `average` parameter.
                        - f"{average}_recall": The recall score, averaged according to the `average` parameter.
                        - f"{average}_f1_score": The F1-score, averaged according to the `average` parameter.
                        - "cohen_kappa": Cohen's Kappa score, which measures inter-annotator agreement.
                        - "matthews_corrcoef": The Matthews Correlation Coefficient, a robust metric for binary classification.
                        - "confusion_matrix_df": A pandas DataFrame representing the confusion matrix.
                        - "classification_report": A dictionary containing a detailed classification report from scikit-learn.
                        - "sample_results": A pandas DataFrame with the true and predicted labels for each sample.
    """
    labels_lst_unsorted = list(set(true_labels) | set(predicted_labels))
    labels_lst = sorted(labels_lst_unsorted)
    all_labels = target_names if target_names is not None else [str(label_var) for label_var in labels_lst]
    idx_to_name = dict(enumerate(target_names)) if target_names else {i: str(i) for i in labels_lst}

    # --- Core Metrics ---
    accuracy = accuracy_score(true_labels, predicted_labels, sample_weight=sample_weight)
    balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels, sample_weight=sample_weight)
    kappa = cohen_kappa_score(true_labels, predicted_labels, sample_weight=sample_weight)
    mcc = matthews_corrcoef(true_labels, predicted_labels, sample_weight=sample_weight)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average=average,  # type: ignore
        zero_division=zero_division,  # type: ignore
        labels=labels_lst,
        sample_weight=sample_weight,
    )

    # --- Detailed Reports and Matrices ---
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels_lst, sample_weight=sample_weight)
    cm_df = pd.DataFrame(cm, index=pd.Index(all_labels), columns=pd.Index(all_labels))
    cm_df.index.name = "True Label"
    cm_df.columns.name = "Predicted Label"

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=labels_lst,
        target_names=all_labels,
        output_dict=True,
        zero_division=zero_division,  # type: ignore
        sample_weight=sample_weight,
    )

    # --- Sample-level Results ---
    sample_results_df = pd.DataFrame({"true_label": true_labels, "predicted_label": predicted_labels})
    sample_results_df["true_label"] = sample_results_df["true_label"].map(lambda x: idx_to_name.get(x))
    sample_results_df["predicted_label"] = sample_results_df["predicted_label"].map(lambda x: idx_to_name.get(x))

    # --- Compile All Metrics ---
    performance_metrics = {
        # Overall Scores
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        f"{average}_precision": precision,
        f"{average}_recall": recall,
        f"{average}_f1_score": f1,
        "cohen_kappa": kappa,
        "matthews_corrcoef": mcc,
        # Detailed Reports
        "confusion_matrix_df": cm_df,
        "classification_report": report,
        # Sample-level Data
        "sample_results": sample_results_df,
    }

    return performance_metrics
