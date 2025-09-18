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
    """
    Calculates a comprehensive suite of classification metrics, providing granular control
    over hyperparameters and returning detailed, sample-level results.
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
        average=average,
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
