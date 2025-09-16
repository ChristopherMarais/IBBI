# src/ibbi/evaluate/classification.py

from typing import Any, Optional

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
    target_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Calculates a suite of classification metrics.
    """
    labels_lst = list(set(true_labels) | set(predicted_labels))
    all_labels = target_names if target_names is not None else sorted(labels_lst)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="macro", zero_division="warn", labels=all_labels
    )

    cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=all_labels,
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
