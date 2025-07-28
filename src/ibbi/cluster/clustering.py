import hdbscan
import numpy as np


def perform_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 0,
    metric: str = "cosine",
    prediction_data: bool = True,
):
    """
    Performs HDBSCAN clustering on the given embeddings.

    HDBSCAN is a density-based clustering algorithm that is well-suited for
    data with varying cluster densities and noise. It can identify a hierarchy
    of clusters and is robust to parameter selection.

    Args:
        embeddings (np.ndarray): A 2D array of feature embeddings where
                                 rows are samples and columns are features.
        min_cluster_size (int): The minimum size of a cluster. This is the
                                most important parameter.
        min_samples (int, optional): The number of samples in a neighborhood
                                     for a point to be considered a core point.
                                     Defaults to min_cluster_size.
        metric (str): The distance metric to use (e.g., 'euclidean', 'cosine').
        prediction_data (bool): Whether to generate data required for predicting
                                cluster membership of new points. Set to True
                                if you plan to use the model to predict on
                                a test set.

    Returns:
        tuple: A tuple containing:
            - clusterer (hdbscan.HDBSCAN): The fitted clusterer object. Contains
                                           labels_, exemplars_, etc.
            - labels (np.ndarray): The cluster labels for each data point.
                                   Noise points are labeled -1.
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("Embeddings must be a NumPy array.")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        prediction_data=prediction_data,
        gen_min_span_tree=True,  # Useful for visualization
    )

    clusterer.fit(embeddings)

    return clusterer, clusterer.labels_
