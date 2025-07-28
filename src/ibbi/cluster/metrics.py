import importlib.resources

import numpy as np
import pandas as pd
from skbio.stats.distance import mantel
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_score,
)


def get_relatedness_matrix(path_to_csv: str = r"../data/ibbi_species_relatedness_matrix.csv") -> pd.DataFrame:
    """
    Loads the species relatedness matrix from the package data.

    Args:
        path_to_csv (str, optional): An explicit path to a CSV file. If None,
                                     it loads the default matrix from the package.

    Returns:
        pd.DataFrame: A DataFrame containing the species relatedness matrix.
    """
    if path_to_csv:
        return pd.read_csv(path_to_csv, index_col=0)
    else:
        # This reliably finds the data file within the installed package
        with importlib.resources.path("ibbi.data", "relatedness_matrix.csv") as path:
            return pd.read_csv(path, index_col=0)


def calculate_ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Calculates the Adjusted Rand Index (ARI)."""
    return adjusted_rand_score(labels_true, labels_pred)


def calculate_nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Calculates the Normalized Mutual Information (NMI)."""
    return normalized_mutual_info_score(labels_true, labels_pred)


def calculate_silhouette(embeddings: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> float:
    """Calculates the Silhouette Score. Returns -1 if only one cluster is found."""
    if len(np.unique(labels)) < 2:
        return -1.0
    return silhouette_score(embeddings, labels, metric=metric)


def calculate_davies_bouldin(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Calculates the Davies-Bouldin Index. Returns infinity if only one cluster is found."""
    if len(np.unique(labels)) < 2:
        return float("inf")
    return davies_bouldin_score(embeddings, labels)


def relatedness_correlation_score(
    embeddings: np.ndarray, species_labels: list, relatedness_matrix: pd.DataFrame, embedding_metric: str = "cosine"
) -> tuple:
    """
    Evaluates clustering by comparing the similarity of embeddings with
    the genetic relatedness of species using a Mantel test.

    A high positive correlation indicates that the model's feature space
    aligns well with the biological ground truth.

    Args:
        embeddings (np.ndarray): The feature embeddings from the model.
        species_labels (list): A list of species names corresponding to each embedding.
        relatedness_matrix (pd.DataFrame): The genetic relatedness matrix.
        embedding_metric (str): The metric for calculating distances between embeddings.

    Returns:
        tuple: A tuple containing:
            - corr (float): The Mantel correlation coefficient (r).
            - p_value (float): The p-value for the correlation.
    """
    # 1. Create the model's distance matrix from embeddings
    model_dist_matrix = pairwise_distances(embeddings, metric=embedding_metric)

    # 2. Create the genetic distance matrix from species labels
    num_samples = len(species_labels)
    genetic_dist_matrix = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i, num_samples):
            species1 = species_labels[i]
            species2 = species_labels[j]
            # Assuming higher relatedness means lower distance, so we take 1 - relatedness
            distance = 1 - relatedness_matrix.loc[species1, species2]
            genetic_dist_matrix[i, j] = genetic_dist_matrix[j, i] = distance

    # 3. Perform the Mantel test
    corr, p_value, _ = mantel(x=model_dist_matrix, y=genetic_dist_matrix, method="pearson", permutations=999)

    return corr, p_value
