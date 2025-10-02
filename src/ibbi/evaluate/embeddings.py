# src/ibbi/evaluate/embeddings.py

from importlib import resources as pkg_resources
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder

# --- Library Import Checks ---
try:
    import umap

    _umap_available = True
except ImportError:
    _umap_available = False

try:
    from skbio.stats.distance import mantel

    _skbio_available = True
except ImportError:
    _skbio_available = False

if TYPE_CHECKING:
    import umap
    from skbio.stats.distance import mantel


def _cluster_purity(y_true, y_pred):
    """Calculates cluster purity.

    This is a simple metric to evaluate the quality of a clustering result by measuring the
    extent to which each cluster contains data points from a single true class.

    Args:
        y_true (array-like): The ground truth labels.
        y_pred (array-like): The predicted cluster labels.

    Returns:
        float: The cluster purity score, ranging from 0 to 1, where 1 indicates perfect purity.
    """
    contingency_matrix = np.histogram2d(y_true, y_pred, bins=(len(np.unique(y_true)), len(np.unique(y_pred))))[0]
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


class EmbeddingEvaluator:
    """A unified class to evaluate feature embeddings.

    This class provides a comprehensive suite of tools for evaluating the quality of
    feature embeddings. It can perform dimensionality reduction using UMAP, cluster the
    embeddings using HDBSCAN, and then calculate a variety of internal and external
    validation metrics.

    Attributes:
        embeddings (np.ndarray): The raw, high-dimensional feature embeddings.
        processed_data (np.ndarray): The embeddings after optional UMAP dimensionality reduction.
                                     This is the data used for clustering.
        predicted_labels (np.ndarray): The cluster labels assigned by HDBSCAN.
    """

    embeddings: np.ndarray
    processed_data: np.ndarray
    predicted_labels: np.ndarray

    def __init__(
        self,
        embeddings: np.ndarray,
        use_umap: bool = True,
        # --- UMAP Parameters ---
        n_neighbors: int = 15,
        n_components: int = 2,
        min_dist: float = 0.1,
        umap_metric: str = "cosine",
        # --- HDBSCAN Parameters ---
        min_cluster_size: int = 15,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        hdbscan_metric: str = "euclidean",
        allow_single_cluster: bool = False,
        random_state: int = 42,
    ):
        """Initializes the evaluator, performing dimensionality reduction and clustering.

        Args:
            embeddings (np.ndarray): A 2D numpy array of feature embeddings, where each row is a sample.
            use_umap (bool, optional): If True, UMAP will be used for dimensionality reduction before clustering.
                                     Defaults to True.
            n_neighbors (int, optional): UMAP parameter. The number of nearest neighbors to consider for manifold approximation.
                                      See `umap.UMAP` for more details. Defaults to 15.
            n_components (int, optional): UMAP parameter. The dimension of the space to embed into.
                                        See `umap.UMAP` for more details. Defaults to 2.
            min_dist (float, optional): UMAP parameter. The minimum distance between embedded points.
                                      See `umap.UMAP` for more details. Defaults to 0.1.
            umap_metric (str, optional): UMAP parameter. The metric to use for distance computation in the high-dimensional space.
                                       See `umap.UMAP` for more details. Defaults to "cosine".
            min_cluster_size (int, optional): HDBSCAN parameter. The minimum size of clusters.
                                            See `sklearn.cluster.HDBSCAN` for more details. Defaults to 15.
            min_samples (Optional[int], optional): HDBSCAN parameter. The number of samples in a neighborhood for a point
                                                 to be considered as a core point. See `sklearn.cluster.HDBSCAN` for more details.
                                                 Defaults to None.
            cluster_selection_epsilon (float, optional): HDBSCAN parameter. A distance threshold. Clusters below this value will be merged.
                                                         See `sklearn.cluster.HDBSCAN` for more details. Defaults to 0.0.
            hdbscan_metric (str, optional): HDBSCAN parameter. The metric to use for clustering.
                                          See `sklearn.cluster.HDBSCAN` for more details. Defaults to "euclidean".
            allow_single_cluster (bool, optional): HDBSCAN parameter. Whether to allow HDBSCAN to return a single cluster.
                                                  See `sklearn.cluster.HDBSCAN` for more details. Defaults to False.
            random_state (int, optional): The random state for UMAP for reproducibility. Defaults to 42.
        """
        if use_umap and not _umap_available:
            raise ImportError("UMAP is selected but 'umap-learn' is not installed.")

        self.embeddings = embeddings
        self.processed_data = embeddings

        # --- 1. Dimensionality Reduction (Optional) ---
        if use_umap:
            print("Performing UMAP dimensionality reduction...")
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                metric=umap_metric,
                random_state=random_state,
            )
            self.processed_data = cast(np.ndarray, reducer.fit_transform(self.embeddings))
            # After UMAP, the natural space is Euclidean.
            clustering_metric = "euclidean"
            print("UMAP complete. Clustering metric set to 'euclidean'.")
        else:
            clustering_metric = hdbscan_metric

        # --- 2. Clustering ---
        print("Performing HDBSCAN clustering...")
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=clustering_metric,
            allow_single_cluster=allow_single_cluster,
        )
        self.predicted_labels = clusterer.fit_predict(self.processed_data)
        print("HDBSCAN clustering complete.")

    def get_sample_results(self, true_labels: Optional[np.ndarray] = None, label_map: Optional[dict[int, str]] = None) -> pd.DataFrame:
        """Returns a DataFrame with true and predicted cluster labels for each sample.

        Args:
            true_labels (Optional[np.ndarray], optional): The ground truth labels for each sample.
                                                         If provided, they will be included in the output DataFrame.
                                                         Defaults to None.
            label_map (Optional[dict[int, str]], optional): A dictionary to map integer labels to string names.
                                                          Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame with columns for true and predicted labels.
        """
        results_df = pd.DataFrame()
        if true_labels is not None:
            results_df["true_label"] = true_labels
            if label_map:
                results_df["true_label"] = results_df["true_label"].map(lambda x: label_map.get(x))

        results_df["predicted_label"] = self.predicted_labels
        if label_map:
            # Map predicted labels, handling noise (-1) separately
            predicted_map = {k: v for k, v in label_map.items() if k != -1}
            results_df["predicted_label"] = results_df["predicted_label"].map(lambda x: predicted_map.get(x)).fillna("Noise")

        return results_df

    def evaluate_against_truth(self, true_labels: np.ndarray) -> pd.DataFrame:
        """Calculates external clustering validation metrics against ground truth labels.

        Args:
            true_labels (np.ndarray): An array of ground truth labels for each sample.

        Returns:
            pd.DataFrame: A DataFrame containing external validation metrics like ARI, NMI, and Cluster Purity.
        """
        # Filter out noise from predictions for a fair comparison
        mask = self.predicted_labels != -1
        if not np.any(mask):
            return pd.DataFrame([{"ARI": 0, "NMI": 0, "Cluster_Purity": 0}])

        filtered_true = true_labels[mask]
        filtered_pred = self.predicted_labels[mask]

        if len(np.unique(filtered_true)) < 2 or len(np.unique(filtered_pred)) < 2:
            return pd.DataFrame([{"ARI": 0, "NMI": 0, "Cluster_Purity": 0}])

        le_true = LabelEncoder().fit(filtered_true)
        true_labels_encoded = le_true.transform(filtered_true)
        predicted_labels_encoded = LabelEncoder().fit_transform(filtered_pred)

        ari = adjusted_rand_score(true_labels_encoded, predicted_labels_encoded)
        nmi = normalized_mutual_info_score(true_labels_encoded, predicted_labels_encoded)
        purity = _cluster_purity(true_labels_encoded, predicted_labels_encoded)

        metrics = {"ARI": ari, "NMI": nmi, "Cluster_Purity": purity}

        return pd.DataFrame([metrics])

    def evaluate_cluster_structure(self) -> pd.DataFrame:
        """Calculates internal clustering validation metrics based on cluster structure.

        Returns:
            pd.DataFrame: A DataFrame containing internal validation metrics like Silhouette Score,
                          Davies-Bouldin Index, and Calinski-Harabasz Index.
        """
        mask = self.predicted_labels != -1
        if np.sum(mask) < 2 or len(set(self.predicted_labels[mask])) < 2:
            return pd.DataFrame([{"Silhouette_Score": -1.0, "Davies-Bouldin_Index": -1.0, "Calinski-Harabasz_Index": -1.0}])

        filtered_data = self.processed_data[mask]
        filtered_labels = self.predicted_labels[mask]

        silhouette = silhouette_score(filtered_data, filtered_labels)
        dbi = davies_bouldin_score(filtered_data, filtered_labels)
        chi = calinski_harabasz_score(filtered_data, filtered_labels)

        metrics = {"Silhouette_Score": silhouette, "Davies-Bouldin_Index": dbi, "Calinski-Harabasz_Index": chi}
        return pd.DataFrame([metrics])

    def compare_to_distance_matrix(
        self,
        true_labels: np.ndarray,
        label_map: Optional[dict[int, str]] = None,
        embedding_metric: str = "cosine",
        ext_dist_matrix_path: str = "ibbi_species_distance_matrix.csv",
        batch_size: int = 32,  # Added batch_size parameter
    ) -> tuple[float, float, int, pd.DataFrame]:
        """Calculates Mantel correlation between embedding distances and an external distance matrix.
        The default is to use a distance matrix based on phylogenetic and taxonomic distance between species.
        This version computes the average pairwise distances between all embeddings for each pair of species
        and leverages a GPU if available.

        Args:
            true_labels (np.ndarray): An array of ground truth labels for each sample.
            label_map (Optional[dict[int, str]], optional): A dictionary to map integer labels to string names.
                                                        Defaults to None.
            embedding_metric (str, optional): The distance metric to use for the embedding space ('cosine' or 'euclidean').
                                            Defaults to "cosine".
            ext_dist_matrix_path (str, optional): The path to the external distance matrix file.
                                                Defaults to "ibbi_species_distance_matrix.csv".
            batch_size (int, optional): The batch size for GPU distance matrix calculation.
                                        Defaults to 32.

        Returns:
            tuple[float, float, int, pd.DataFrame]: A tuple containing the Mantel correlation coefficient (r),
                                                    the p-value, the number of items compared, and a DataFrame
                                                    of the mean embedding vector for each class (for inspection).
        """
        if not _skbio_available:
            raise ImportError("Mantel test requires 'scikit-bio' to be installed.")

        # --- 1. Create full pairwise distance matrix from original embeddings using PyTorch ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device} for distance matrix calculation.")

        embeddings_tensor = torch.tensor(self.embeddings, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(true_labels, dtype=torch.int64).to(device)

        num_embeddings = len(embeddings_tensor)
        dist_matrix = torch.zeros((num_embeddings, num_embeddings), device=device)

        for i in range(0, num_embeddings, batch_size):
            i_end = min(i + batch_size, num_embeddings)
            batch_i = embeddings_tensor[i:i_end]

            for j in range(0, num_embeddings, batch_size):
                j_end = min(j + batch_size, num_embeddings)
                batch_j = embeddings_tensor[j:j_end]

                if embedding_metric == "cosine":
                    sim = torch.nn.functional.cosine_similarity(batch_i[:, None, :], batch_j[None, :, :], dim=-1)
                    dist_matrix[i:i_end, j:j_end] = 1 - sim
                elif embedding_metric == "euclidean":
                    dist_matrix[i:i_end, j:j_end] = torch.cdist(batch_i, batch_j, p=2)
                else:
                    raise ValueError("Unsupported embedding_metric. Choose 'cosine' or 'euclidean'.")

        # --- 2. Aggregate pairwise distances to get average inter-species distances ---
        unique_labels = torch.unique(labels_tensor)
        unique_labels = unique_labels[unique_labels != -1]  # Exclude noise if present
        num_labels = len(unique_labels)
        avg_dist_matrix = torch.zeros((num_labels, num_labels), device=device)

        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels):
                if i <= j:
                    mask1 = labels_tensor == label1
                    mask2 = labels_tensor == label2

                    relevant_dists = dist_matrix[mask1][:, mask2]

                    if relevant_dists.numel() > 0:
                        avg_dist_matrix[i, j] = relevant_dists.mean()
                        avg_dist_matrix[j, i] = avg_dist_matrix[i, j]

        # Convert to pandas DataFrame for alignment
        sorted_labels, _ = torch.sort(unique_labels)
        sorted_labels_np = sorted_labels.cpu().numpy()
        if label_map:
            class_names = [label_map.get(lbl, f"unknown_{lbl}") for lbl in sorted_labels_np]
        else:
            class_names = [str(lbl) for lbl in sorted_labels_np]

        embedding_dist_matrix = pd.DataFrame(
            avg_dist_matrix.cpu().numpy(),
            index=pd.Index(class_names),
            columns=pd.Index(class_names),
        )

        try:
            with pkg_resources.path("ibbi.data", ext_dist_matrix_path) as data_file_path:
                ext_matrix_df = pd.read_csv(str(data_file_path), index_col=0)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"The '{ext_dist_matrix_path}' file was not found within the package data. "
                "Ensure the package was installed correctly with the data file included."
            ) from e

        # --- 3. Align matrices and run test ---
        common_labels_list = list(set(embedding_dist_matrix.index) & set(ext_matrix_df.index))
        common_labels = sorted(common_labels_list)

        if len(common_labels) < 3:
            raise ValueError("Need at least 3 overlapping labels between embedding groups and the external matrix to run Mantel test.")

        embedding_dist_aligned = embedding_dist_matrix.loc[common_labels, common_labels]
        ext_dist_aligned = ext_matrix_df.loc[common_labels, common_labels]

        # Ensure the matrices are hollow and have the correct data type
        embedding_dist_aligned_np = embedding_dist_aligned.to_numpy().astype(np.float32)
        ext_dist_aligned_np = ext_dist_aligned.to_numpy().astype(np.float32)
        np.fill_diagonal(embedding_dist_aligned_np, 0)
        np.fill_diagonal(ext_dist_aligned_np, 0)

        mantel_result = mantel(embedding_dist_aligned_np, ext_dist_aligned_np)
        typed_mantel_result = cast(tuple[float, float, int], mantel_result)

        r_val, p_val, n_items = typed_mantel_result

        # --- For supplementary output, calculate the mean embedding for each class ---
        labels_df = pd.DataFrame({"label": true_labels})
        embeddings_df = pd.DataFrame(self.embeddings)
        df = pd.concat([labels_df, embeddings_df], axis=1).dropna()

        grouped_embeddings = df.groupby("label").mean()
        mean_embeddings: np.ndarray = grouped_embeddings.to_numpy()
        class_labels_index: pd.Index = grouped_embeddings.index
        if label_map:
            class_labels_index = class_labels_index.map(label_map)

        per_class_mean_embeddings_df = pd.DataFrame(mean_embeddings, index=class_labels_index)
        per_class_mean_embeddings_df.index.name = "label"
        per_class_mean_embeddings_df = per_class_mean_embeddings_df.reset_index()

        return float(r_val), float(p_val), int(n_items), per_class_mean_embeddings_df
