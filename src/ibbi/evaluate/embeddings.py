# src/ibbi/evaluate/embeddings.py

from importlib import resources as pkg_resources
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
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
    """
    Calculates cluster purity.
    """
    contingency_matrix = np.histogram2d(y_true, y_pred, bins=(len(np.unique(y_true)), len(np.unique(y_pred))))[0]
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


class EmbeddingEvaluator:
    """
    A unified class to evaluate feature embeddings.
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
        """
        Initializes the evaluator, performing dimensionality reduction and clustering.
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

    def get_sample_results(
        self, true_labels: Optional[np.ndarray] = None, label_map: Optional[dict[int, str]] = None
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with true labels (if provided) and predicted cluster
        labels for each sample, with text labels if a map is provided.
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
            results_df["predicted_label"] = (
                results_df["predicted_label"].map(lambda x: predicted_map.get(x)).fillna("Noise")
            )

        return results_df

    def evaluate_against_truth(self, true_labels: np.ndarray) -> pd.DataFrame:
        """
        Calculates external clustering validation metrics against ground truth labels.
        """
        # Filter out noise from predictions for a fair comparison
        mask = self.predicted_labels != -1
        if not np.any(mask):
            return pd.DataFrame([{"ARI": 0, "NMI": 0, "V-Measure": 0, "Cluster_Purity": 0}])

        filtered_true = true_labels[mask]
        filtered_pred = self.predicted_labels[mask]

        if len(np.unique(filtered_true)) < 2 or len(np.unique(filtered_pred)) < 2:
            return pd.DataFrame([{"ARI": 0, "NMI": 0, "V-Measure": 0, "Cluster_Purity": 0}])

        le_true = LabelEncoder().fit(filtered_true)
        true_labels_encoded = le_true.transform(filtered_true)
        predicted_labels_encoded = LabelEncoder().fit_transform(filtered_pred)

        ari = adjusted_rand_score(true_labels_encoded, predicted_labels_encoded)
        nmi = normalized_mutual_info_score(true_labels_encoded, predicted_labels_encoded)
        v_measure = v_measure_score(true_labels_encoded, predicted_labels_encoded)
        purity = _cluster_purity(true_labels_encoded, predicted_labels_encoded)

        metrics = {"ARI": ari, "NMI": nmi, "V-Measure": v_measure, "Cluster_Purity": purity}

        return pd.DataFrame([metrics])

    def evaluate_cluster_structure(self) -> pd.DataFrame:
        """
        Calculates internal clustering validation metrics based on cluster structure.
        """
        mask = self.predicted_labels != -1
        if np.sum(mask) < 2 or len(set(self.predicted_labels[mask])) < 2:
            return pd.DataFrame(
                [{"Silhouette_Score": -1.0, "Davies-Bouldin_Index": -1.0, "Calinski-Harabasz_Index": -1.0}]
            )

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
    ) -> tuple[float, float, int, pd.DataFrame]:
        """
        Calculates Mantel correlation between embedding distances and an external distance matrix.
        """
        if not _skbio_available:
            raise ImportError("Mantel test requires 'scikit-bio' to be installed.")

        # --- 1. Create embedding distance matrix from original embeddings ---
        labels_df = pd.DataFrame({"label": true_labels})
        embeddings_df = pd.DataFrame(self.embeddings)
        df = pd.concat([labels_df, embeddings_df], axis=1)

        grouped_centroids = df.groupby("label").mean()
        centroids: np.ndarray = grouped_centroids.to_numpy()
        centroid_index: pd.Index = grouped_centroids.index
        if label_map:
            # Map integer labels to species names
            centroid_index = centroid_index.map(label_map)

        embedding_dist_matrix = pd.DataFrame(
            squareform(pdist(centroids, metric=embedding_metric)),  # type: ignore
            index=centroid_index,
            columns=centroid_index,
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
            raise ValueError(
                "Need at least 3 overlapping labels between embedding groups and "
                "the external matrix to run Mantel test."
            )

        embedding_dist_aligned = embedding_dist_matrix.loc[common_labels, common_labels]
        ext_dist_aligned = ext_matrix_df.loc[common_labels, common_labels]

        mantel_result = mantel(embedding_dist_aligned, ext_dist_aligned)
        typed_mantel_result = cast(tuple[float, float, int], mantel_result)

        r_val = typed_mantel_result[0]
        p_val = typed_mantel_result[1]
        n_items = typed_mantel_result[2]

        per_class_results_df = pd.DataFrame({"label": centroid_index, "centroid": list(centroids)})

        return float(r_val), float(p_val), int(n_items), per_class_results_df
