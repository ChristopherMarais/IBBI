# src/ibbi/evaluate/embeddings.py

from importlib import resources as pkg_resources
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

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
        )
        self.predicted_labels = clusterer.fit_predict(self.processed_data)
        print("HDBSCAN clustering complete.")

    def evaluate_against_truth(self, true_labels: np.ndarray) -> dict[str, float]:
        """
        Calculates external clustering validation metrics against ground truth labels.
        """
        ari = adjusted_rand_score(true_labels, self.predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, self.predicted_labels)
        return {"ARI": ari, "NMI": nmi}

    def evaluate_cluster_structure(self) -> dict[str, float]:
        """
        Calculates internal clustering validation metrics based on cluster structure.
        """
        mask = self.predicted_labels != -1
        if np.sum(mask) < 2 or len(set(self.predicted_labels[mask])) < 2:
            return {"Silhouette_Score": -1.0, "Davies-Bouldin_Index": -1.0}

        filtered_data = self.processed_data[mask]
        filtered_labels = self.predicted_labels[mask]

        silhouette = silhouette_score(filtered_data, filtered_labels)
        dbi = davies_bouldin_score(filtered_data, filtered_labels)
        return {"Silhouette_Score": silhouette, "Davies-Bouldin_Index": dbi}

    def compare_to_distance_matrix(
        self,
        true_labels: np.ndarray,
        label_map: Optional[dict[int, str]] = None,
        embedding_metric: str = "cosine",
        ext_dist_matrix: Optional[np.ndarray] = None,
        ext_dist_labels: Optional[list[str]] = None,
    ) -> tuple[float, float, int]:
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
            with pkg_resources.path("ibbi.data", "ibbi_species_distance_matrix.csv") as data_file_path:
                ext_matrix_df = pd.read_csv(str(data_file_path), index_col=0)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "The 'ibbi_species_distance_matrix.csv' file was not found within the package data. "
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

        return float(r_val), float(p_val), int(n_items)
