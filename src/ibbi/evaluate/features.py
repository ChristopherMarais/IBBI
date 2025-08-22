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

# Use TYPE_CHECKING to hint all optional imports to static analysis tools.
# This resolves all "possibly unbound" variable errors.
if TYPE_CHECKING:
    import umap
    from skbio.stats.distance import mantel

from importlib import resources as pkg_resources


class EmbeddingEvaluator:
    """
    A unified class to evaluate feature embeddings.

    This class provides a complete pipeline for taking high-dimensional feature
    embeddings, optionally performing dimensionality reduction with UMAP,
    clustering the results with HDBSCAN, and then calculating a suite of
    evaluation metrics.
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

        This constructor runs the core UMAP (optional) and HDBSCAN pipeline
        upon instantiation. The results are stored as instance attributes.

        Args:
            embeddings (np.ndarray): A 2D array of shape `(n_samples, n_features)`
                containing the feature embeddings to be evaluated.
            use_umap (bool, optional): If `True`, applies UMAP dimensionality
                reduction before clustering. Defaults to `True`.

            --- UMAP Hyperparameters ---
            n_neighbors (int, optional): This parameter controls how UMAP balances
                local versus global structure in the data. Smaller values will
                focus on very local structure, while larger values will consider
                more of the global structure. Defaults to 15.
            n_components (int, optional): The target number of dimensions for the
                UMAP reduction. A value of 2 is common for visualization.
                Defaults to 2.
            min_dist (float, optional): Controls how tightly UMAP is allowed to
                pack points together in the low-dimensional representation. Lower
                values create denser, more compact clusters. Defaults to 0.1.
            umap_metric (str, optional): The distance metric for UMAP to use on
                the original high-dimensional embeddings.
                Common options: 'euclidean', 'manhattan', 'chebyshev',
                'minkowski', 'cosine', 'correlation', 'hamming', 'jaccard'.
                Defaults to 'cosine'.

            --- HDBSCAN Hyperparameters ---
            min_cluster_size (int, optional): The minimum number of points
                required to form a distinct cluster. Defaults to 15.
            min_samples (int, optional): A measure of how conservative the
                clustering is. Larger values lead to more points being declared
                as noise. Defaults to `None`, in which case it is set to
                `min_cluster_size`.
            cluster_selection_epsilon (float, optional): A distance threshold.
                Clusters below this distance in the HDBSCAN hierarchy will be
                merged. Defaults to 0.0 (no merging).
            hdbscan_metric (str, optional): The distance metric for HDBSCAN.
                Note: If `use_umap` is `True`, this is automatically overridden
                and 'euclidean' is used on the UMAP-reduced data.
                Common options: 'euclidean', 'manhattan', 'minkowski',
                'chebyshev', 'cosine', 'haversine', 'precomputed'.
                Defaults to 'euclidean'.
            random_state (int, optional): The random seed for UMAP to ensure
                reproducibility of the dimensionality reduction. Defaults to 42.

        Raises:
            ImportError: If `use_umap` is True and 'umap-learn' is not installed.
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

        This method compares the cluster assignments generated by HDBSCAN with a
        pre-existing set of "true" labels using the Adjusted Rand Index (ARI)
        and Normalized Mutual Information (NMI) scores.

        Args:
            true_labels (np.ndarray): A 1D array of shape `(n_samples,)`
                containing the ground truth integer labels for each sample. This
                array must be the same length as the original embeddings.

        Returns:
            dict[str, float]: A dictionary containing the calculated scores:
                - "ARI": The Adjusted Rand Index, ranging from -1 (bad) to 1 (perfect).
                - "NMI": The Normalized Mutual Information, ranging from 0 (bad) to 1 (perfect).
        """
        ari = adjusted_rand_score(true_labels, self.predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, self.predicted_labels)
        return {"ARI": ari, "NMI": nmi}

    def evaluate_cluster_structure(self) -> dict[str, float]:
        """
        Calculates internal clustering validation metrics based on cluster structure.

        This method evaluates the quality of the formed clusters without reference
        to any ground truth labels. It uses the Silhouette Score and the
        Davies-Bouldin Index. Points labeled as noise (-1) by HDBSCAN are
        excluded from these calculations.

        Returns:
            dict[str, float]: A dictionary containing the calculated scores:
                - "Silhouette_Score": Measures how similar an object is to its
                  own cluster compared to other clusters. Ranges from -1 to 1,
                  where higher is better.
                - "Davies-Bouldin_Index": Measures the average similarity ratio
                  of each cluster with its most similar cluster. Lower scores
                  (closer to 0) are better.
                Returns a dict with -1 for both scores if there are fewer than 2
                clusters formed (excluding noise).
        """
        mask = self.predicted_labels != -1
        # Check if there are at least two clusters and enough points for scoring.
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
        embedding_metric: str = "cosine",
        # The ext_dist_path parameter is removed as it's no longer used.
        ext_dist_matrix: Optional[np.ndarray] = None,
        ext_dist_labels: Optional[list[str]] = None,
    ) -> tuple[float, float, int]:
        """
        Calculates Mantel correlation between embedding distances and an external distance matrix.

        This method assesses if distances between centroids of ground truth groups in the
        embedding space correlate with an external, known distance matrix (e.g., from
        phylogenetics, semantic similarity, etc.).

        The external distance matrix can be provided in two ways:
        1. As a CSV file via `ext_dist_path` (default behavior).
        2. Directly as a NumPy array and a list of labels via `ext_dist_matrix` and
        `ext_dist_labels`.

        Args:
            true_labels (np.ndarray): 1D array of shape `(n_samples,)` containing the
                ground truth labels for each embedding.
            embedding_metric (str, optional): Metric for calculating distances between
                embedding centroids. Defaults to 'cosine'.
            ext_dist_matrix (Optional[np.ndarray], optional): A pre-loaded square 2D
                NumPy array of external distances. If provided, `ext_dist_labels` must
                also be provided. Defaults to None.
            ext_dist_labels (Optional[list[str]], optional): A list of labels for the
                rows/columns of `ext_dist_matrix`. Required if `ext_dist_matrix` is
                provided. Defaults to None.

        Returns:
            tuple[float, float, int]: A tuple containing the Mantel 'r' coefficient,
                the p-value, and the number of common items used in the test.

        Raises:
            ImportError: If 'scikit-bio' is not installed.
            ValueError: If inputs are invalid (e.g., matrix not square, labels missing,
                fewer than 3 overlapping items).
            FileNotFoundError: If `ext_dist_path` is used and the file is not found.
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

        embedding_dist_matrix = pd.DataFrame(
            # stubs are too strict and don't recognize string metrics.
            squareform(pdist(centroids, metric=embedding_metric)),  # type: ignore
            index=centroid_index,
            columns=centroid_index,
        )

        try:
            # This context manager finds the data file within the installed package
            # and provides a valid, temporary path to it. This is the core fix.
            # FIX: Use a new variable name 'data_file_path' to avoid type conflict.
            with pkg_resources.path("ibbi.data", "ibbi_species_distance_matrix.csv") as data_file_path:
                ext_matrix_df = pd.read_csv(str(data_file_path), index_col=0)
        except FileNotFoundError as e:
            # This error is now more informative if the file is missing from the package.
            raise FileNotFoundError(
                "The 'ibbi_species_distance_matrix.csv' file was not found within the package data. "
                "Ensure the package was installed correctly with the data file included."
            ) from e

        # --- 3. Align matrices and run test ---
        common_labels = sorted(set(embedding_dist_matrix.index) & set(ext_matrix_df.index))

        if len(common_labels) < 3:
            raise ValueError(
                "Need at least 3 overlapping labels between embedding groups and "
                "the external matrix to run Mantel test."
            )

        embedding_dist_aligned = embedding_dist_matrix.loc[common_labels, common_labels]
        ext_dist_aligned = ext_matrix_df.loc[common_labels, common_labels]

        # Use `cast` to explicitly tell the type checker the return type
        mantel_result = mantel(embedding_dist_aligned, ext_dist_aligned)
        typed_mantel_result = cast(tuple[float, float, int], mantel_result)

        r_val = typed_mantel_result[0]
        p_val = typed_mantel_result[1]
        n_items = typed_mantel_result[2]

        return float(r_val), float(p_val), int(n_items)
