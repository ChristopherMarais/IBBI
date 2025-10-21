---
license: mit
---
# Model Card: rtdetrx_bb_detect_model

## Model Details
- **Model Name:** `rtdetrx_bb_detect_model`
- **Model Type:** Single-Class Object Detection and Feature Extractor
- **Description:** This model is designed to detect the presence of bark beetles in images. It identifies and places a bounding box around the target but does not classify different species of bark beetles. It operates under the single class label: **'bark_beetle'**.

---

## Evaluation Datasets

To understand the model's capabilities, its performance was tested on two different types of datasets:

-   **In-Distribution (ID):** This dataset contains images that are **similar to the data the model was trained on**. Performance on this dataset shows how well the model performs on familiar types of images.
-   **Out-of-Distribution (OOD):** This dataset contains images that are **intentionally different species from the training data**.

---

## Performance

### Object Detection
The model's ability to correctly identify and locate bark beetles is measured by its **mean Average Precision (mAP)**. This metric evaluates both the accuracy of the bounding box placement and the classification confidence. The score is averaged over multiple Intersection over Union (IoU) thresholds, from 50% overlap (`0.50`) to 95% overlap (`0.95`), providing a comprehensive view of prediction accuracy. A higher mAP score indicates better performance.

| Dataset | mAP (0.50 : 0.95) | Notes |
| :--- | :--- | :--- |
| **In-Distribution (ID)** | 0.9243 | Excellent performance at lower precision requirements (AP@0.50 = 0.99), but accuracy drops for more precise bounding boxes (AP@0.95 = 0.72). |
| **Out-of-Distribution (OOD)**| 0.9541 | Superior performance with more consistent precision across all IoU thresholds (AP@0.95 = 0.81). |

<br>

### Feature Extraction (Embedding Performance)
The model can also convert images into numerical representations (embeddings). The quality of these embeddings is evaluated by how well they group similar species together in a feature space.

#### Internal Cluster Validation
These metrics measure the quality of the clusters formed by the embeddings without referring to ground-truth labels. They assess how dense and well-separated the clusters are.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Silhouette Score** | 0.6128 | 0.4248 | Measures how similar an object is to its own cluster compared to others. **Higher is better (closer to 1)**. The ID embeddings form better-defined clusters. |
| **Davies-Bouldin Index**| 0.3335 | 0.3406 | Measures the average similarity between each cluster and its most similar one. **Lower is better (closer to 0)**. The ID embeddings show slightly less overlap between clusters. |
| **Calinski-Harabasz Index**| 618.045 | 482.346 | Measures the ratio of between-cluster dispersion to within-cluster dispersion. **Higher is better**. The ID embeddings form denser and more separated clusters. |

#### External Cluster Validation
These metrics evaluate the clustering performance by comparing the results to the true species labels.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Adjusted Rand Index (ARI)** | 0.1283 | 0.0049 | Measures the similarity between true and predicted labels, correcting for chance. **Higher is better (closer to 1)**. |
| **Normalized Mutual Info (NMI)** | 0.4036 | 0.2612 | Measures the agreement between the clustering and the true labels. **Higher is better (closer to 1)**. |
| **Cluster Purity** | 0.2624 | 0.1142 | Measures the extent to which clusters contain a single class. **Higher is better (closer to 1)**. |

**Conclusion:** The external validation scores are low for both datasets, indicating the model's feature representations do **not** effectively separate different species of bark beetles on their own.

#### Phylogenetic Correlation (Mantel Test)
This test determines if the model's learned features correlate with the evolutionary relationships (phylogeny) between different bark beetle species.

-   **Mantel R-statistic:** This value ranges from -1 to 1. A positive value means species that are close in the model's feature space are also close evolutionarily. A value near zero indicates no correlation.
-   **p-value:** Indicates the statistical significance of the result. A p-value below 0.05 typically suggests a significant correlation.

| Dataset | Mantel R-statistic | p-value | Interpretation |
| :--- | :--- | :--- | :--- |
| **In-Distribution (ID)** | -0.0928 | 0.3530 | There is **no statistically significant correlation** between the model's feature embeddings and the species' evolutionary history. |
| **Out-of-Distribution (OOD)**| -0.0309 | 0.7530 | There is **no statistically significant correlation** for the OOD data either. |
