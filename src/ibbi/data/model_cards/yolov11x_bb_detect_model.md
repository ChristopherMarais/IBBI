---
license: mit
---
# Model Card: yolov11x_bb_detect_model

## Model Details
- **Model Name:** `yolov11x_bb_detect_model`
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
| **In-Distribution (ID)** | 🟩 0.9485 | Shows excellent detection performance on images similar to its training data. |
| **Out-of-Distribution (OOD)**| 🟦 0.9271 | Retains strong performance on novel species, indicating good generalization. |

<br>

### Feature Extraction (Embedding Performance)
The model can also convert images into numerical representations (embeddings). The quality of these embeddings is evaluated by how well they group similar species together in a feature space.

#### Internal Cluster Validation
These metrics measure the quality of the clusters formed by the embeddings without referring to ground-truth labels. They assess how dense and well-separated the clusters are.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Silhouette Score** | 0.6000 | 0.4412 | Measures how similar an object is to its own cluster compared to others. **Higher is better (closer to 1)**. The ID embeddings form better-defined clusters. |
| **Davies-Bouldin Index**| 0.3823 | 0.2859 | Measures the average similarity between each cluster and its most similar one. **Lower is better (closer to 0)**. The OOD embeddings show less overlap between clusters. |
| **Calinski-Harabasz Index**| 1504.67 | 824.437 | Measures the ratio of between-cluster dispersion to within-cluster dispersion. **Higher is better**. The ID embeddings form denser and more separated clusters. |

#### External Cluster Validation
These metrics evaluate the clustering performance by comparing the results to the true species labels.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Adjusted Rand Index (ARI)** | 0.1131 | 0.0049 | Measures the similarity between true and predicted labels, correcting for chance. **Higher is better (closer to 1)**. |
| **Normalized Mutual Info (NMI)** | 0.4576 | 0.2666 | Measures the agreement between the clustering and the true labels. **Higher is better (closer to 1)**. |
| **Cluster Purity** | 0.3051 | 0.1249 | Measures the extent to which clusters contain a single class. **Higher is better (closer to 1)**. |

**Conclusion:** The external validation scores are low for both datasets, indicating the model's feature representations do **not** effectively separate different species of bark beetles on their own.

#### Phylogenetic Correlation (Mantel Test)
This test determines if the model's learned features correlate with the evolutionary relationships (phylogeny) between different bark beetle species.

-   **Mantel R-statistic:** This value ranges from -1 to 1. A positive value means species that are close in the model's feature space are also close evolutionarily. A value near zero indicates no correlation.
-   **p-value:** Indicates the statistical significance of the result. A p-value below 0.05 typically suggests a significant correlation.

| Dataset | Mantel R-statistic | p-value | Interpretation |
| :--- | :--- | :--- | :--- |
| **In-Distribution (ID)** | 0.0451 | 0.6860 | There is **no statistically significant correlation** between the model's feature embeddings and the species' evolutionary history. |
| **Out-of-Distribution (OOD)**| 0.0631 | 0.4460 | There is **no statistically significant correlation** for the OOD data either. |
