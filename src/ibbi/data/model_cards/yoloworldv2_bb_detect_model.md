---
license: mit
---
# Model Card: yoloworldv2_bb_detect_model

## Model Details
- **Model Name:** `yoloworldv2_bb_detect_model`
- **Model Type:** Zero-Shot Object Detection and Feature Extractor
- **Description:** This is a general-purpose, zero-shot detection model. It was **not trained specifically on bark beetle data**. The performance reported here was achieved by prompting the model to detect the generic class **'insect'** in the images, showcasing its ability to identify objects without direct training.

---

## Evaluation Datasets

To understand the model's capabilities, its performance was tested on two different types of datasets:

-   **In-Distribution (ID):** This dataset contains images that are **similar to the data the fine-tuned models were trained on**.
-   **Out-of-Distribution (OOD):** This dataset contains images that are **intentionally different species from the ID training data**.

---

## Performance

### Object Detection
The model's ability to correctly identify and locate insects is measured by its **mean Average Precision (mAP)**. This metric evaluates both the accuracy of the bounding box placement and the classification confidence. A higher mAP score indicates better performance.

| Dataset | mAP (0.50 : 0.95) | Notes |
| :--- | :--- | :--- |
| **In-Distribution (ID)** | 🟥 0.2056 | The model demonstrates a basic capability for zero-shot detection, but its performance is low. |
| **Out-of-Distribution (OOD)**| 🟥 0.1667 | Performance is further reduced on OOD data, indicating limited generalization for this task. |

<br>

### Feature Extraction (Embedding Performance)
The model can also convert images into numerical representations (embeddings). The quality of these embeddings is evaluated by how well they group similar species together in a feature space.

#### Internal Cluster Validation
These metrics measure the quality of the clusters formed by the embeddings without referring to ground-truth labels. They assess how dense and well-separated the clusters are.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Silhouette Score** | 0.6580 | 0.5230 | Measures how similar an object is to its own cluster compared to others. **Higher is better (closer to 1)**. The ID embeddings form better-defined clusters. |
| **Davies-Bouldin Index**| 0.5033 | 0.4026 | Measures the average similarity between each cluster and its most similar one. **Lower is better (closer to 0)**. The OOD embeddings show less overlap between clusters. |
| **Calinski-Harabasz Index**| 3575.17 | 7332.84 | Measures the ratio of between-cluster dispersion to within-cluster dispersion. **Higher is better**. The OOD embeddings form denser and more separated clusters. |

#### External Cluster Validation
These metrics evaluate the clustering performance by comparing the results to the true species labels.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Adjusted Rand Index (ARI)** | 0.2478 | 0.0814 | Measures the similarity between true and predicted labels, correcting for chance. **Higher is better (closer to 1)**. |
| **Normalized Mutual Info (NMI)** | 0.4893 | 0.4851 | Measures the agreement between the clustering and the true labels. **Higher is better (closer to 1)**. |
| **Cluster Purity** | 0.2757 | 0.2367 | Measures the extent to which clusters contain a single class. **Higher is better (closer to 1)**. |

**Conclusion:** The external validation scores are low for both datasets. As expected from a model detecting a generic 'insect' class, its feature representations do **not** effectively separate different species of bark beetles on their own.

#### Phylogenetic Correlation (Mantel Test)
This test determines if the model's learned features correlate with the evolutionary relationships (phylogeny) between different bark beetle species.

-   **Mantel R-statistic:** This value ranges from -1 to 1. A positive value means species that are close in the model's feature space are also close evolutionarily. A value near zero indicates no correlation.
-   **p-value:** Indicates the statistical significance of the result. A p-value below 0.05 typically suggests a significant correlation.

| Dataset | Mantel R-statistic | p-value | Interpretation |
| :--- | :--- | :--- | :--- |
| **In-Distribution (ID)** | -0.1852 | 0.0540 | There is **no statistically significant correlation** (p > 0.05) between the model's features and evolutionary history, though the result is marginal. |
| **Out-of-Distribution (OOD)**| -0.1737 | 0.0760 | There is **no statistically significant correlation** for the OOD data either. |
