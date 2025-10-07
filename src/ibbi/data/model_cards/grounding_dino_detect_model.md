---
license: mit
---
# Model Card: grounding_dino_detect_model

## Model Details
- **Model Name:** `grounding_dino_detect_model`
- **Model Type:** Zero-Shot Object Detection and Feature Extractor
- **Description:** This is a general-purpose, zero-shot detection model. It was **not trained specifically on bark beetle data**. The performance reported here was achieved by prompting the model to detect the generic class **'insect'** in the images, demonstrating its ability to identify objects without prior explicit training on them.

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
| **In-Distribution (ID)** | 🟨 0.7225 | Achieves respectable performance by detecting the general class 'insect' without any specific fine-tuning. |
| **Out-of-Distribution (OOD)**| 🟨 0.6439 | Performance decreases on OOD data but remains effective for a zero-shot task. |

<br>

### Feature Extraction (Embedding Performance)
The model can also convert images into numerical representations (embeddings). The quality of these embeddings is evaluated by how well they group similar species together in a feature space.

#### Internal Cluster Validation
These metrics measure the quality of the clusters formed by the embeddings without referring to ground-truth labels. They assess how dense and well-separated the clusters are.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Silhouette Score** | 0.6127 | 0.5921 | Measures how similar an object is to its own cluster compared to others. **Higher is better (closer to 1)**. Both datasets form reasonably well-defined clusters. |
| **Davies-Bouldin Index**| 0.3986 | 0.4660 | Measures the average similarity between each cluster and its most similar one. **Lower is better (closer to 0)**. The ID embeddings show less overlap between clusters. |
| **Calinski-Harabasz Index**| 5267.25 | 12726.2 | Measures the ratio of between-cluster dispersion to within-cluster dispersion. **Higher is better**. The OOD embeddings form significantly denser and more separated clusters. |

#### External Cluster Validation
These metrics evaluate the clustering performance by comparing the results to the true species labels.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Adjusted Rand Index (ARI)** | 0.2009 | 0.1056 | Measures the similarity between true and predicted labels, correcting for chance. **Higher is better (closer to 1)**. |
| **Normalized Mutual Info (NMI)** | 0.4928 | 0.4693 | Measures the agreement between the clustering and the true labels. **Higher is better (closer to 1)**. |
| **Cluster Purity** | 0.3023 | 0.2165 | Measures the extent to which clusters contain a single class. **Higher is better (closer to 1)**. |

**Conclusion:** The external validation scores are low for both datasets. As expected from a model detecting a generic 'insect' class, its feature representations do **not** effectively separate different species of bark beetles on their own.

#### Phylogenetic Correlation (Mantel Test)
This test determines if the model's learned features correlate with the evolutionary relationships (phylogeny) between different bark beetle species.

-   **Mantel R-statistic:** This value ranges from -1 to 1. A positive value means species that are close in the model's feature space are also close evolutionarily. A value near zero indicates no correlation.
-   **p-value:** Indicates the statistical significance of the result. A p-value below 0.05 typically suggests a significant correlation.

| Dataset | Mantel R-statistic | p-value | Interpretation |
| :--- | :--- | :--- | :--- |
| **In-Distribution (ID)** | -0.1040 | 0.3740 | There is **no statistically significant correlation** between the model's feature embeddings and the species' evolutionary history. |
| **Out-of-Distribution (OOD)**| -0.0051 | 0.9640 | There is **no statistically significant correlation** for the OOD data either. |
