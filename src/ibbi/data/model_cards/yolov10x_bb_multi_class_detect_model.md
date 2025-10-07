---
license: mit
---
# Model Card: yolov10x_bb_multi_class_detect_model

## Model Details
- **Model Name:** `yolov10x_bb_multi_class_detect_model`
- **Model Type:** Multi-Class Object Detection and Classifier
- **Description:** This model is designed to detect and classify specific species and genera of bark beetles from images. Unlike single-class models, it has been fine-tuned on a labeled dataset of bark beetle species.

---

## Evaluation Datasets

To understand the model's capabilities, its performance was tested on two different types of datasets:

-   **In-Distribution (ID):** This dataset contains images of species **the model was trained on**. Performance on this dataset shows how well the model identifies familiar species.
-   **Out-of-Distribution (OOD):** This dataset contains images of species that are **intentionally different from the training data**. Performance here tests the model's ability to handle novel species.

---

## Performance

### Object Detection & Classification
The model's performance is measured by its **mean Average Precision (mAP)**. This score reflects the model's accuracy in both locating the beetle (bounding box) and assigning the correct species or genus label.

#### Species-Level Performance
This evaluates the model's ability to identify individual species.

| Dataset | Species mAP (0.50 : 0.95) | Notes |
| :--- | :--- | :--- |
| **In-Distribution (ID)** | 🟩 0.8969 | High overall accuracy, but performance varies significantly for certain species. |
| **Out-of-Distribution (OOD)**| 🟥 0.0000 | As expected, the model cannot classify species it has not been trained on. |

<br>

<details>
<summary><b>Click to see Per-Species Performance (ID Dataset)</b></summary>

*The following list is sorted by Average Precision (AP) from lowest to highest to highlight the most challenging species for the model to identify.*

| Species | AP Score |
| :--- | :--- |
| Dendroctonus_rufipennis | 0.1000 |
| Scolytus_multistriatus | 0.1111 |
| Euwallacea_validus | 0.3778 |
| Dryocoetes_autographus | 0.4447 |
| Orthotomicus_caelatus | 0.4444 |
| Hylesinus_aculeatus | 0.5128 |
| Xyleborus_celsus | 0.5215 |
| Ips_grandicollis | 0.5556 |
| Ambrosiodmus_minor | 0.7273 |
| Trypodendron_domesticum | 0.7359 |
| Pityogenes_chalcographus | 0.9153 |
| Hylurgus_ligniperda | 0.9173 |
| Xylosandrus_germanus | 0.9388 |
| Ambrosiophilus_atratus | 0.9494 |
| Xylosandrus_crassiusculus | 0.9605 |
| Taphrorychus_bicolor | 0.9636 |
| Ips_typographus | 0.9661 |
| Ips_calligraphus | 0.9667 |
| Scolytus_schevyrewi | 0.9672 |
| Dendroctonus_terebrans | 0.9682 |
| Cnestus_mutilatus | 0.9697 |
| Xylosandrus_compactus | 0.9701 |
| Ips_sexdentatus | 0.9716 |
| Monarthrum_mali | 0.9719 |
| Coccotrypes_dactyliperda | 0.9721 |
| Orthotomicus_erosus | 0.9741 |
| Xyleborinus_saxesenii | 0.9765 |
| Anisandrus_dispar | 0.9787 |
| Cryptocarenus_heveae | 0.9794 |
| Xyleborus_ferrugineus | 0.9794 |
| Xylosandrus_amputatus | 0.9798 |
| Hypothenemus_hampei | 0.9800 |
| Monarthrum_fasciatum | 0.9814 |
| Pityophthorus_juglandis | 0.9827 |
| Hylesinus_varius | 0.9840 |
| Dendroctonus_valens | 0.9854 |
| Euplatypus_compositus | 0.9861 |
| Pagiocerus_frontalis | 0.9869 |
| Euwallacea_fornicatus | 0.9870 |
| Scolytodes_glaber | 0.9870 |
| Cyclorhipidion_pelliculosum | 0.9886 |
| Hylurgops_palliatus | 0.9886 |
| Xyleborus_glabratus | 0.9887 |
| Hylesinus_toranio | 0.9890 |
| Ips_avulsus | 0.9893 |
| Ctonoxylon_hagedorn | 0.9914 |
| Xyleborus_affinis | 0.9915 |
| Xylosandrus_morigerus | 0.9919 |
| Hylastes_salebrosus | 0.9921 |
| Euwallacea_perbrevis | 0.9941 |
| Myoplatypus_flavicornis | 0.9944 |
| Ips_acuminatus | 0.9952 |
| Ips_duplicatus | 0.9953 |
| Phloeosinus_dentatus | 0.9957 |
| Coccotrypes_carpophagus | 0.9960 |
| Platypus_cylindrus | 0.9979 |
| Tomicus_destruens | 0.9979 |
| Hylastes_porculus | 0.9987 |
| Pycnarthrum_hispidium | 0.9988 |
| Platypus_koryoensis | 0.9999 |
| Anisandrus_sayi | 0.9994 |
| Coptoborus_ricini | 1.0000 |
| Hylesinus_crenatus | 1.0000 |

</details>

---
#### Genus-Level Performance
This evaluates the model's ability to identify the genus, a broader taxonomic rank than species.

| Dataset | Genus mAP (0.50 : 0.95) | Notes |
| :--- | :--- | :--- |
| **In-Distribution (ID)** | 🟩 0.9458 | Very strong performance at the genus level for familiar genera. |
| **Out-of-Distribution (OOD)**| 🟨 0.6897 | Demonstrates some ability to generalize to novel genera, though with reduced accuracy. |

<br>

<details>
<summary><b>Click to see Per-Genus Performance (ID and OOD Datasets)</b></summary>

*The following lists are sorted by Average Precision (AP) from lowest to highest to highlight the most challenging genera for the model to identify.*

**In-Distribution (ID) Genus Performance**
| Genus | AP Score |
| :--- | :--- |
| Dryocoetes | 0.6167 |
| Trypodendron | 0.7125 |
| Hypothenemus | 0.9000 |
| Pityogenes | 0.9146 |
| Scolytus | 0.9200 |
| Xylosandrus | 0.9238 |
| Orthotomicus | 0.9246 |
| Pityophthorus | 0.9306 |
| Cryptocarenus | 0.9309 |
| Xyleborinus | 0.9361 |
| Scolytodes | 0.9372 |
| Hylurgus | 0.9402 |
| Coccotrypes | 0.9442 |
| Taphrorychus | 0.9468 |
| Coptoborus | 0.9494 |
| Monarthrum | 0.9514 |
| Xyleborus | 0.9624 |
| Euwallacea | 0.9664 |
| Cnestus | 0.9681 |
| Ambrosiodmus | 0.9727 |
| Pycnarthrum | 0.9752 |
| Ips | 0.9764 |
| Dendroctonus | 0.9786 |
| Phloeosinus | 0.9797 |
| Anisandrus | 0.9821 |
| Hylesinus | 0.9866 |
| Euplatypus | 0.9869 |
| Ctonoxylon | 0.9870 |
| Ambrosiophilus | 0.9902 |
| Hylurgops | 0.9908 |
| Hylastes | 0.9909 |
| Platypus | 0.9911 |
| Pagiocerus | 0.9929 |
| Cyclorhipidion | 0.9943 |
| Myoplatypus | 0.9968 |
| Tomicus | 0.9990 |

<br>

**Out-of-Distribution (OOD) Genus Performance**
| Genus | AP Score |
| :--- | :--- |
| Cryphalus | 0.0595 |
| Dendroctonus | 0.3309 |
| Dactylotrypes | 0.3741 |
| Pityogenes | 0.4025 |
| Scolytus | 0.4255 |
| Leptoxyleborus | 0.4529 |
| Pycnarthrum | 0.4583 |
| Xyloterinus | 0.5214 |
| Eidophelus | 0.5282 |
| Hypothenemus | 0.5737 |
| Gnathotrichus | 0.5778 |
| Crypturgus | 0.5846 |
| Polygraphus | 0.6028 |
| Metacorthylus | 0.6071 |
| Carphoborus | 0.6111 |
| Cryptocarenus | 0.6227 |
| Ambrosiodmus | 0.6429 |
| Cnesinus | 0.6462 |
| Diuncus | 0.6500 |
| Monarthrum | 0.6518 |
| Heteroborips | 0.6533 |
| Hadrodemius | 0.6750 |
| Cyclorhipidion | 0.6870 |
| Crossotarsus | 0.6912 |
| Dendroterus | 0.7000 |
| Xyleborus | 0.7085 |
| Beaverium | 0.7182 |
| Truncaudum | 0.7152 |
| Chaetoptelius | 0.7294 |
| Tricosa | 0.7308 |
| Platypus | 0.7357 |
| Dinoplatypus | 0.7444 |
| Procryphalus | 0.7462 |
| Coptoborus | 0.7500 |
| Trypodendron | 0.7534 |
| Ips | 0.7545 |
| Premnobius | 0.7571 |
| Hylastes | 0.7679 |
| Hylocurus | 0.7710 |
| Stegomerus | 0.7842 |
| Wallacellus | 0.7900 |
| Xyleborinus | 0.7952 |
| Cnestus | 0.8000 |
| Eccoptopterus | 0.8000 |
| Microperus | 0.8000 |
| Pityoborus | 0.8000 |
| Euwallacea | 0.8224 |
| Webbia | 0.8333 |
| Anisandrus | 0.8398 |
| Tomicus | 0.8415 |
| Debus | 0.8500 |
| Ernoporus | 0.8746 |
| Dryocoetes | 0.8833 |
| Pseudopityophthorus | 0.9123 |
| Hylurgus | 0.9525 |
| Pityophthorus | 0.9551 |
| Pseudowebbia | 0.9917 |

</details>

---
### Feature Extraction (Embedding Performance)
The quality of the model's learned feature representations (embeddings) is evaluated by how well they group similar species together.

#### Internal Cluster Validation
These metrics measure the quality of the clusters formed by the embeddings without referring to ground-truth labels.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Silhouette Score** | 0.7394 | 0.2165 | Measures how similar an object is to its own cluster compared to others. **Higher is better (closer to 1)**. The ID embeddings form excellent, well-defined clusters. |
| **Davies-Bouldin Index**| 0.3539 | 0.3208 | Measures the average similarity between each cluster and its most similar one. **Lower is better (closer to 0)**. Both show low overlap. |
| **Calinski-Harabasz Index**| 13638.5 | 729.779 | Measures the ratio of between-cluster dispersion to within-cluster dispersion. **Higher is better**. The ID embeddings form exceptionally dense and well-separated clusters. |

#### External Cluster Validation
These metrics evaluate the clustering performance by comparing the results to the true species labels.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Adjusted Rand Index (ARI)** | 0.3987 | 0.0061 | Measures the similarity between true and predicted labels, correcting for chance. **Higher is better (closer to 1)**. |
| **Normalized Mutual Info (NMI)** | 0.6969 | 0.3066 | Measures the agreement between the clustering and the true labels. **Higher is better (closer to 1)**. |
| **Cluster Purity** | 0.6808 | 0.1678 | Measures the extent to which clusters contain a single class. **Higher is better (closer to 1)**. |

**Conclusion:** The high external validation scores for the ID dataset show that the model's feature representations are **effective** at separating the different species it was trained on. This is a significant improvement over single-class or zero-shot models.

#### Phylogenetic Correlation (Mantel Test)
This test determines if the model's learned features correlate with the evolutionary relationships (phylogeny) between different bark beetle species.

| Dataset | Mantel R-statistic | p-value | Interpretation |
| :--- | :--- | :--- | :--- |
| **In-Distribution (ID)** | -0.1989 | 0.1370 | There is **no statistically significant correlation** between the model's features and the species' evolutionary history. |
| **Out-of-Distribution (OOD)**| 0.0810 | 0.2940 | There is **no statistically significant correlation** for the OOD data either. |
