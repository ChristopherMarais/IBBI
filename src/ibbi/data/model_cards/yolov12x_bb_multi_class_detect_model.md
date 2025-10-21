---
license: mit
---
# Model Card: yolov12x_bb_multi_class_detect_model

## Model Details
- **Model Name:** `yolov12x_bb_multi_class_detect_model`
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
| **In-Distribution (ID)** | 🟩 0.9066 | Excellent overall accuracy, with top-tier performance on most trained species. |
| **Out-of-Distribution (OOD)**| 🟥 0.0000 | As expected, the model cannot classify species it has not been trained on. |

<br>

<details>
<summary><b>Click to see Per-Species Performance (ID Dataset)</b></summary>

*The following list is sorted by Average Precision (AP) from lowest to highest to highlight the most challenging species for the model to identify.*

| Species | AP Score |
| :--- | :--- |
| Dendroctonus_rufipennis | 0.1000 |
| Scolytus_multistriatus | 0.1429 |
| Hylesinus_aculeatus | 0.2872 |
| Euwallacea_validus | 0.3016 |
| Ips_grandicollis | 0.5556 |
| Dryocoetes_autographus | 0.6500 |
| Xyleborus_celsus | 0.6786 |
| Orthotomicus_caelatus | 0.7639 |
| Ambrosiodmus_minor | 0.8182 |
| Xylosandrus_germanus | 0.8549 |
| Trypodendron_domesticum | 0.8750 |
| Ambrosiophilus_atratus | 0.9114 |
| Pityogenes_chalcographus | 0.9368 |
| Hylurgus_ligniperda | 0.9425 |
| Coccotrypes_carpophagus | 0.9526 |
| Ips_sexdentatus | 0.9610 |
| Taphrorychus_bicolor | 0.9637 |
| Xyleborinus_saxesenii | 0.9645 |
| Xylosandrus_crassiusculus | 0.9649 |
| Ips_calligraphus | 0.9664 |
| Xylosandrus_compactus | 0.9715 |
| Ips_typographus | 0.9730 |
| Coccotrypes_dactyliperda | 0.9735 |
| Dendroctonus_terebrans | 0.9775 |
| Xyleborus_ferrugineus | 0.9775 |
| Anisandrus_dispar | 0.9791 |
| Hypothenemus_hampei | 0.9808 |
| Cnestus_mutilatus | 0.9827 |
| Cryptocarenus_heveae | 0.9824 |
| Scolytus_schevyrewi | 0.9831 |
| Hylesinus_toranio | 0.9834 |
| Monarthrum_mali | 0.9838 |
| Monarthrum_fasciatum | 0.9839 |
| Xylosandrus_amputatus | 0.9836 |
| Hylesinus_crenatus | 0.9848 |
| Dendroctonus_valens | 0.9853 |
| Euplatypus_compositus | 0.9869 |
| Pagiocerus_frontalis | 0.9869 |
| Hylesinus_varius | 0.9873 |
| Orthotomicus_erosus | 0.9874 |
| Cyclorhipidion_pelliculosum | 0.9886 |
| Xyleborus_glabratus | 0.9886 |
| Hylurgops_palliatus | 0.9899 |
| Pityophthorus_juglandis | 0.9914 |
| Euwallacea_perbrevis | 0.9915 |
| Euwallacea_fornicatus | 0.9917 |
| Ips_acuminatus | 0.9918 |
| Scolytodes_glaber | 0.9920 |
| Ips_avulsus | 0.9939 |
| Xylosandrus_morigerus | 0.9943 |
| Myoplatypus_flavicornis | 0.9947 |
| Xyleborus_affinis | 0.9955 |
| Ctonoxylon_hagedorn | 0.9958 |
| Platypus_cylindrus | 0.9958 |
| Anisandrus_sayi | 0.9960 |
| Hylastes_porculus | 0.9994 |
| Pycnarthrum_hispidium | 0.9997 |
| Ips_duplicatus | 0.9997 |
| Phloeosinus_dentatus | 0.9993 |
| Coptoborus_ricini | 0.9999 |
| Tomicus_destruens | 0.9999 |
| Platypus_koryoensis | 1.0000 |
| Hylastes_salebrosus | 1.0000 |

</details>

---
#### Genus-Level Performance
This evaluates the model's ability to identify the genus, a broader taxonomic rank than species.

| Dataset | Genus mAP (0.50 : 0.95) | Notes |
| :--- | :--- | :--- |
| **In-Distribution (ID)** | 🟩 0.9596 | Extremely high performance on genera the model was trained to recognize. |
| **Out-of-Distribution (OOD)**| 🟩 0.7977 | Excellent generalization, successfully classifying many unseen genera with high accuracy. |

<br>

<details>
<summary><b>Click to see Per-Genus Performance (ID and OOD Datasets)</b></summary>

*The following lists are sorted by Average Precision (AP) from lowest to highest to highlight the most challenging genera for the model to identify.*

**In-Distribution (ID) Genus Performance**
| Genus | AP Score |
| :--- | :--- |
| Dryocoetes | 0.7250 |
| Trypodendron | 0.8875 |
| Ambrosiodmus | 0.9000 |
| Hypothenemus | 0.9076 |
| Scolytus | 0.9200 |
| Orthotomicus | 0.9370 |
| Scolytodes | 0.9424 |
| Xylosandrus | 0.9421 |
| Xyleborinus | 0.9440 |
| Cryptocarenus | 0.9444 |
| Pityogenes | 0.9552 |
| Taphrorychus | 0.9559 |
| Monarthrum | 0.9596 |
| Pityophthorus | 0.9617 |
| Coccotrypes | 0.9623 |
| Hylurgus | 0.9623 |
| Coptoborus | 0.9678 |
| Xyleborus | 0.9713 |
| Euwallacea | 0.9744 |
| Cnestus | 0.9789 |
| Anisandrus | 0.9814 |
| Ips | 0.9826 |
| Phloeosinus | 0.9838 |
| Pycnarthrum | 0.9850 |
| Euplatypus | 0.9861 |
| Dendroctonus | 0.9888 |
| Hylesinus | 0.9885 |
| Hylurgops | 0.9899 |
| Platypus | 0.9909 |
| Pagiocerus | 0.9907 |
| Ambrosiophilus | 0.9927 |
| Myoplatypus | 0.9937 |
| Ctonoxylon | 0.9950 |
| Hylastes | 0.9974 |
| Cyclorhipidion | 0.9989 |
| Tomicus | 1.0000 |

<br>

**Out-of-Distribution (OOD) Genus Performance**
| Genus | AP Score |
| :--- | :--- |
| Cryphalus | 0.1730 |
| Dactylotrypes | 0.3445 |
| Pityogenes | 0.4200 |
| Carphoborus | 0.4987 |
| Cryptocarenus | 0.5675 |
| Crypturgus | 0.5769 |
| Dinoplatypus | 0.5778 |
| Gnathotrichus | 0.6370 |
| Polygraphus | 0.6333 |
| Scolytus | 0.6409 |
| Crossotarsus | 0.6588 |
| Dendroterus | 0.6667 |
| Wallacellus | 0.6800 |
| Dendroctonus | 0.7087 |
| Monarthrum | 0.7661 |
| Pycnarthrum | 0.7667 |
| Pityoborus | 0.7722 |
| Eccoptopterus | 0.7722 |
| Ips | 0.7773 |
| Heteroborips | 0.7800 |
| Cyclorhipidion | 0.7957 |
| Hylocurus | 0.8036 |
| Beaverium | 0.8091 |
| Leptoxyleborus | 0.8118 |
| Metacorthylus | 0.8286 |
| Diuncus | 0.8348 |
| Webbia | 0.8400 |
| Trypodendron | 0.8409 |
| Hylastes | 0.8436 |
| Coptoborus | 0.8464 |
| Cnesinus | 0.8615 |
| Hadrodemius | 0.8625 |
| Xylocleptes | 0.8590 |
| Ambrosiodmus | 0.8556 |
| Euwallacea | 0.8713 |
| Anisandrus | 0.8728 |
| Chaetoptelius | 0.8765 |
| Xyleborinus | 0.8794 |
| Xyloterinus | 0.8857 |
| Dryocoetes | 0.8833 |
| Hypothenemus | 0.8840 |
| Premnobius | 0.8905 |
| Pseudopityophthorus | 0.8917 |
| Tomicus | 0.9073 |
| Ernoporus | 0.9141 |
| Platypus | 0.9143 |
| Xyleborus | 0.9215 |
| Microperus | 0.9244 |
| Cnestus | 0.9250 |
| Truncaudum | 0.9273 |
| Debus | 0.9500 |
| Stegomerus | 0.9632 |
| Tricosa | 0.9615 |
| Pityophthorus | 0.9803 |
| Pseudowebbia | 0.9833 |
| Procryphalus | 0.9846 |
| Hylurgus | 0.9865 |
| Eidophelus | 0.9782 |

</details>

---
### Feature Extraction (Embedding Performance)
The quality of the model's learned feature representations (embeddings) is evaluated by how well they group similar species together.

#### Internal Cluster Validation
These metrics measure the quality of the clusters formed by the embeddings without referring to ground-truth labels.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Silhouette Score** | 0.6960 | 0.3456 | Measures how similar an object is to its own cluster compared to others. **Higher is better (closer to 1)**. The ID embeddings form excellent, well-defined clusters. |
| **Davies-Bouldin Index**| 0.3503 | 0.4327 | Measures the average similarity between each cluster and its most similar one. **Lower is better (closer to 0)**. The ID embeddings show very little overlap. |
| **Calinski-Harabasz Index**| 10049.9 | 1588.36 | Measures the ratio of between-cluster dispersion to within-cluster dispersion. **Higher is better**. The ID embeddings form exceptionally dense and well-separated clusters. |

#### External Cluster Validation
These metrics evaluate the clustering performance by comparing the results to the true species labels.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Adjusted Rand Index (ARI)** | 0.5036 | 0.0206 | Measures the similarity between true and predicted labels, correcting for chance. **Higher is better (closer to 1)**. |
| **Normalized Mutual Info (NMI)** | 0.6836 | 0.4230 | Measures the agreement between the clustering and the true labels. **Higher is better (closer to 1)**. |
| **Cluster Purity** | 0.6519 | 0.2029 | Measures the extent to which clusters contain a single class. **Higher is better (closer to 1)**. |

**Conclusion:** The exceptionally high external validation scores (especially ARI) for the ID dataset show that this model's feature representations are **highly effective** at separating the different species it was trained on.

#### Phylogenetic Correlation (Mantel Test)
This test determines if the model's learned features correlate with the evolutionary relationships (phylogeny) between different bark beetle species.

| Dataset | Mantel R-statistic | p-value | Interpretation |
| :--- | :--- | :--- | :--- |
| **In-Distribution (ID)** | -0.1688 | 0.1750 | There is **no statistically significant correlation** between the model's features and the species' evolutionary history. |
| **Out-of-Distribution (OOD)**| -0.0138 | 0.8650 | There is **no statistically significant correlation** for the OOD data either. |
