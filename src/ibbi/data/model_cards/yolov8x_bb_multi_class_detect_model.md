---
license: mit
---
# Model Card: yolov8x_bb_multi_class_detect_model

## Model Details
- **Model Name:** `yolov8x_bb_multi_class_detect_model`
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
| **In-Distribution (ID)** | 🟩 0.8951 | High overall accuracy, but performance on a few specific species is low. |
| **Out-of-Distribution (OOD)**| 🟥 0.0000 | As expected, the model cannot classify species it has not been trained on. |

<br>

<details>
<summary><b>Click to see Per-Species Performance (ID Dataset)</b></summary>

*The following list is sorted by Average Precision (AP) from lowest to highest to highlight the most challenging species for the model to identify.*

| Species | AP Score |
| :--- | :--- |
| Dendroctonus_rufipennis | 0.1933 |
| Scolytus_multistriatus | 0.2222 |
| Dryocoetes_autographus | 0.2976 |
| Ips_grandicollis | 0.3333 |
| Euwallacea_validus | 0.3889 |
| Hylesinus_aculeatus | 0.4359 |
| Orthotomicus_caelatus | 0.5556 |
| Xyleborus_celsus | 0.5930 |
| Hypothenemus_hampei | 0.6999 |
| Ambrosiodmus_minor | 0.8182 |
| Trypodendron_domesticum | 0.9077 |
| Pityogenes_chalcographus | 0.9058 |
| Ambrosiophilus_atratus | 0.9043 |
| Hylurgus_ligniperda | 0.9333 |
| Taphrorychus_bicolor | 0.9549 |
| Ips_sexdentatus | 0.9615 |
| Ips_typographus | 0.9654 |
| Scolytus_schevyrewi | 0.9672 |
| Anisandrus_dispar | 0.9686 |
| Xylosandrus_crassiusculus | 0.9722 |
| Pagiocerus_frontalis | 0.9749 |
| Xyleborus_ferrugineus | 0.9751 |
| Dendroctonus_terebrans | 0.9751 |
| Monarthrum_mali | 0.9777 |
| Ips_calligraphus | 0.9798 |
| Orthotomicus_erosus | 0.9806 |
| Cryptocarenus_heveae | 0.9810 |
| Monarthrum_fasciatum | 0.9811 |
| Hylesinus_crenatus | 0.9818 |
| Xylosandrus_compactus | 0.9816 |
| Dendroctonus_valens | 0.9821 |
| Coccotrypes_dactyliperda | 0.9823 |
| Ips_avulsus | 0.9824 |
| Euwallacea_fornicatus | 0.9827 |
| Xylosandrus_amputatus | 0.9836 |
| Xyleborinus_saxesenii | 0.9846 |
| Xyleborus_glabratus | 0.9851 |
| Cnestus_mutilatus | 0.9854 |
| Cyclorhipidion_pelliculosum | 0.9864 |
| Euplatypus_compositus | 0.9872 |
| Hylesinus_toranio | 0.9890 |
| Hylurgops_palliatus | 0.9902 |
| Pityophthorus_juglandis | 0.9914 |
| Xyleborus_affinis | 0.9917 |
| Hylesinus_varius | 0.9921 |
| Myoplatypus_flavicornis | 0.9947 |
| Anisandrus_sayi | 0.9957 |
| Ctonoxylon_hagedorn | 0.9958 |
| Euwallacea_perbrevis | 0.9960 |
| Scolytodes_glaber | 0.9963 |
| Platypus_cylindrus | 0.9979 |
| Phloeosinus_dentatus | 0.9981 |
| Pycnarthrum_hispidium | 0.9998 |
| Hylastes_porculus | 0.9996 |
| Ips_duplicatus | 0.9997 |
| Ips_acuminatus | 0.9999 |
| Coptoborus_ricini | 1.0000 |
| Platypus_koryoensis | 1.0000 |
| Xylosandrus_morigerus | 1.0000 |
| Tomicus_destruens | 1.0000 |
| Hylastes_salebrosus | 1.0000 |
| Coccotrypes_carpophagus | 0.8956 |

</details>

---
#### Genus-Level Performance
This evaluates the model's ability to identify the genus, a broader taxonomic rank than species.

| Dataset | Genus mAP (0.50 : 0.95) | Notes |
| :--- | :--- | :--- |
| **In-Distribution (ID)** | 🟩 0.9646 | Exceptional performance on genera the model was trained to recognize. |
| **Out-of-Distribution (OOD)**| 🟩 0.7911 | Very strong generalization, successfully classifying many unseen genera with high accuracy. |

<br>

<details>
<summary><b>Click to see Per-Genus Performance (ID and OOD Datasets)</b></summary>

*The following lists are sorted by Average Precision (AP) from lowest to highest to highlight the most challenging genera for the model to identify.*

**In-Distribution (ID) Genus Performance**
| Genus | AP Score |
| :--- | :--- |
| Dryocoetes | 0.7750 |
| Hypothenemus | 0.8129 |
| Trypodendron | 0.8812 |
| Cryptocarenus | 0.9476 |
| Orthotomicus | 0.9471 |
| Scolytus | 0.9457 |
| Pityogenes | 0.9500 |
| Hylurgus | 0.9541 |
| Xylosandrus | 0.9554 |
| Xyleborinus | 0.9575 |
| Taphrorychus | 0.9613 |
| Pityophthorus | 0.9652 |
| Scolytodes | 0.9671 |
| Monarthrum | 0.9674 |
| Coptoborus | 0.9678 |
| Coccotrypes | 0.9712 |
| Cnestus | 0.9753 |
| Xyleborus | 0.9760 |
| Euwallacea | 0.9786 |
| Dendroctonus | 0.9817 |
| Ips | 0.9844 |
| Phloeosinus | 0.9848 |
| Anisandrus | 0.9856 |
| Pycnarthrum | 0.9879 |
| Hylurgops | 0.9903 |
| Hylesinus | 0.9939 |
| Platypus | 0.9934 |
| Ctonoxylon | 0.9941 |
| Pagiocerus | 0.9942 |
| Myoplatypus | 0.9968 |
| Tomicus | 0.9990 |
| Ambrosiophilus | 0.9927 |
| Euplatypus | 0.9924 |
| Hylastes | 0.9996 |
| Cyclorhipidion | 0.9989 |
| Ambrosiodmus | 1.0000 |

<br>

**Out-of-Distribution (OOD) Genus Performance**
| Genus | AP Score |
| :--- | :--- |
| Cryphalus | 0.1572 |
| Dactylotrypes | 0.3231 |
| Pityogenes | 0.5115 |
| Scolytus | 0.5880 |
| Platypus | 0.6143 |
| Cryptocarenus | 0.6125 |
| Dendroctonus | 0.6267 |
| Hypothenemus | 0.6354 |
| Polygraphus | 0.6333 |
| Crypturgus | 0.6538 |
| Stegomerus | 0.6632 |
| Wallacellus | 0.6800 |
| Beaverium | 0.7091 |
| Monarthrum | 0.7179 |
| Leptoxyleborus | 0.7294 |
| Dendroterus | 0.7400 |
| Cyclorhipidion | 0.7522 |
| Crossotarsus | 0.7588 |
| Cnesinus | 0.7615 |
| Pycnarthrum | 0.7667 |
| Pityoborus | 0.7667 |
| Dryocoetes | 0.7778 |
| Gnathotrichus | 0.7815 |
| Eccoptopterus | 0.7833 |
| Coptoborus | 0.8143 |
| Ambrosiodmus | 0.8143 |
| Trypodendron | 0.8205 |
| Microperus | 0.8244 |
| Ips | 0.8273 |
| Metacorthylus | 0.8286 |
| Premnobius | 0.8286 |
| Diuncus | 0.8283 |
| Webbia | 0.8333 |
| Dinoplatypus | 0.8389 |
| Cnestus | 0.8438 |
| Xyleborus | 0.8479 |
| Xyloterinus | 0.8500 |
| Hylastes | 0.8667 |
| Chaetoptelius | 0.8676 |
| Euwallacea | 0.8671 |
| Xylocleptes | 0.8692 |
| Xyleborinus | 0.8802 |
| Debus | 0.8821 |
| Eidophelus | 0.8877 |
| Pseudowebbia | 0.8917 |
| Tricosa | 0.9000 |
| Truncaudum | 0.9212 |
| Hylocurus | 0.9201 |
| Anisandrus | 0.9345 |
| Hadrodemius | 0.9375 |
| Ernoporus | 0.9352 |
| Tomicus | 0.9455 |
| Carphoborus | 0.9458 |
| Pseudopityophthorus | 0.9463 |
| Pityophthorus | 0.9850 |
| Hylurgus | 0.9806 |
| Procryphalus | 0.9846 |
| Heteroborips | 0.9933 |

</details>

---
### Feature Extraction (Embedding Performance)
The quality of the model's learned feature representations (embeddings) is evaluated by how well they group similar species together.

#### Internal Cluster Validation
These metrics measure the quality of the clusters formed by the embeddings without referring to ground-truth labels.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Silhouette Score** | 0.7865 | 0.3977 | Measures how similar an object is to its own cluster compared to others. **Higher is better (closer to 1)**. The ID embeddings form exceptional, well-defined clusters. |
| **Davies-Bouldin Index**| 0.3468 | 0.3520 | Measures the average similarity between each cluster and its most similar one. **Lower is better (closer to 0)**. Both embeddings show very little overlap between clusters. |
| **Calinski-Harabasz Index**| 10088.3 | 1129.94 | Measures the ratio of between-cluster dispersion to within-cluster dispersion. **Higher is better**. The ID embeddings form exceptionally dense and well-separated clusters. |

#### External Cluster Validation
These metrics evaluate the clustering performance by comparing the results to the true species labels.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Adjusted Rand Index (ARI)** | 0.5876 | 0.0061 | Measures the similarity between true and predicted labels, correcting for chance. **Higher is better (closer to 1)**. |
| **Normalized Mutual Info (NMI)** | 0.7399 | 0.3029 | Measures the agreement between the clustering and the true labels. **Higher is better (closer to 1)**. |
| **Cluster Purity** | 0.6835 | 0.1658 | Measures the extent to which clusters contain a single class. **Higher is better (closer to 1)**. |

**Conclusion:** The exceptionally high external validation scores (especially ARI and NMI) for the ID dataset show that this model's feature representations are **highly effective** at separating the different species it was trained on.

#### Phylogenetic Correlation (Mantel Test)
This test determines if the model's learned features correlate with the evolutionary relationships (phylogeny) between different bark beetle species.

| Dataset | Mantel R-statistic | p-value | Interpretation |
| :--- | :--- | :--- | :--- |
| **In-Distribution (ID)** | -0.1305 | 0.2170 | There is **no statistically significant correlation** between the model's features and the species' evolutionary history. |
| **Out-of-Distribution (OOD)**| -0.0695 | 0.3260 | There is **no statistically significant correlation** for the OOD data either. |
