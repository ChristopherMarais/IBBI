---
license: mit
---
# Model Card: yolov9e_bb_multi_class_detect_model

## Model Details
- **Model Name:** `yolov9e_bb_multi_class_detect_model`
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
| **In-Distribution (ID)** | 🟩 0.9017 | Excellent overall accuracy, though a few species prove very difficult to classify. |
| **Out-of-Distribution (OOD)**| 🟥 0.0000 | As expected, the model cannot classify species it has not been trained on. |

<br>

<details>
<summary><b>Click to see Per-Species Performance (ID Dataset)</b></summary>

*The following list is sorted by Average Precision (AP) from lowest to highest to highlight the most challenging species for the model to identify.*

| Species | AP Score |
| :--- | :--- |
| Dendroctonus_rufipennis | 0.0000 |
| Scolytus_multistriatus | 0.1667 |
| Euwallacea_validus | 0.2525 |
| Dryocoetes_autographus | 0.4028 |
| Ips_grandicollis | 0.4074 |
| Hylesinus_aculeatus | 0.4475 |
| Orthotomicus_caelatus | 0.6667 |
| Xyleborus_celsus | 0.7301 |
| Trypodendron_domesticum | 0.7958 |
| Ambrosiodmus_minor | 0.8182 |
| Pityogenes_chalcographus | 0.9162 |
| Hylurgus_ligniperda | 0.9417 |
| Taphrorychus_bicolor | 0.9550 |
| Xylosandrus_crassiusculus | 0.9568 |
| Anisandrus_dispar | 0.9578 |
| Xylosandrus_germanus | 0.9600 |
| Ips_sexdentatus | 0.9654 |
| Ips_calligraphus | 0.9669 |
| Scolytus_schevyrewi | 0.9672 |
| Ips_typographus | 0.9689 |
| Ambrosiophilus_atratus | 0.9689 |
| Monarthrum_mali | 0.9695 |
| Orthotomicus_erosus | 0.9734 |
| Hylastes_porculus | 0.9765 |
| Dendroctonus_terebrans | 0.9774 |
| Xyleborinus_saxesenii | 0.9780 |
| Coccotrypes_dactyliperda | 0.9786 |
| Xylosandrus_compactus | 0.9800 |
| Monarthrum_fasciatum | 0.9804 |
| Pagiocerus_frontalis | 0.9806 |
| Hypothenemus_hampei | 0.9810 |
| Xyleborus_ferrugineus | 0.9829 |
| Cnestus_mutilatus | 0.9843 |
| Hylesinus_varius | 0.9845 |
| Dendroctonus_valens | 0.9880 |
| Xyleborus_glabratus | 0.9888 |
| Cyclorhipidion_pelliculosum | 0.9886 |
| Hylurgops_palliatus | 0.9899 |
| Hylesinus_toranio | 0.9890 |
| Cryptocarenus_heveae | 0.9911 |
| Euwallacea_fornicatus | 0.9918 |
| Scolytodes_glaber | 0.9919 |
| Xyleborus_affinis | 0.9931 |
| Pityophthorus_juglandis | 0.9938 |
| Coccotrypes_carpophagus | 0.9943 |
| Myoplatypus_flavicornis | 0.9947 |
| Ips_avulsus | 0.9947 |
| Ctonoxylon_hagedorn | 0.9958 |
| Phloeosinus_dentatus | 0.9976 |
| Platypus_cylindrus | 0.9979 |
| Euplatypus_compositus | 0.9871 |
| Anisandrus_sayi | 0.9997 |
| Hylesinus_crenatus | 0.9991 |
| Euwallacea_perbrevis | 0.9995 |
| Tomicus_destruens | 0.9997 |
| Coptoborus_ricini | 0.9998 |
| Pycnarthrum_hispidium | 0.9999 |
| Ips_duplicatus | 0.9999 |
| Xylosandrus_morigerus | 0.9999 |
| Ips_acuminatus | 1.0000 |
| Platypus_koryoensis | 1.0000 |
| Hylastes_salebrosus | 1.0000 |
| Xylosandrus_amputatus | 1.0000 |

</details>

---
#### Genus-Level Performance
This evaluates the model's ability to identify the genus, a broader taxonomic rank than species.

| Dataset | Genus mAP (0.50 : 0.95) | Notes |
| :--- | :--- | :--- |
| **In-Distribution (ID)** | 🟩 0.9641 | Outstanding performance on genera the model was trained to recognize. |
| **Out-of-Distribution (OOD)**| 🟩 0.7978 | Excellent generalization, successfully classifying many unseen genera with high accuracy. |

<br>

<details>
<summary><b>Click to see Per-Genus Performance (ID and OOD Datasets)</b></summary>

*The following lists are sorted by Average Precision (AP) from lowest to highest to highlight the most challenging genera for the model to identify.*

**In-Distribution (ID) Genus Performance**
| Genus | AP Score |
| :--- | :--- |
| Dryocoetes | 0.6000 |
| Trypodendron | 0.9125 |
| Hypothenemus | 0.9366 |
| Orthotomicus | 0.9348 |
| Taphrorychus | 0.9450 |
| Cryptocarenus | 0.9506 |
| Xylosandrus | 0.9514 |
| Xyleborinus | 0.9548 |
| Pityogenes | 0.9604 |
| Scolytus | 0.9600 |
| Monarthrum | 0.9635 |
| Scolytodes | 0.9659 |
| Pityophthorus | 0.9690 |
| Coccotrypes | 0.9676 |
| Hylurgus | 0.9697 |
| Coptoborus | 0.9724 |
| Xyleborus | 0.9779 |
| Euwallacea | 0.9804 |
| Cnestus | 0.9819 |
| Ips | 0.9827 |
| Ambrosiophilus | 0.9854 |
| Anisandrus | 0.9856 |
| Pycnarthrum | 0.9864 |
| Phloeosinus | 0.9874 |
| Dendroctonus | 0.9888 |
| Cyclorhipidion | 0.9909 |
| Hylastes | 0.9913 |
| Hylurgops | 0.9913 |
| Euplatypus | 0.9911 |
| Hylesinus | 0.9912 |
| Platypus | 0.9935 |
| Ctonoxylon | 0.9946 |
| Pagiocerus | 0.9960 |
| Myoplatypus | 0.9968 |
| Tomicus | 1.0000 |
| Ambrosiodmus | 1.0000 |

<br>

**Out-of-Distribution (OOD) Genus Performance**
| Genus | AP Score |
| :--- | :--- |
| Dactylotrypes | 0.4089 |
| Pityogenes | 0.4819 |
| Crypturgus | 0.5154 |
| Dendroctonus | 0.5242 |
| Cryphalus | 0.5505 |
| Polygraphus | 0.5694 |
| Dinoplatypus | 0.6500 |
| Cryptocarenus | 0.6293 |
| Dryocoetes | 0.6611 |
| Platypus | 0.6786 |
| Crossotarsus | 0.6882 |
| Beaverium | 0.7091 |
| Cnestus | 0.7188 |
| Leptoxyleborus | 0.7235 |
| Hypothenemus | 0.7570 |
| Procryphalus | 0.7538 |
| Wallacellus | 0.7600 |
| Pityoborus | 0.7611 |
| Dendroterus | 0.7667 |
| Premnobius | 0.7714 |
| Chaetoptelius | 0.7706 |
| Monarthrum | 0.7732 |
| Webbia | 0.7867 |
| Heteroborips | 0.7867 |
| Pycnarthrum | 0.7917 |
| Xylocleptes | 0.7949 |
| Coptoborus | 0.7929 |
| Trypodendron | 0.8068 |
| Anisandrus | 0.8357 |
| Eidophelus | 0.8327 |
| Xyloterinus | 0.8429 |
| Euwallacea | 0.8427 |
| Diuncus | 0.8478 |
| Ambrosiodmus | 0.8526 |
| Xyleborinus | 0.8611 |
| Hylastes | 0.8654 |
| Cyclorhipidion | 0.8652 |
| Truncaudum | 0.8667 |
| Scolytus | 0.8689 |
| Ips | 0.8682 |
| Stegomerus | 0.8737 |
| Carphoborus | 0.8739 |
| Debus | 0.8750 |
| Eccoptopterus | 0.8889 |
| Tricosa | 0.8923 |
| Xyleborus | 0.8982 |
| Gnathotrichus | 0.9111 |
| Tomicus | 0.9211 |
| Cnesinus | 0.9231 |
| Ernoporus | 0.9305 |
| Microperus | 0.9439 |
| Pseudopityophthorus | 0.9584 |
| Hylocurus | 0.9769 |
| Pseudowebbia | 0.9833 |
| Pityophthorus | 0.9873 |
| Hadrodemius | 0.9875 |
| Hylurgus | 0.9865 |

</details>

---
### Feature Extraction (Embedding Performance)
The quality of the model's learned feature representations (embeddings) is evaluated by how well they group similar species together.

#### Internal Cluster Validation
These metrics measure the quality of the clusters formed by the embeddings without referring to ground-truth labels.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Silhouette Score** | 0.7507 | 0.3464 | Measures how similar an object is to its own cluster compared to others. **Higher is better (closer to 1)**. The ID embeddings form excellent, well-defined clusters. |
| **Davies-Bouldin Index**| 0.2533 | 0.3745 | Measures the average similarity between each cluster and its most similar one. **Lower is better (closer to 0)**. The ID embeddings show exceptionally little overlap. |
| **Calinski-Harabasz Index**| 7706.84 | 1086.46 | Measures the ratio of between-cluster dispersion to within-cluster dispersion. **Higher is better**. The ID embeddings form very dense and well-separated clusters. |

#### External Cluster Validation
These metrics evaluate the clustering performance by comparing the results to the true species labels.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Adjusted Rand Index (ARI)** | 0.5725 | 0.0061 | Measures the similarity between true and predicted labels, correcting for chance. **Higher is better (closer to 1)**. |
| **Normalized Mutual Info (NMI)** | 0.7468 | 0.2955 | Measures the agreement between the clustering and the true labels. **Higher is better (closer to 1)**. |
| **Cluster Purity** | 0.7101 | 0.1683 | Measures the extent to which clusters contain a single class. **Higher is better (closer to 1)**. |

**Conclusion:** The exceptionally high external validation scores for the ID dataset show that this model's feature representations are **highly effective** at separating the different species it was trained on.

#### Phylogenetic Correlation (Mantel Test)
This test determines if the model's learned features correlate with the evolutionary relationships (phylogeny) between different bark beetle species.

| Dataset | Mantel R-statistic | p-value | Interpretation |
| :--- | :--- | :--- | :--- |
| **In-Distribution (ID)** | -0.1528 | 0.2110 | There is **no statistically significant correlation** between the model's features and the species' evolutionary history. |
| **Out-of-Distribution (OOD)**| -0.0007 | 0.9890 | There is **no statistically significant correlation** for the OOD data either. |
