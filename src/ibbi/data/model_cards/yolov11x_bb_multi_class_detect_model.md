---
license: mit
---
# Model Card: yolov11x_bb_multi_class_detect_model

## Model Details
- **Model Name:** `yolov11x_bb_multi_class_detect_model`
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
| **In-Distribution (ID)** | 🟩 0.9002 | Excellent overall accuracy, with strong performance on most trained species. |
| **Out-of-Distribution (OOD)**| 🟥 0.0000 | As expected, the model cannot classify species it has not been trained on. |

<br>

<details>
<summary><b>Click to see Per-Species Performance (ID Dataset)</b></summary>

*The following list is sorted by Average Precision (AP) from lowest to highest to highlight the most challenging species for the model to identify.*

| Species | AP Score |
| :--- | :--- |
| Scolytus_multistriatus | 0.0222 |
| Dendroctonus_rufipennis | 0.1667 |
| Euwallacea_validus | 0.2222 |
| Hylesinus_aculeatus | 0.3626 |
| Ips_grandicollis | 0.4074 |
| Dryocoetes_autographus | 0.5500 |
| Trypodendron_domesticum | 0.6875 |
| Xyleborus_celsus | 0.7069 |
| Orthotomicus_caelatus | 0.7284 |
| Ambrosiodmus_minor | 0.8017 |
| Ambrosiophilus_atratus | 0.9049 |
| Hylurgus_ligniperda | 0.9361 |
| Pityogenes_chalcographus | 0.9367 |
| Xylosandrus_germanus | 0.9600 |
| Ips_typographus | 0.9659 |
| Coccotrypes_dactyliperda | 0.9671 |
| Scolytus_schevyrewi | 0.9672 |
| Ips_sexdentatus | 0.9678 |
| Monarthrum_mali | 0.9690 |
| Dendroctonus_terebrans | 0.9695 |
| Xylosandrus_crassiusculus | 0.9707 |
| Xyleborus_ferrugineus | 0.9718 |
| Xyleborinus_saxesenii | 0.9732 |
| Xylosandrus_compactus | 0.9777 |
| Monarthrum_fasciatum | 0.9791 |
| Euplatypus_compositus | 0.9828 |
| Pagiocerus_frontalis | 0.9829 |
| Xylosandrus_amputatus | 0.9826 |
| Hypothenemus_hampei | 0.9822 |
| Cnestus_mutilatus | 0.9853 |
| Cryptocarenus_heveae | 0.9854 |
| Hylesinus_varius | 0.9859 |
| Ips_calligraphus | 0.9862 |
| Ips_avulsus | 0.9877 |
| Orthotomicus_erosus | 0.9879 |
| Cyclorhipidion_pelliculosum | 0.9886 |
| Dendroctonus_valens | 0.9898 |
| Hylesinus_toranio | 0.9890 |
| Scolytodes_glaber | 0.9896 |
| Hylurgops_palliatus | 0.9901 |
| Pityophthorus_juglandis | 0.9902 |
| Hylastes_porculus | 0.9902 |
| Xyleborus_glabratus | 0.9905 |
| Xyleborus_affinis | 0.9916 |
| Euwallacea_fornicatus | 0.9918 |
| Coccotrypes_carpophagus | 0.9921 |
| Myoplatypus_flavicornis | 0.9947 |
| Ctonoxylon_hagedorn | 0.9958 |
| Euwallacea_perbrevis | 0.9968 |
| Platypus_cylindrus | 0.9979 |
| Phloeosinus_dentatus | 0.9980 |
| Tomicus_destruens | 0.9995 |
| Anisandrus_sayi | 0.9998 |
| Xylosandrus_morigerus | 0.9998 |
| Ips_duplicatus | 0.9998 |
| Pycnarthrum_hispidium | 0.9999 |
| Coptoborus_ricini | 0.9999 |
| Anisandrus_dispar | 0.9581 |
| Ips_acuminatus | 1.0000 |
| Platypus_koryoensis | 1.0000 |
| Hylesinus_crenatus | 1.0000 |
| Hylastes_salebrosus | 1.0000 |

</details>

---
#### Genus-Level Performance
This evaluates the model's ability to identify the genus, a broader taxonomic rank than species.

| Dataset | Genus mAP (0.50 : 0.95) | Notes |
| :--- | :--- | :--- |
| **In-Distribution (ID)** | 🟩 0.9446 | Excellent performance on genera the model was trained to recognize. |
| **Out-of-Distribution (OOD)**| 🟩 0.7741 | Shows strong generalization, successfully classifying many unseen genera with good accuracy. |

<br>

<details>
<summary><b>Click to see Per-Genus Performance (ID and OOD Datasets)</b></summary>

*The following lists are sorted by Average Precision (AP) from lowest to highest to highlight the most challenging genera for the model to identify.*

**In-Distribution (ID) Genus Performance**
| Genus | AP Score |
| :--- | :--- |
| Dryocoetes | 0.7000 |
| Hypothenemus | 0.8866 |
| Trypodendron | 0.8875 |
| Pityogenes | 0.8938 |
| Scolytus | 0.9000 |
| Cryptocarenus | 0.9070 |
| Xylosandrus | 0.9113 |
| Taphrorychus | 0.9126 |
| Coptoborus | 0.9180 |
| Pityophthorus | 0.9213 |
| Xyleborinus | 0.9294 |
| Scolytodes | 0.9302 |
| Coccotrypes | 0.9326 |
| Euwallacea | 0.9415 |
| Pycnarthrum | 0.9467 |
| Pagiocerus | 0.9498 |
| Orthotomicus | 0.9529 |
| Xyleborus | 0.9555 |
| Monarthrum | 0.9565 |
| Ambrosiophilus | 0.9585 |
| Anisandrus | 0.9629 |
| Hylurgus | 0.9697 |
| Cnestus | 0.9735 |
| Ips | 0.9754 |
| Euplatypus | 0.9755 |
| Cyclorhipidion | 0.9773 |
| Phloeosinus | 0.9779 |
| Ctonoxylon | 0.9803 |
| Dendroctonus | 0.9832 |
| Hylesinus | 0.9846 |
| Hylurgops | 0.9865 |
| Platypus | 0.9909 |
| Ambrosiodmus | 0.9909 |
| Hylastes | 0.9922 |
| Myoplatypus | 0.9958 |
| Tomicus | 0.9980 |

<br>

**Out-of-Distribution (OOD) Genus Performance**
| Genus | AP Score |
| :--- | :--- |
| Dactylotrypes | 0.2583 |
| Cryptocarenus | 0.3832 |
| Cryphalus | 0.3869 |
| Pityogenes | 0.4214 |
| Crypturgus | 0.4846 |
| Carphoborus | 0.5185 |
| Hylocurus | 0.5189 |
| Polygraphus | 0.5222 |
| Premnobius | 0.6095 |
| Leptoxyleborus | 0.6294 |
| Dendroctonus | 0.6638 |
| Pycnarthrum | 0.6750 |
| Dinoplatypus | 0.6833 |
| Ernoporus | 0.6859 |
| Cyclorhipidion | 0.7087 |
| Hypothenemus | 0.7114 |
| Crossotarsus | 0.7176 |
| Wallacellus | 0.7400 |
| Pityoborus | 0.7556 |
| Diuncus | 0.7587 |
| Trypodendron | 0.7591 |
| Webbia | 0.7733 |
| Xylocleptes | 0.7872 |
| Dendroterus | 0.7867 |
| Monarthrum | 0.7929 |
| Gnathotrichus | 0.7926 |
| Cnestus | 0.7937 |
| Truncaudum | 0.8121 |
| Scolytus | 0.8172 |
| Hylastes | 0.8218 |
| Ambrosiodmus | 0.8233 |
| Xyloterinus | 0.8393 |
| Chaetoptelius | 0.8382 |
| Xyleborus | 0.8461 |
| Eccoptopterus | 0.8444 |
| Debus | 0.8500 |
| Heteroborips | 0.8533 |
| Xyleborinus | 0.8548 |
| Euwallacea | 0.8608 |
| Ips | 0.8682 |
| Hadrodemius | 0.8688 |
| Microperus | 0.8756 |
| Tomicus | 0.8829 |
| Eidophelus | 0.8838 |
| Coptoborus | 0.8929 |
| Pseudopityophthorus | 0.8970 |
| Metacorthylus | 0.9000 |
| Platypus | 0.9143 |
| Hylurgus | 0.9196 |
| Anisandrus | 0.9160 |
| Procryphalus | 0.9231 |
| Pityophthorus | 0.9401 |
| Dryocoetes | 0.9444 |
| Beaverium | 0.9545 |
| Tricosa | 0.9615 |
| Stegomerus | 0.9842 |
| Pseudowebbia | 0.9917 |
| Cnesinus | 1.0000 |

</details>

---
### Feature Extraction (Embedding Performance)
The quality of the model's learned feature representations (embeddings) is evaluated by how well they group similar species together.

#### Internal Cluster Validation
These metrics measure the quality of the clusters formed by the embeddings without referring to ground-truth labels.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Silhouette Score** | 0.7606 | 0.2305 | Measures how similar an object is to its own cluster compared to others. **Higher is better (closer to 1)**. The ID embeddings form exceptional, well-defined clusters. |
| **Davies-Bouldin Index**| 0.3303 | 0.4092 | Measures the average similarity between each cluster and its most similar one. **Lower is better (closer to 0)**. The ID embeddings show very little overlap. |
| **Calinski-Harabasz Index**| 14511.8 | 948.297 | Measures the ratio of between-cluster dispersion to within-cluster dispersion. **Higher is better**. The ID embeddings form exceptionally dense and well-separated clusters. |

#### External Cluster Validation
These metrics evaluate the clustering performance by comparing the results to the true species labels.

| Metric | ID Score | OOD Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **Adjusted Rand Index (ARI)** | 0.3519 | 0.0071 | Measures the similarity between true and predicted labels, correcting for chance. **Higher is better (closer to 1)**. |
| **Normalized Mutual Info (NMI)** | 0.6865 | 0.3254 | Measures the agreement between the clustering and the true labels. **Higher is better (closer to 1)**. |
| **Cluster Purity** | 0.6646 | 0.1656 | Measures the extent to which clusters contain a single class. **Higher is better (closer to 1)**. |

**Conclusion:** The high external validation scores for the ID dataset show that the model's feature representations are **effective** at separating the different species it was trained on. This is a significant improvement over single-class or zero-shot models.

#### Phylogenetic Correlation (Mantel Test)
This test determines if the model's learned features correlate with the evolutionary relationships (phylogeny) between different bark beetle species.

| Dataset | Mantel R-statistic | p-value | Interpretation |
| :--- | :--- | :--- | :--- |
| **In-Distribution (ID)** | -0.1517 | 0.1860 | There is **no statistically significant correlation** between the model's features and the species' evolutionary history. |
| **Out-of-Distribution (OOD)**| -0.0068 | 0.9310 | There is **no statistically significant correlation** for the OOD data either. |
